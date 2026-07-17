//! Adapter from Codex app-server JSONL to the ACP surface consumed by the TUI.

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    path::PathBuf,
    rc::Rc,
};

use agent_client_protocol as acp;
use anyhow::{Context, Result};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
    sync::{mpsc, oneshot},
};
use xai_acp_lib::AcpGatewaySender;
use xai_grok_shell::sampling::types::{
    REASONING_EFFORT_META_KEY, ReasoningEffort, parse_reasoning_effort_meta,
};

struct CodexRpc {
    outbound: mpsc::UnboundedSender<Value>,
    pending: Rc<RefCell<HashMap<u64, oneshot::Sender<Result<Value>>>>>,
    next_id: Cell<u64>,
    closed: Rc<Cell<bool>>,
}

impl CodexRpc {
    async fn spawn(executable: PathBuf) -> Result<(Self, mpsc::UnboundedReceiver<Value>)> {
        let mut child = Command::new(&executable)
            .arg("app-server")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| {
                format!(
                    "failed to start `{}`; install Codex CLI or pass --codex-bin",
                    executable.display()
                )
            })?;
        let mut stdin = child
            .stdin
            .take()
            .context("Codex app-server stdin unavailable")?;
        let stdout = child
            .stdout
            .take()
            .context("Codex app-server stdout unavailable")?;
        let (outbound, mut outbound_rx) = mpsc::unbounded_channel::<Value>();
        let (inbound_tx, inbound) = mpsc::unbounded_channel();
        let pending: Rc<RefCell<HashMap<u64, oneshot::Sender<Result<Value>>>>> =
            Rc::new(RefCell::new(HashMap::new()));
        let reader_pending = pending.clone();
        let writer_pending = pending.clone();
        let closed = Rc::new(Cell::new(false));
        let writer_closed = closed.clone();
        let reader_closed = closed.clone();

        tokio::task::spawn_local(async move {
            while let Some(message) = outbound_rx.recv().await {
                let mut bytes = match serde_json::to_vec(&message) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        tracing::error!(%error, "failed to encode Codex app-server message");
                        continue;
                    }
                };
                bytes.push(b'\n');
                if stdin.write_all(&bytes).await.is_err() {
                    writer_closed.set(true);
                    fail_pending(&writer_pending, "Codex app-server input closed");
                    break;
                }
            }
        });
        tokio::task::spawn_local(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let value: Value = match serde_json::from_str(&line) {
                    Ok(value) => value,
                    Err(error) => {
                        tracing::warn!(%error, %line, "invalid JSON from Codex app-server");
                        continue;
                    }
                };
                if value.get("method").is_some() {
                    let _ = inbound_tx.send(value);
                } else if let Some(id) = value.get("id").and_then(Value::as_u64) {
                    if let Some(waiter) = reader_pending.borrow_mut().remove(&id) {
                        let result = if let Some(error) = value.get("error") {
                            Err(anyhow::anyhow!("Codex app-server error: {error}"))
                        } else {
                            Ok(value.get("result").cloned().unwrap_or(Value::Null))
                        };
                        let _ = waiter.send(result);
                    }
                }
            }
            reader_closed.set(true);
            fail_pending(&reader_pending, "Codex app-server closed its output");
            drop(child);
        });
        Ok((
            Self {
                outbound,
                pending,
                next_id: Cell::new(1),
                closed,
            },
            inbound,
        ))
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        if self.closed.get() {
            anyhow::bail!("Codex app-server stopped");
        }
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        let (tx, rx) = oneshot::channel();
        self.pending.borrow_mut().insert(id, tx);
        if self
            .outbound
            .send(json!({"id": id, "method": method, "params": params}))
            .is_err()
        {
            self.pending.borrow_mut().remove(&id);
            anyhow::bail!("Codex app-server stopped");
        }
        rx.await.context("Codex app-server stopped")?
    }

    fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.outbound
            .send(json!({"method": method, "params": params}))
            .map_err(|_| anyhow::anyhow!("Codex app-server stopped"))
    }

    fn respond(&self, id: Value, result: Value) -> Result<()> {
        self.outbound
            .send(json!({"id": id, "result": result}))
            .map_err(|_| anyhow::anyhow!("Codex app-server stopped"))
    }

    fn respond_error(&self, id: Value, code: i64, message: impl Into<String>) -> Result<()> {
        self.outbound
            .send(json!({
                "id": id,
                "error": {"code": code, "message": message.into()}
            }))
            .map_err(|_| anyhow::anyhow!("Codex app-server stopped"))
    }
}

fn fail_pending(
    pending: &Rc<RefCell<HashMap<u64, oneshot::Sender<Result<Value>>>>>,
    message: &'static str,
) {
    for (_, waiter) in pending.borrow_mut().drain() {
        let _ = waiter.send(Err(anyhow::anyhow!(message)));
    }
}

/// ACP agent backed by a local `codex app-server` child process.
pub struct CodexAgent {
    rpc: Rc<CodexRpc>,
    client: AcpGatewaySender<acp::AgentSide>,
    models: Rc<RefCell<Vec<acp::ModelInfo>>>,
    selected_models: Rc<RefCell<HashMap<String, CodexModelSelection>>>,
    active_turns: Rc<RefCell<HashMap<String, String>>>,
    completions: Rc<RefCell<HashMap<String, oneshot::Sender<acp::StopReason>>>>,
    login_completions: Rc<RefCell<HashMap<String, oneshot::Sender<Result<()>>>>>,
    authenticated: Cell<bool>,
    openai_docs_mcp: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CodexModelSelection {
    model: String,
    effort: Option<ReasoningEffort>,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexRequestUserInputParams {
    thread_id: String,
    turn_id: String,
    item_id: String,
    questions: Vec<CodexRequestUserInputQuestion>,
    #[serde(default)]
    auto_resolution_ms: Option<u64>,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexRequestUserInputQuestion {
    id: String,
    header: String,
    question: String,
    #[serde(default)]
    options: Option<Vec<CodexRequestUserInputOption>>,
    #[serde(default)]
    is_other: bool,
    #[serde(default)]
    is_secret: bool,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct CodexRequestUserInputOption {
    label: String,
    description: String,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
struct CodexRequestUserInputResponse {
    answers: HashMap<String, CodexRequestUserInputAnswer>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
struct CodexRequestUserInputAnswer {
    answers: Vec<String>,
}

impl CodexAgent {
    pub async fn spawn(
        executable: PathBuf,
        client: AcpGatewaySender<acp::AgentSide>,
        openai_docs_mcp: bool,
    ) -> Result<Self> {
        let (rpc, inbound) = CodexRpc::spawn(executable).await?;
        let rpc = Rc::new(rpc);
        let active_turns = Rc::new(RefCell::new(HashMap::new()));
        let completions = Rc::new(RefCell::new(HashMap::new()));
        let login_completions = Rc::new(RefCell::new(HashMap::new()));
        tokio::task::spawn_local(forward_events(
            inbound,
            rpc.clone(),
            client.clone(),
            active_turns.clone(),
            completions.clone(),
            login_completions.clone(),
        ));
        Ok(Self {
            rpc,
            client,
            models: Rc::new(RefCell::new(Vec::new())),
            selected_models: Rc::new(RefCell::new(HashMap::new())),
            active_turns,
            completions,
            login_completions,
            authenticated: Cell::new(false),
            openai_docs_mcp,
        })
    }

    async fn refresh_models(&self) -> Result<Option<acp::SessionModelState>> {
        let response = self
            .rpc
            .request("model/list", json!({"limit": 100}))
            .await?;
        let models = map_models(&response);
        *self.models.borrow_mut() = models.clone();
        Ok(map_model_state(&response, models))
    }

    fn model_state(
        &self,
        current: String,
        effort: Option<ReasoningEffort>,
    ) -> acp::SessionModelState {
        let mut models = self.models.borrow().clone();
        if let Some(effort) = effort
            && let Some(info) = models
                .iter_mut()
                .find(|info| info.model_id.0.as_ref() == current)
        {
            info.meta
                .get_or_insert_with(serde_json::Map::new)
                .insert(REASONING_EFFORT_META_KEY.into(), json!(effort.as_str()));
        }
        acp::SessionModelState::new(current, models)
    }

    fn record_session_selection(
        &self,
        session_id: String,
        model: String,
        response: &Value,
    ) -> Option<ReasoningEffort> {
        let effort = response
            .get("reasoningEffort")
            .and_then(Value::as_str)
            .and_then(|value| value.parse().ok())
            .or_else(|| default_effort_for_model(&self.models.borrow(), &model));
        self.selected_models
            .borrow_mut()
            .insert(session_id, CodexModelSelection { model, effort });
        effort
    }
}

#[async_trait::async_trait(?Send)]
impl acp::Agent for CodexAgent {
    async fn initialize(
        &self,
        args: acp::InitializeRequest,
    ) -> acp::Result<acp::InitializeResponse> {
        self.rpc
            .request(
                "initialize",
                json!({
                    "clientInfo": {
                        "name": "codex_tui",
                        "title": "Codex TUI",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "capabilities": {"experimentalApi": true}
                }),
            )
            .await
            .map_err(acp_error)?;
        self.rpc
            .notify("initialized", json!({}))
            .map_err(acp_error)?;
        let initial_model_state = self.refresh_models().await.map_err(acp_error)?;
        let account = self
            .rpc
            .request("account/read", json!({}))
            .await
            .map_err(acp_error)?;
        let auth_methods = if account.get("account").is_none_or(Value::is_null) {
            let mut meta = serde_json::Map::new();
            meta.insert("external_provider".into(), Value::Bool(true));
            vec![acp::AuthMethod::Agent(
                acp::AuthMethodAgent::new("oidc", "Sign in with ChatGPT")
                    .description("Use the Codex CLI browser login")
                    .meta(meta),
            )]
        } else {
            self.authenticated.set(true);
            vec![acp::AuthMethod::Agent(acp::AuthMethodAgent::new(
                "cached_token",
                "Codex CLI session",
            ))]
        };
        let mut response = acp::InitializeResponse::new(args.protocol_version)
            .agent_capabilities(acp::AgentCapabilities::new().load_session(true))
            .auth_methods(auth_methods)
            .agent_info(
                acp::Implementation::new("codex", env!("CARGO_PKG_VERSION")).title("Codex"),
            );
        if let Some(model_state) = initial_model_state {
            response = response.meta(json!({"modelState": model_state}).as_object().cloned());
        }
        Ok(response)
    }

    async fn authenticate(
        &self,
        _args: acp::AuthenticateRequest,
    ) -> acp::Result<acp::AuthenticateResponse> {
        if self.authenticated.get() {
            return Ok(acp::AuthenticateResponse::new());
        }
        let response = self
            .rpc
            .request(
                "account/login/start",
                json!({
                    "type": "chatgpt",
                    "appBrand": "codex",
                    "codexStreamlinedLogin": true,
                    "useHostedLoginSuccessPage": true
                }),
            )
            .await
            .map_err(acp_error)?;
        let login_id = response
            .get("loginId")
            .and_then(Value::as_str)
            .ok_or_else(|| acp_error(anyhow::anyhow!("Codex login returned no login id")))?
            .to_owned();
        let auth_url = response
            .get("authUrl")
            .and_then(Value::as_str)
            .ok_or_else(|| acp_error(anyhow::anyhow!("Codex login returned no browser URL")))?;
        let (tx, rx) = oneshot::channel();
        self.login_completions.borrow_mut().insert(login_id, tx);
        webbrowser::open(auth_url)
            .map_err(|error| acp_error(anyhow::anyhow!("failed to open login browser: {error}")))?;
        rx.await
            .map_err(|_| acp_error(anyhow::anyhow!("Codex login was interrupted")))?
            .map_err(acp_error)?;
        self.authenticated.set(true);
        Ok(acp::AuthenticateResponse::new())
    }

    async fn new_session(
        &self,
        args: acp::NewSessionRequest,
    ) -> acp::Result<acp::NewSessionResponse> {
        let requested_model = args
            .meta
            .as_ref()
            .and_then(|meta| meta.get("modelId"))
            .and_then(Value::as_str);
        let response = self
            .rpc
            .request(
                "thread/start",
                new_thread_params(args.cwd, self.openai_docs_mcp, requested_model),
            )
            .await
            .map_err(acp_error)?;
        let thread_id = response
            .pointer("/thread/id")
            .and_then(Value::as_str)
            .ok_or_else(|| acp_error(anyhow::anyhow!("thread/start returned no thread id")))?
            .to_owned();
        let model = response
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                self.models
                    .borrow()
                    .first()
                    .map(|model| model.model_id.0.to_string())
            })
            .unwrap_or_else(|| "default".to_owned());
        let effort = self.record_session_selection(thread_id.clone(), model.clone(), &response);
        notify_session_startup_complete(&self.client, &thread_id);
        Ok(acp::NewSessionResponse::new(thread_id)
            .modes(codex_session_modes("default"))
            .models(self.model_state(model, effort)))
    }

    async fn load_session(
        &self,
        args: acp::LoadSessionRequest,
    ) -> acp::Result<acp::LoadSessionResponse> {
        let session_id = args.session_id.0.to_string();
        let mut params = json!({
            "threadId": session_id,
            "cwd": args.cwd
        });
        apply_thread_config(&mut params, self.openai_docs_mcp);
        let response = self
            .rpc
            .request("thread/resume", params)
            .await
            .map_err(acp_error)?;
        let model = response
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("default")
            .to_owned();
        let effort = self.record_session_selection(session_id.clone(), model.clone(), &response);
        replay_thread(&self.client, &response).await;
        notify_session_startup_complete(&self.client, &session_id);
        Ok(acp::LoadSessionResponse::new()
            .modes(codex_session_modes("default"))
            .models(self.model_state(model, effort)))
    }

    async fn prompt(&self, args: acp::PromptRequest) -> acp::Result<acp::PromptResponse> {
        let thread_id = args.session_id.0.to_string();
        let prompt_id = args
            .meta
            .as_ref()
            .and_then(|meta| meta.get("promptId"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let input = prompt_to_codex_input(&args.prompt);
        let selection = self.selected_models.borrow().get(&thread_id).cloned();
        let model = selection.as_ref().map(|selection| selection.model.clone());
        let effort = selection
            .as_ref()
            .and_then(|selection| selection.effort)
            .map(ReasoningEffort::as_str);
        let response = self
            .rpc
            .request(
                "turn/start",
                turn_start_params(&thread_id, input, model, effort, prompt_id.as_deref()),
            )
            .await
            .map_err(acp_error)?;
        let turn_id = response
            .pointer("/turn/id")
            .and_then(Value::as_str)
            .ok_or_else(|| acp_error(anyhow::anyhow!("turn/start returned no turn id")))?
            .to_owned();
        self.active_turns
            .borrow_mut()
            .insert(thread_id, turn_id.clone());
        let (tx, rx) = oneshot::channel();
        self.completions.borrow_mut().insert(turn_id, tx);
        let reason = rx.await.unwrap_or(acp::StopReason::EndTurn);
        Ok(
            acp::PromptResponse::new(reason).meta(prompt_id.map(|prompt_id| {
                json!({"promptId": prompt_id})
                    .as_object()
                    .expect("prompt response metadata is an object")
                    .clone()
            })),
        )
    }

    async fn cancel(&self, args: acp::CancelNotification) -> acp::Result<()> {
        let thread_id = args.session_id.0.to_string();
        let turn_id = self.active_turns.borrow().get(&thread_id).cloned();
        if let Some(turn_id) = turn_id {
            self.rpc
                .request(
                    "turn/interrupt",
                    json!({
                        "threadId": thread_id,
                        "turnId": turn_id
                    }),
                )
                .await
                .map_err(acp_error)?;
        }
        Ok(())
    }

    async fn set_session_mode(
        &self,
        args: acp::SetSessionModeRequest,
    ) -> acp::Result<acp::SetSessionModeResponse> {
        let mode = match args.mode_id.0.as_ref() {
            "plan" => "plan",
            "default" => "default",
            unsupported => {
                return Err(acp::Error::invalid_params()
                    .data(format!("unsupported Codex session mode: {unsupported}")));
            }
        };
        let thread_id = args.session_id.0.to_string();
        let selection = self
            .selected_models
            .borrow()
            .get(&thread_id)
            .cloned()
            .ok_or_else(|| {
                acp::Error::invalid_params().data(format!("unknown Codex session id: {thread_id}"))
            })?;
        self.rpc
            .request(
                "thread/settings/update",
                json!({
                    "threadId": thread_id,
                    "collaborationMode": {
                        "mode": mode,
                        "settings": {
                            "model": selection.model,
                            "reasoning_effort": selection.effort.map(ReasoningEffort::as_str),
                            "developer_instructions": null
                        }
                    }
                }),
            )
            .await
            .map_err(acp_error)?;
        self.client
            .forward_fire_and_forget(acp::SessionNotification::new(
                args.session_id,
                acp::SessionUpdate::CurrentModeUpdate(acp::CurrentModeUpdate::new(args.mode_id)),
            ));
        Ok(acp::SetSessionModeResponse::new())
    }

    async fn set_session_model(
        &self,
        args: acp::SetSessionModelRequest,
    ) -> acp::Result<acp::SetSessionModelResponse> {
        let model = args.model_id.0.to_string();
        let effort = parse_reasoning_effort_meta(args.meta.as_ref())
            .or_else(|| default_effort_for_model(&self.models.borrow(), &model));
        self.selected_models.borrow_mut().insert(
            args.session_id.0.to_string(),
            CodexModelSelection { model, effort },
        );
        Ok(acp::SetSessionModelResponse::new())
    }

    async fn ext_method(&self, args: acp::ExtRequest) -> acp::Result<acp::ExtResponse> {
        if args.method.as_ref() != "x.ai/session/list" {
            return Err(acp::Error::method_not_found());
        }
        let params: Value = serde_json::from_str(args.params.get()).unwrap_or_default();
        let response = self
            .rpc
            .request(
                "thread/list",
                json!({
                    "cwd": params.get("cwd").cloned(),
                    "limit": params.get("limit").and_then(Value::as_u64).unwrap_or(30),
                    "searchTerm": params.get("query").cloned(),
                    "sortKey": "updated_at",
                    "sortDirection": "desc"
                }),
            )
            .await
            .map_err(acp_error)?;
        let sessions: Vec<Value> = response.get("data").and_then(Value::as_array)
            .into_iter().flatten().map(|thread| {
                let created = thread.get("createdAt").and_then(Value::as_i64).unwrap_or_default();
                let updated = thread.get("updatedAt").and_then(Value::as_i64).unwrap_or(created);
                json!({
                    "sessionId": thread.get("id").cloned().unwrap_or(Value::Null),
                    "summary": thread.get("preview").cloned().unwrap_or(Value::String("Codex session".into())),
                    "firstPrompt": thread.get("preview").cloned(),
                    "cwd": thread.get("cwd").cloned().unwrap_or(Value::Null),
                    "createdAt": timestamp(created),
                    "updatedAt": timestamp(updated),
                    "source": "codex"
                })
            }).collect();
        let value = json!({"sessions": sessions});
        let raw =
            serde_json::value::to_raw_value(&value).map_err(|error| acp_error(error.into()))?;
        Ok(acp::ExtResponse::new(raw.into()))
    }

    async fn ext_notification(&self, _args: acp::ExtNotification) -> acp::Result<()> {
        Ok(())
    }
}

async fn replay_thread(client: &AcpGatewaySender<acp::AgentSide>, response: &Value) {
    use acp::Client as _;
    let Some(thread) = response.get("thread") else {
        return;
    };
    let session_id = thread
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let turns = thread
        .get("turns")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for turn in turns {
        for item in turn
            .get("items")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let update = match item.get("type").and_then(Value::as_str).unwrap_or_default() {
                "userMessage" => {
                    let text = item
                        .get("content")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(|input| input.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n");
                    Some(acp::SessionUpdate::UserMessageChunk(
                        acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(
                            text,
                        ))),
                    ))
                }
                "agentMessage" => item.get("text").and_then(Value::as_str).map(|text| {
                    acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                        acp::ContentBlock::Text(acp::TextContent::new(text.to_owned())),
                    ))
                }),
                "reasoning" => {
                    let text = item
                        .get("summary")
                        .and_then(Value::as_array)
                        .filter(|parts| !parts.is_empty())
                        .or_else(|| item.get("content").and_then(Value::as_array))
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join("\n");
                    (!text.is_empty()).then(|| {
                        acp::SessionUpdate::AgentThoughtChunk(acp::ContentChunk::new(
                            acp::ContentBlock::Text(acp::TextContent::new(text)),
                        ))
                    })
                }
                _ => item_as_tool(Some(item), true).map(acp::SessionUpdate::ToolCall),
            };
            if let Some(update) = update {
                let _ = client
                    .session_notification(acp::SessionNotification::new(session_id.clone(), update))
                    .await;
            }
        }
    }
}

fn timestamp(seconds: i64) -> String {
    chrono::DateTime::from_timestamp(seconds, 0)
        .unwrap_or(chrono::DateTime::<chrono::Utc>::UNIX_EPOCH)
        .to_rfc3339()
}

fn codex_question_ext_request(
    request: &CodexRequestUserInputParams,
) -> xai_grok_tools::implementations::grok_build::ask_user_question::AskUserQuestionExtRequest {
    use xai_grok_tools::implementations::grok_build::ask_user_question::{
        AskUserQuestionExtRequest, AskUserQuestionMode, Question, QuestionOption,
    };

    let questions = request
        .questions
        .iter()
        .map(|question| Question {
            question: codex_question_display_text(question),
            options: question
                .options
                .clone()
                .unwrap_or_default()
                .into_iter()
                .map(|option| QuestionOption {
                    label: option.label,
                    description: option.description,
                    preview: None,
                    id: None,
                })
                .collect(),
            multi_select: Some(false),
            id: Some(question.id.clone()),
        })
        .collect();

    AskUserQuestionExtRequest {
        session_id: request.thread_id.clone(),
        tool_call_id: request.item_id.clone(),
        questions,
        mode: AskUserQuestionMode::Default,
    }
}

fn codex_question_display_text(question: &CodexRequestUserInputQuestion) -> String {
    if question.header.trim().is_empty() {
        question.question.clone()
    } else {
        format!("{}\n\n{}", question.header, question.question)
    }
}

fn codex_user_input_result(
    request: &CodexRequestUserInputParams,
    response: xai_grok_tools::implementations::grok_build::ask_user_question::AskUserQuestionExtResponse,
) -> CodexRequestUserInputResponse {
    use xai_grok_tools::implementations::grok_build::ask_user_question::AskUserQuestionExtResponse;

    let AskUserQuestionExtResponse::Accepted {
        answers,
        annotations,
    } = response
    else {
        return CodexRequestUserInputResponse {
            answers: HashMap::new(),
        };
    };

    let mut codex_answers = HashMap::new();
    for question in &request.questions {
        let display_text = codex_question_display_text(question);
        let answer_key = [&question.id, &question.question, &display_text]
            .into_iter()
            .find(|key| answers.contains_key(key.as_str()));
        let Some(answer_key) = answer_key else {
            continue;
        };
        let selected = &answers[answer_key.as_str()];
        let note = annotations
            .as_ref()
            .and_then(|all| all.get(answer_key.as_str()))
            .and_then(|annotation| annotation.notes.as_deref())
            .filter(|note| !note.trim().is_empty());

        let mut values = selected.clone();
        if let Some(note) = note {
            if let Some(other) = values.iter_mut().find(|value| value.as_str() == "Other") {
                *other = note.to_owned();
            } else {
                values.push(note.to_owned());
            }
        }
        codex_answers.insert(
            question.id.clone(),
            CodexRequestUserInputAnswer { answers: values },
        );
    }
    CodexRequestUserInputResponse {
        answers: codex_answers,
    }
}

fn notify_codex_question_pending(
    client: &AcpGatewaySender<acp::AgentSide>,
    request: &CodexRequestUserInputParams,
) {
    let params = serde_json::value::to_raw_value(&json!({
        "sessionId": request.thread_id,
        "update": {
            "sessionUpdate": "pending_interaction",
            "tool_call_id": request.item_id,
            "kind": "question",
        }
    }))
    .expect("static pending-interaction payload must serialize");
    client.forward_fire_and_forget(acp::ExtNotification::new(
        "x.ai/session_notification",
        params.into(),
    ));
}

fn notify_codex_question_resolved(
    client: &AcpGatewaySender<acp::AgentSide>,
    request: &CodexRequestUserInputParams,
) {
    let params = serde_json::value::to_raw_value(&json!({
        "sessionId": request.thread_id,
        "update": {
            "sessionUpdate": "interaction_resolved",
            "tool_call_id": request.item_id,
        }
    }))
    .expect("static interaction-resolved payload must serialize");
    client.forward_fire_and_forget(acp::ExtNotification::new(
        "x.ai/session_notification",
        params.into(),
    ));
}

/// Brackets a Codex question with the same lifecycle notifications emitted by
/// Grok's pending-interaction registry. Dropping the future on cancellation
/// also resolves the interaction, so external subscribers cannot remain stuck.
struct CodexPendingQuestion<'a> {
    client: &'a AcpGatewaySender<acp::AgentSide>,
    request: &'a CodexRequestUserInputParams,
}

impl<'a> CodexPendingQuestion<'a> {
    fn new(
        client: &'a AcpGatewaySender<acp::AgentSide>,
        request: &'a CodexRequestUserInputParams,
    ) -> Self {
        notify_codex_question_pending(client, request);
        Self { client, request }
    }
}

impl Drop for CodexPendingQuestion<'_> {
    fn drop(&mut self) {
        notify_codex_question_resolved(self.client, self.request);
    }
}

async fn request_codex_user_input(
    client: &AcpGatewaySender<acp::AgentSide>,
    request: &CodexRequestUserInputParams,
) -> Result<CodexRequestUserInputResponse> {
    use acp::Client as _;
    use xai_grok_tools::implementations::grok_build::ask_user_question::AskUserQuestionExtResponse;

    validate_codex_user_input(request)?;

    let ext_request = acp::ExtRequest::new(
        "x.ai/ask_user_question",
        serde_json::value::to_raw_value(&codex_question_ext_request(request))
            .expect("Codex question conversion must serialize")
            .into(),
    );
    let _pending_question = CodexPendingQuestion::new(client, request);
    let response = if let Some(timeout_ms) = request.auto_resolution_ms {
        match tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            client.ext_method(ext_request),
        )
        .await
        {
            Ok(response) => response,
            Err(_) => {
                return Ok(CodexRequestUserInputResponse {
                    answers: HashMap::new(),
                });
            }
        }
    } else {
        client.ext_method(ext_request).await
    }?;
    let response: AskUserQuestionExtResponse = serde_json::from_str(response.0.get())
        .context("pager returned an invalid ask_user_question response")?;
    Ok(codex_user_input_result(request, response))
}

fn validate_codex_user_input(request: &CodexRequestUserInputParams) -> Result<()> {
    if request.thread_id.trim().is_empty()
        || request.turn_id.trim().is_empty()
        || request.item_id.trim().is_empty()
    {
        anyhow::bail!("Codex request_user_input contained an empty routing id");
    }
    if request.questions.is_empty() {
        anyhow::bail!("Codex request_user_input contained no questions");
    }
    if request.questions.iter().any(|question| question.is_secret) {
        anyhow::bail!("secret request_user_input questions are not supported by Codex TUI");
    }
    if request.questions.iter().any(|question| !question.is_other) {
        anyhow::bail!("request_user_input without the free-form Other option is not supported");
    }
    if request.questions.iter().any(|question| {
        question
            .options
            .as_ref()
            .is_none_or(|options| options.is_empty())
    }) {
        anyhow::bail!("Codex request_user_input requires options for every question");
    }
    Ok(())
}

async fn forward_events(
    mut inbound: mpsc::UnboundedReceiver<Value>,
    rpc: Rc<CodexRpc>,
    client: AcpGatewaySender<acp::AgentSide>,
    active_turns: Rc<RefCell<HashMap<String, String>>>,
    completions: Rc<RefCell<HashMap<String, oneshot::Sender<acp::StopReason>>>>,
    login_completions: Rc<RefCell<HashMap<String, oneshot::Sender<Result<()>>>>>,
) {
    use acp::Client as _;
    while let Some(message) = inbound.recv().await {
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        let thread_id = params
            .get("threadId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        match method {
            "item/agentMessage/delta" => {
                if let Some(delta) = params.get("delta").and_then(Value::as_str) {
                    let update = acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                        acp::ContentBlock::Text(acp::TextContent::new(delta)),
                    ));
                    let _ = client
                        .session_notification(acp::SessionNotification::new(
                            thread_id.clone(),
                            update,
                        ))
                        .await;
                }
            }
            "item/reasoning/summaryTextDelta" | "item/reasoning/textDelta" => {
                if let Some(delta) = params.get("delta").and_then(Value::as_str) {
                    let update = acp::SessionUpdate::AgentThoughtChunk(acp::ContentChunk::new(
                        acp::ContentBlock::Text(acp::TextContent::new(delta)),
                    ));
                    let _ = client
                        .session_notification(acp::SessionNotification::new(
                            thread_id.clone(),
                            update,
                        ))
                        .await;
                }
            }
            "item/started" => {
                if let Some(tool) = item_as_tool(params.get("item"), false) {
                    let _ = client
                        .session_notification(acp::SessionNotification::new(
                            thread_id.clone(),
                            acp::SessionUpdate::ToolCall(tool),
                        ))
                        .await;
                }
            }
            "item/completed" => {
                if let Some(item) = params.get("item")
                    && let Some(id) = item.get("id").and_then(Value::as_str)
                    && item_as_tool(Some(item), true).is_some()
                {
                    let fields = acp::ToolCallUpdateFields::new()
                        .status(acp::ToolCallStatus::Completed)
                        .raw_output(item.clone());
                    let update = acp::ToolCallUpdate::new(id.to_owned(), fields);
                    let _ = client
                        .session_notification(acp::SessionNotification::new(
                            thread_id.clone(),
                            acp::SessionUpdate::ToolCallUpdate(update),
                        ))
                        .await;
                }
            }
            "item/commandExecution/outputDelta" | "item/fileChange/outputDelta" => {
                if let Some(item_id) = params.get("itemId").and_then(Value::as_str) {
                    let output = params
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let fields = acp::ToolCallUpdateFields::new().content(vec![
                        acp::ContentBlock::Text(acp::TextContent::new(output)).into(),
                    ]);
                    let update = acp::ToolCallUpdate::new(item_id.to_owned(), fields);
                    let _ = client
                        .session_notification(acp::SessionNotification::new(
                            thread_id.clone(),
                            acp::SessionUpdate::ToolCallUpdate(update),
                        ))
                        .await;
                }
            }
            "item/commandExecution/requestApproval" | "item/fileChange/requestApproval" => {
                if let Some(id) = message.get("id").cloned() {
                    let item_id = params
                        .get("itemId")
                        .and_then(Value::as_str)
                        .unwrap_or("approval");
                    let title = params
                        .get("command")
                        .and_then(Value::as_str)
                        .or_else(|| params.get("reason").and_then(Value::as_str))
                        .unwrap_or("Codex requests permission");
                    let tool = acp::ToolCallUpdate::new(
                        item_id.to_owned(),
                        acp::ToolCallUpdateFields::new().title(title.to_owned()),
                    );
                    let options = vec![
                        acp::PermissionOption::new(
                            "accept",
                            "Allow once",
                            acp::PermissionOptionKind::AllowOnce,
                        ),
                        acp::PermissionOption::new(
                            "acceptForSession",
                            "Allow for session",
                            acp::PermissionOptionKind::AllowAlways,
                        ),
                        acp::PermissionOption::new(
                            "decline",
                            "Deny",
                            acp::PermissionOptionKind::RejectOnce,
                        ),
                    ];
                    let request =
                        acp::RequestPermissionRequest::new(thread_id.clone(), tool, options);
                    let decision = match client.request_permission(request).await {
                        Ok(response) => match response.outcome {
                            acp::RequestPermissionOutcome::Selected(selected) => {
                                selected.option_id.0.to_string()
                            }
                            acp::RequestPermissionOutcome::Cancelled => "cancel".to_owned(),
                            _ => "decline".to_owned(),
                        },
                        Err(_) => "decline".to_owned(),
                    };
                    let _ = rpc.respond(id, json!({"decision": decision}));
                }
            }
            "item/tool/requestUserInput" => {
                let Some(id) = message.get("id").cloned() else {
                    tracing::warn!(?params, "Codex request_user_input had no request id");
                    continue;
                };
                match serde_json::from_value::<CodexRequestUserInputParams>(params.clone()) {
                    Ok(request) => {
                        if let Err(error) = validate_codex_user_input(&request) {
                            let _ = rpc.respond_error(id, -32602, error.to_string());
                            continue;
                        }
                        let client = client.clone();
                        let rpc = rpc.clone();
                        tokio::task::spawn_local(async move {
                            match request_codex_user_input(&client, &request).await {
                                Ok(result) => {
                                    let result = serde_json::to_value(result)
                                        .expect("Codex request_user_input response must serialize");
                                    let _ = rpc.respond(id, result);
                                }
                                Err(error) => {
                                    let _ = rpc.respond_error(id, -32603, error.to_string());
                                }
                            }
                        });
                    }
                    Err(error) => {
                        let _ = rpc.respond_error(
                            id,
                            -32602,
                            format!("invalid Codex request_user_input params: {error}"),
                        );
                    }
                }
            }
            "turn/completed" => {
                let turn_id = params
                    .pointer("/turn/id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let status = params
                    .pointer("/turn/status")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let reason = if status == "interrupted" {
                    acp::StopReason::Cancelled
                } else {
                    acp::StopReason::EndTurn
                };
                if let Some(waiter) = completions.borrow_mut().remove(turn_id) {
                    let _ = waiter.send(reason);
                }
                active_turns
                    .borrow_mut()
                    .retain(|_, active| active != turn_id);
            }
            "account/login/completed" => {
                if let Some(login_id) = params.get("loginId").and_then(Value::as_str)
                    && let Some(waiter) = login_completions.borrow_mut().remove(login_id)
                {
                    let result = if params.get("success").and_then(Value::as_bool) == Some(true) {
                        Ok(())
                    } else {
                        Err(anyhow::anyhow!(
                            "Codex login failed: {}",
                            params
                                .get("error")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown error")
                        ))
                    };
                    let _ = waiter.send(result);
                }
            }
            "error" => {
                tracing::error!(?params, "Codex app-server error");
                let message = params
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Codex app-server reported an error");
                let update = acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                    acp::ContentBlock::Text(acp::TextContent::new(format!(
                        "Codex error: {message}"
                    ))),
                ));
                let _ = client
                    .session_notification(acp::SessionNotification::new(thread_id.clone(), update))
                    .await;
            }
            _ => tracing::trace!(method, ?params, "unmapped Codex app-server event"),
        }
    }
}

fn item_as_tool(item: Option<&Value>, completed: bool) -> Option<acp::ToolCall> {
    let item = item?;
    let item_type = item.get("type").and_then(Value::as_str)?;
    let id = item.get("id").and_then(Value::as_str)?;
    let (kind, title) = match item_type {
        "commandExecution" => (
            acp::ToolKind::Execute,
            item.get("command")
                .and_then(Value::as_str)
                .unwrap_or("Run command"),
        ),
        "fileChange" => (acp::ToolKind::Edit, "Edit files"),
        "mcpToolCall" => (
            acp::ToolKind::Other,
            item.get("tool")
                .and_then(Value::as_str)
                .unwrap_or("MCP tool"),
        ),
        "webSearch" => (acp::ToolKind::Search, "Search the web"),
        _ => return None,
    };
    Some(
        acp::ToolCall::new(id.to_owned(), title.to_owned())
            .kind(kind)
            .status(if completed {
                acp::ToolCallStatus::Completed
            } else {
                acp::ToolCallStatus::InProgress
            })
            .raw_input(item.clone()),
    )
}

fn prompt_to_codex_input(prompt: &[acp::ContentBlock]) -> Vec<Value> {
    prompt
        .iter()
        .filter_map(|block| match block {
            acp::ContentBlock::Text(text) => Some(json!({"type": "text", "text": text.text})),
            acp::ContentBlock::Image(image) => Some(json!({
                "type": "image",
                "url": image.uri.clone().unwrap_or_else(|| {
                    format!("data:{};base64,{}", image.mime_type, image.data)
                })
            })),
            _ => None,
        })
        .collect()
}

fn map_models(response: &Value) -> Vec<acp::ModelInfo> {
    response
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|model| {
            let id = model_id(model)?;
            let name = model
                .get("displayName")
                .or_else(|| model.get("name"))
                .and_then(Value::as_str)
                .unwrap_or(id);
            let default_effort = model.get("defaultReasoningEffort").and_then(Value::as_str);
            let effort_options: Vec<Value> = model
                .get("supportedReasoningEfforts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|option| {
                    let effort = option.get("reasoningEffort")?.as_str()?;
                    Some(json!({
                        "id": effort,
                        "value": effort,
                        "label": humanize_effort(effort),
                        "description": option.get("description").and_then(Value::as_str),
                        "default": default_effort == Some(effort),
                    }))
                })
                .collect();
            let mut meta = serde_json::Map::new();
            if !effort_options.is_empty() {
                meta.insert("supportsReasoningEffort".into(), Value::Bool(true));
                meta.insert("reasoningEfforts".into(), Value::Array(effort_options));
            }
            if let Some(default_effort) = default_effort {
                meta.insert(REASONING_EFFORT_META_KEY.into(), json!(default_effort));
            }
            if let Some(modalities) = model.get("inputModalities") {
                meta.insert("inputModalities".into(), modalities.clone());
            }
            let mut info = acp::ModelInfo::new(id.to_owned(), name.to_owned())
                .meta((!meta.is_empty()).then_some(meta));
            if let Some(description) = model.get("description").and_then(Value::as_str) {
                info = info.description(description.to_owned());
            }
            Some(info)
        })
        .collect()
}

fn map_model_state(
    response: &Value,
    models: Vec<acp::ModelInfo>,
) -> Option<acp::SessionModelState> {
    let default_model = response
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|model| model.get("isDefault").and_then(Value::as_bool) == Some(true))
        .and_then(model_id)
        .map(str::to_owned)
        .or_else(|| models.first().map(|model| model.model_id.0.to_string()))?;
    Some(acp::SessionModelState::new(default_model, models))
}

fn codex_session_modes(current: &str) -> acp::SessionModeState {
    acp::SessionModeState::new(
        current.to_owned(),
        vec![
            acp::SessionMode::new("default", "Default")
                .description("Work normally and make changes when requested."),
            acp::SessionMode::new("plan", "Plan")
                .description("Investigate and produce a plan without making changes."),
        ],
    )
}

fn model_id(model: &Value) -> Option<&str> {
    model
        .get("id")
        .or_else(|| model.get("model"))
        .and_then(Value::as_str)
}

fn humanize_effort(effort: &str) -> String {
    let mut chars = effort.chars();
    chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

fn default_effort_for_model(models: &[acp::ModelInfo], model: &str) -> Option<ReasoningEffort> {
    models
        .iter()
        .find(|info| info.model_id.0.as_ref() == model)
        .and_then(|info| parse_reasoning_effort_meta(info.meta.as_ref()))
}

/// This optional global MCP can spend several seconds repeating OAuth metadata
/// discovery during every embedded thread start. Keep it out by default while
/// allowing users who need its tools to preserve it explicitly.
fn apply_thread_config(params: &mut Value, openai_docs_mcp: bool) {
    if !openai_docs_mcp {
        params["config"] = json!({"mcp_servers.openaiDeveloperDocs.enabled": false});
    }
}

fn new_thread_params(cwd: PathBuf, openai_docs_mcp: bool, model: Option<&str>) -> Value {
    let mut params = json!({
        "cwd": cwd,
        "approvalPolicy": "on-request",
        "sandbox": "workspace-write"
    });
    apply_thread_config(&mut params, openai_docs_mcp);
    if let Some(model) = model {
        params["model"] = json!(model);
    }
    params
}

fn turn_start_params(
    thread_id: &str,
    input: Vec<Value>,
    model: Option<String>,
    effort: Option<&str>,
    prompt_id: Option<&str>,
) -> Value {
    json!({
        "threadId": thread_id,
        "input": input,
        "model": model,
        "effort": effort,
        "clientUserMessageId": prompt_id,
    })
}

fn notify_session_startup_complete(client: &AcpGatewaySender<acp::AgentSide>, session_id: &str) {
    let params = serde_json::value::to_raw_value(&json!({"sessionId": session_id}))
        .expect("static MCP initialization payload must serialize");
    client.forward_fire_and_forget(acp::ExtNotification::new(
        "x.ai/mcp_initialized",
        params.into(),
    ));
}

fn acp_error(error: anyhow::Error) -> acp::Error {
    acp::Error::new(acp::ErrorCode::InternalError.into(), error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rpc() -> (Rc<CodexRpc>, mpsc::UnboundedReceiver<Value>) {
        let (outbound, outbound_rx) = mpsc::unbounded_channel();
        (
            Rc::new(CodexRpc {
                outbound,
                pending: Rc::new(RefCell::new(HashMap::new())),
                next_id: Cell::new(1),
                closed: Rc::new(Cell::new(false)),
            }),
            outbound_rx,
        )
    }

    #[tokio::test]
    async fn setting_acp_plan_mode_updates_codex_collaboration_mode() {
        let (outbound, mut outbound_rx) = mpsc::unbounded_channel();
        let pending = Rc::new(RefCell::new(HashMap::new()));
        let rpc = Rc::new(CodexRpc {
            outbound,
            pending: pending.clone(),
            next_id: Cell::new(1),
            closed: Rc::new(Cell::new(false)),
        });
        let (mut client_channel, agent_channel) = xai_acp_lib::acp_channels();
        let agent = CodexAgent {
            rpc,
            client: AcpGatewaySender::new(agent_channel.tx),
            models: Rc::new(RefCell::new(Vec::new())),
            selected_models: Rc::new(RefCell::new(HashMap::from([(
                "thread-1".into(),
                CodexModelSelection {
                    model: "gpt-5.6-sol".into(),
                    effort: Some(ReasoningEffort::Ultra),
                },
            )]))),
            active_turns: Rc::new(RefCell::new(HashMap::new())),
            completions: Rc::new(RefCell::new(HashMap::new())),
            login_completions: Rc::new(RefCell::new(HashMap::new())),
            authenticated: Cell::new(true),
            openai_docs_mcp: false,
        };

        let set_mode = acp::Agent::set_session_mode(
            &agent,
            acp::SetSessionModeRequest::new("thread-1", "plan"),
        );
        let app_server = async {
            let request = outbound_rx.recv().await.expect("app-server request");
            let id = request["id"].as_u64().expect("numeric request id");
            pending
                .borrow_mut()
                .remove(&id)
                .expect("pending app-server request")
                .send(Ok(json!({})))
                .expect("set mode request still waiting");
            request
        };
        let (result, request) = tokio::join!(set_mode, app_server);

        result.expect("Codex plan mode should be supported");
        assert_eq!(request["method"], "thread/settings/update");
        assert_eq!(
            request["params"],
            json!({
                "threadId": "thread-1",
                "collaborationMode": {
                    "mode": "plan",
                    "settings": {
                        "model": "gpt-5.6-sol",
                        "reasoning_effort": "ultra",
                        "developer_instructions": null
                    }
                }
            })
        );

        let message = client_channel.rx.recv().await.expect("mode notification");
        let xai_acp_lib::AcpClientMessage::SessionNotification(args) = message else {
            panic!("expected current mode notification");
        };
        assert_eq!(args.request.session_id.0.as_ref(), "thread-1");
        let acp::SessionUpdate::CurrentModeUpdate(update) = args.request.update else {
            panic!("expected current mode update");
        };
        assert_eq!(update.current_mode_id.0.as_ref(), "plan");
    }

    #[tokio::test]
    async fn codex_session_success_notifies_pager_that_startup_is_complete() {
        let (mut client_channel, agent_channel) = xai_acp_lib::acp_channels();
        let client = AcpGatewaySender::new(agent_channel.tx);

        notify_session_startup_complete(&client, "thread-1");

        let message = client_channel.rx.recv().await.unwrap();
        let xai_acp_lib::AcpClientMessage::ExtNotification(args) = message else {
            panic!("expected MCP initialization notification");
        };
        assert_eq!(args.request.method.as_ref(), "x.ai/mcp_initialized");
        let params: Value = serde_json::from_str(args.request.params.get()).unwrap();
        assert_eq!(params, json!({"sessionId": "thread-1"}));
    }

    #[tokio::test]
    async fn codex_request_user_input_round_trips_through_the_pager() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
                let (rpc, mut outbound_rx) = test_rpc();
                let (mut client_channel, agent_channel) = xai_acp_lib::acp_channels();
                let client = AcpGatewaySender::new(agent_channel.tx);
                let active_turns = Rc::new(RefCell::new(HashMap::new()));
                let completions = Rc::new(RefCell::new(HashMap::new()));
                let login_completions = Rc::new(RefCell::new(HashMap::new()));

                inbound_tx
                    .send(json!({
                        "id": 41,
                        "method": "item/tool/requestUserInput",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": "turn-1",
                            "itemId": "item-1",
                            "questions": [{
                                "id": "speed",
                                "header": "Strategy",
                                "question": "Which approach?",
                                "isOther": true,
                                "options": [
                                    {"label": "Fast", "description": "Ship the smallest fix"},
                                    {"label": "Thorough", "description": "Cover every edge case"}
                                ]
                            }]
                        }
                    }))
                    .unwrap();
                drop(inbound_tx);

                let drive_client = async {
                    let pending = client_channel
                        .rx
                        .recv()
                        .await
                        .expect("pager should receive a pending-interaction notification");
                    let xai_acp_lib::AcpClientMessage::ExtNotification(pending) = pending else {
                        panic!("question must be marked pending before opening the modal");
                    };
                    assert_eq!(pending.request.method.as_ref(), "x.ai/session_notification");
                    let pending: Value =
                        serde_json::from_str(pending.request.params.get()).unwrap();
                    assert_eq!(
                        pending,
                        json!({
                            "sessionId": "thread-1",
                            "update": {
                                "sessionUpdate": "pending_interaction",
                                "tool_call_id": "item-1",
                                "kind": "question"
                            }
                        })
                    );

                    let message = client_channel
                        .rx
                        .recv()
                        .await
                        .expect("pager should receive a question request");
                    let xai_acp_lib::AcpClientMessage::ExtMethod(args) = message else {
                        panic!("expected ask_user_question ext method");
                    };
                    assert_eq!(args.request.method.as_ref(), "x.ai/ask_user_question");
                    let request: Value = serde_json::from_str(args.request.params.get()).unwrap();
                    assert_eq!(request["sessionId"], "thread-1");
                    assert_eq!(request["toolCallId"], "item-1");
                    assert_eq!(
                        request["questions"][0]["question"],
                        "Strategy\n\nWhich approach?"
                    );
                    assert_eq!(request["questions"][0]["id"], "speed");

                    let response = serde_json::value::RawValue::from_string(
                        json!({
                            "outcome": "accepted",
                            "answers": {"speed": ["Fast"]}
                        })
                        .to_string(),
                    )
                    .unwrap();
                    args.response_tx
                        .send(Ok(acp::ExtResponse::new(response.into())))
                        .unwrap();

                    let resolved = client_channel
                        .rx
                        .recv()
                        .await
                        .expect("pager should receive an interaction-resolved notification");
                    let xai_acp_lib::AcpClientMessage::ExtNotification(resolved) = resolved else {
                        panic!("question must be resolved after the answer");
                    };
                    assert_eq!(
                        resolved.request.method.as_ref(),
                        "x.ai/session_notification"
                    );
                    let resolved: Value =
                        serde_json::from_str(resolved.request.params.get()).unwrap();
                    assert_eq!(
                        resolved,
                        json!({
                            "sessionId": "thread-1",
                            "update": {
                                "sessionUpdate": "interaction_resolved",
                                "tool_call_id": "item-1"
                            }
                        })
                    );

                    let response = outbound_rx
                        .recv()
                        .await
                        .expect("Codex app-server should receive the answer");
                    assert_eq!(response["id"], 41);
                    assert_eq!(
                        response["result"],
                        json!({"answers": {"speed": {"answers": ["Fast"]}}})
                    );
                };

                let forward = forward_events(
                    inbound_rx,
                    rpc,
                    client,
                    active_turns,
                    completions,
                    login_completions,
                );
                tokio::time::timeout(std::time::Duration::from_millis(250), async {
                    tokio::join!(forward, drive_client)
                })
                .await
                .expect("request_user_input must not remain unanswered");
            })
            .await;
    }

    #[tokio::test]
    async fn codex_questions_for_two_sessions_are_dispatched_concurrently() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
                let (rpc, mut outbound_rx) = test_rpc();
                let (mut client_channel, agent_channel) = xai_acp_lib::acp_channels();
                let client = AcpGatewaySender::new(agent_channel.tx);
                for (id, thread_id, item_id) in [
                    (51, "thread-foreground", "item-foreground"),
                    (52, "thread-background", "item-background"),
                ] {
                    inbound_tx
                        .send(json!({
                            "id": id,
                            "method": "item/tool/requestUserInput",
                            "params": {
                                "threadId": thread_id,
                                "turnId": format!("turn-{id}"),
                                "itemId": item_id,
                                "questions": [{
                                    "id": "choice",
                                    "header": "Choice",
                                    "question": "Pick one",
                                    "isOther": true,
                                    "options": [{
                                        "label": "One",
                                        "description": "Choose one"
                                    }]
                                }]
                            }
                        }))
                        .unwrap();
                }
                drop(inbound_tx);

                let drive_client = async {
                    let mut methods = Vec::new();
                    let mut pending_sessions = Vec::new();
                    for _ in 0..4 {
                        match client_channel.rx.recv().await.unwrap() {
                            xai_acp_lib::AcpClientMessage::ExtMethod(method) => {
                                methods.push(method);
                            }
                            xai_acp_lib::AcpClientMessage::ExtNotification(notification) => {
                                let notification: Value =
                                    serde_json::from_str(notification.request.params.get())
                                        .unwrap();
                                assert_eq!(
                                    notification["update"]["sessionUpdate"],
                                    "pending_interaction"
                                );
                                pending_sessions
                                    .push(notification["sessionId"].as_str().unwrap().to_owned());
                            }
                            _ => panic!("unexpected ACP message while opening Codex questions"),
                        }
                    }
                    pending_sessions.sort();
                    assert_eq!(
                        pending_sessions,
                        vec!["thread-background", "thread-foreground"]
                    );
                    assert_eq!(methods.len(), 2);

                    for method in methods {
                        let response = serde_json::value::RawValue::from_string(
                            json!({"outcome": "cancelled"}).to_string(),
                        )
                        .unwrap();
                        method
                            .response_tx
                            .send(Ok(acp::ExtResponse::new(response.into())))
                            .unwrap();
                    }

                    let mut resolved_sessions = Vec::new();
                    for _ in 0..2 {
                        let xai_acp_lib::AcpClientMessage::ExtNotification(notification) =
                            client_channel.rx.recv().await.unwrap()
                        else {
                            panic!("cancelled questions must resolve their pending interactions");
                        };
                        let notification: Value =
                            serde_json::from_str(notification.request.params.get()).unwrap();
                        assert_eq!(
                            notification["update"]["sessionUpdate"],
                            "interaction_resolved"
                        );
                        resolved_sessions
                            .push(notification["sessionId"].as_str().unwrap().to_owned());
                    }
                    resolved_sessions.sort();
                    assert_eq!(
                        resolved_sessions,
                        vec!["thread-background", "thread-foreground"]
                    );

                    let mut response_ids = vec![
                        outbound_rx.recv().await.unwrap()["id"].as_u64().unwrap(),
                        outbound_rx.recv().await.unwrap()["id"].as_u64().unwrap(),
                    ];
                    response_ids.sort_unstable();
                    assert_eq!(response_ids, vec![51, 52]);
                };

                let forward = forward_events(
                    inbound_rx,
                    rpc,
                    client,
                    Rc::new(RefCell::new(HashMap::new())),
                    Rc::new(RefCell::new(HashMap::new())),
                    Rc::new(RefCell::new(HashMap::new())),
                );
                tokio::time::timeout(std::time::Duration::from_millis(500), async {
                    tokio::join!(forward, drive_client)
                })
                .await
                .expect("a foreground question must not block a background session");
            })
            .await;
    }

    #[test]
    fn codex_request_user_input_maps_other_text_to_the_question_id() {
        let request: CodexRequestUserInputParams = serde_json::from_value(json!({
            "threadId": "thread-1",
            "turnId": "turn-1",
            "itemId": "item-1",
            "questions": [{
                "id": "approach",
                "header": "Approach",
                "question": "How should we proceed?",
                "isOther": true,
                "options": [{"label": "Small", "description": "Small change"}]
            }]
        }))
        .unwrap();
        let response = serde_json::from_value(json!({
            "outcome": "accepted",
            "answers": {"approach": ["Other"]},
            "annotations": {"approach": {"notes": "Use a feature flag"}}
        }))
        .unwrap();

        assert_eq!(
            serde_json::to_value(codex_user_input_result(&request, response)).unwrap(),
            json!({
                "answers": {
                    "approach": {"answers": ["Use a feature flag"]}
                }
            })
        );
    }

    #[test]
    fn cancelling_codex_request_user_input_returns_empty_answers() {
        let request: CodexRequestUserInputParams = serde_json::from_value(json!({
            "threadId": "thread-1",
            "turnId": "turn-1",
            "itemId": "item-1",
            "questions": [{
                "id": "approach",
                "header": "Approach",
                "question": "How should we proceed?",
                "isOther": true,
                "options": [{"label": "Small", "description": "Small change"}]
            }]
        }))
        .unwrap();
        let response = serde_json::from_value(json!({"outcome": "cancelled"})).unwrap();

        assert_eq!(
            serde_json::to_value(codex_user_input_result(&request, response)).unwrap(),
            json!({"answers": {}})
        );
    }

    #[tokio::test]
    async fn codex_request_user_input_auto_resolves_and_retracts_the_modal() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
                let (rpc, mut outbound_rx) = test_rpc();
                let (mut client_channel, agent_channel) = xai_acp_lib::acp_channels();
                inbound_tx
                    .send(json!({
                        "id": 44,
                        "method": "item/tool/requestUserInput",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": "turn-1",
                            "itemId": "item-1",
                            "autoResolutionMs": 1,
                            "questions": [{
                                "id": "approach",
                                "header": "Approach",
                                "question": "How should we proceed?",
                                "isOther": true,
                                "options": [{
                                    "label": "Small",
                                    "description": "Small change"
                                }]
                            }]
                        }
                    }))
                    .unwrap();
                drop(inbound_tx);

                let drive_client = async {
                    let xai_acp_lib::AcpClientMessage::ExtNotification(pending) =
                        client_channel.rx.recv().await.unwrap()
                    else {
                        panic!("auto-resolving question must first be marked pending");
                    };
                    let pending: Value =
                        serde_json::from_str(pending.request.params.get()).unwrap();
                    assert_eq!(pending["update"]["sessionUpdate"], "pending_interaction");
                    assert!(matches!(
                        client_channel.rx.recv().await.unwrap(),
                        xai_acp_lib::AcpClientMessage::ExtMethod(_)
                    ));
                    let xai_acp_lib::AcpClientMessage::ExtNotification(notification) =
                        client_channel.rx.recv().await.unwrap()
                    else {
                        panic!("auto-resolution should retract the pager modal");
                    };
                    assert_eq!(
                        notification.request.method.as_ref(),
                        "x.ai/session_notification"
                    );
                    let notification: Value =
                        serde_json::from_str(notification.request.params.get()).unwrap();
                    assert_eq!(notification["sessionId"], "thread-1");
                    assert_eq!(
                        notification["update"],
                        json!({
                            "sessionUpdate": "interaction_resolved",
                            "tool_call_id": "item-1"
                        })
                    );

                    let response = outbound_rx.recv().await.unwrap();
                    assert_eq!(response["id"], 44);
                    assert_eq!(response["result"], json!({"answers": {}}));
                    assert!(outbound_rx.try_recv().is_err());
                };
                let forward = forward_events(
                    inbound_rx,
                    rpc,
                    AcpGatewaySender::new(agent_channel.tx),
                    Rc::new(RefCell::new(HashMap::new())),
                    Rc::new(RefCell::new(HashMap::new())),
                    Rc::new(RefCell::new(HashMap::new())),
                );
                tokio::time::timeout(std::time::Duration::from_millis(250), async {
                    tokio::join!(forward, drive_client)
                })
                .await
                .expect("auto-resolution must answer the original app-server request");
            })
            .await;
    }

    #[tokio::test]
    async fn malformed_codex_request_user_input_receives_an_error_response() {
        let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
        let (rpc, mut outbound_rx) = test_rpc();
        let (_client_channel, agent_channel) = xai_acp_lib::acp_channels();
        inbound_tx
            .send(json!({
                "id": 42,
                "method": "item/tool/requestUserInput",
                "params": {
                    "threadId": "thread-1",
                    "itemId": "item-1",
                    "questions": [{"question": "Missing an id"}]
                }
            }))
            .unwrap();
        drop(inbound_tx);

        forward_events(
            inbound_rx,
            rpc,
            AcpGatewaySender::new(agent_channel.tx),
            Rc::new(RefCell::new(HashMap::new())),
            Rc::new(RefCell::new(HashMap::new())),
            Rc::new(RefCell::new(HashMap::new())),
        )
        .await;

        let response = outbound_rx.recv().await.unwrap();
        assert_eq!(response["id"], 42);
        assert_eq!(response["error"]["code"], -32602);
        assert!(
            response["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid Codex request_user_input params")
        );
    }

    #[tokio::test]
    async fn secret_codex_request_user_input_fails_closed() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
                let (rpc, mut outbound_rx) = test_rpc();
                let (mut client_channel, agent_channel) = xai_acp_lib::acp_channels();
                inbound_tx
                    .send(json!({
                        "id": 43,
                        "method": "item/tool/requestUserInput",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": "turn-1",
                            "itemId": "item-1",
                            "questions": [{
                                "id": "token",
                                "header": "Token",
                                "question": "Enter a token",
                                "isOther": true,
                                "options": [{
                                    "label": "Enter",
                                    "description": "Provide the token"
                                }],
                                "isSecret": true
                            }]
                        }
                    }))
                    .unwrap();
                drop(inbound_tx);

                let drive_client = async {
                    let response = outbound_rx.recv().await.unwrap();
                    assert_eq!(response["id"], 43);
                    assert_eq!(response["error"]["code"], -32602);
                    assert!(
                        response["error"]["message"]
                            .as_str()
                            .unwrap()
                            .contains("secret")
                    );
                    assert!(outbound_rx.try_recv().is_err());
                    assert!(client_channel.rx.try_recv().is_err());
                };
                let forward = forward_events(
                    inbound_rx,
                    rpc,
                    AcpGatewaySender::new(agent_channel.tx),
                    Rc::new(RefCell::new(HashMap::new())),
                    Rc::new(RefCell::new(HashMap::new())),
                    Rc::new(RefCell::new(HashMap::new())),
                );
                tokio::time::timeout(std::time::Duration::from_millis(250), async {
                    tokio::join!(forward, drive_client)
                })
                .await
                .expect("secret questions must receive one correlated error response");
            })
            .await;
    }

    #[test]
    fn maps_codex_default_model_catalog_to_initial_acp_state() {
        let response = json!({"data": [
            {"id": "gpt-5.4", "displayName": "GPT-5.4", "isDefault": false},
            {
                "model": "gpt-5.6-sol",
                "name": "GPT-5.6 Sol",
                "description": "Latest frontier agentic coding model.",
                "isDefault": true,
                "defaultReasoningEffort": "low",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "low", "description": "Fast"},
                    {"reasoningEffort": "xhigh", "description": "Extra high"},
                    {"reasoningEffort": "max", "description": "Maximum"},
                    {"reasoningEffort": "ultra", "description": "Automatic delegation"}
                ]
            }
        ]});
        let model_state = map_model_state(&response, map_models(&response)).unwrap();
        assert_eq!(model_state.current_model_id.0.as_ref(), "gpt-5.6-sol");
        assert_eq!(model_state.available_models.len(), 2);
        assert_eq!(
            model_state.available_models[0].model_id.0.as_ref(),
            "gpt-5.4"
        );
        let sol = &model_state.available_models[1];
        assert_eq!(sol.name, "GPT-5.6 Sol");
        assert_eq!(
            sol.description.as_deref(),
            Some("Latest frontier agentic coding model.")
        );
        let meta = sol.meta.as_ref().unwrap();
        assert_eq!(meta["reasoningEffort"], "low");
        let efforts = meta["reasoningEfforts"].as_array().unwrap();
        assert_eq!(
            efforts
                .iter()
                .map(|entry| entry["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["low", "xhigh", "max", "ultra"]
        );
        assert_eq!(efforts[0]["default"], true);
    }

    #[test]
    fn codex_thread_start_disables_redundant_docs_mcp_discovery() {
        let params = new_thread_params(PathBuf::from("/workspace"), false, None);
        assert_eq!(
            params.pointer("/config/mcp_servers.openaiDeveloperDocs.enabled"),
            Some(&Value::Bool(false))
        );
    }

    #[test]
    fn codex_thread_start_can_preserve_docs_mcp() {
        let params = new_thread_params(PathBuf::from("/workspace"), true, None);
        assert!(params.get("config").is_none());
    }

    #[test]
    fn codex_thread_start_uses_requested_model() {
        let params = new_thread_params(PathBuf::from("/workspace"), true, Some("gpt-5.6-terra"));
        assert_eq!(params["model"], "gpt-5.6-terra");
    }

    #[test]
    fn codex_selection_preserves_max_and_ultra_as_distinct_efforts() {
        assert_eq!(
            "xhigh".parse::<ReasoningEffort>().unwrap().as_str(),
            "xhigh"
        );
        assert_eq!("max".parse::<ReasoningEffort>().unwrap().as_str(), "max");
        assert_eq!(
            "ultra".parse::<ReasoningEffort>().unwrap().as_str(),
            "ultra"
        );
    }

    #[test]
    fn codex_turn_start_forwards_selected_model_and_effort() {
        let params = turn_start_params(
            "thread-1",
            vec![json!({"type": "text", "text": "hello"})],
            Some("gpt-5.6-sol".into()),
            Some("ultra"),
            Some("prompt-42"),
        );
        assert_eq!(params["model"], "gpt-5.6-sol");
        assert_eq!(params["effort"], "ultra");
        assert_eq!(params["clientUserMessageId"], "prompt-42");
    }

    #[test]
    fn translates_text_prompt_to_codex_user_input() {
        let input =
            prompt_to_codex_input(&[acp::ContentBlock::Text(acp::TextContent::new("hello"))]);
        assert_eq!(input, vec![json!({"type": "text", "text": "hello"})]);
    }
}
