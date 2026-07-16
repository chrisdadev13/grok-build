# Codex harness startup: model seeding and MCP discovery

Research date: 2026-07-15. Sources are pinned to the revisions inspected.

## Executive findings

1. **OpenCode and Pi are not Codex app-server harnesses.** Both implement ChatGPT OAuth and call `https://chatgpt.com/backend-api/codex/responses` directly. Their fast startup is therefore not evidence that Codex app-server can start a thread without initializing its configured MCP servers. [OpenCode defines and rewrites requests to the Codex Responses endpoint](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/opencode/src/plugin/openai/codex.ts#L10-L16), [then attaches its stored OAuth token and account id itself](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/opencode/src/plugin/openai/codex.ts#L340-L425). Pi likewise constructs an `openai-codex` provider around a static catalog and its own Responses implementation, rather than launching Codex app-server. [Pi provider](https://github.com/earendil-works/pi/blob/c6d8371521fc8357958bb21fd43552c15f46c7f4/packages/ai/src/providers/openai-codex.ts#L1-L17), [Pi Codex Responses implementation](https://github.com/earendil-works/pi/blob/c6d8371521fc8357958bb21fd43552c15f46c7f4/packages/ai/src/api/openai-codex-responses.ts#L249-L340).

2. **Both seed model data before a conversation starts.** OpenCode builds its provider database from a cached or bundled `models.dev` snapshot and applies an OAuth-specific model filter. [Catalog construction](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/opencode/src/provider/provider.ts#L1330-L1374), [Codex model filtering](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/opencode/src/plugin/openai/codex.ts#L278-L318). Pi compiles an auto-generated `OPENAI_CODEX_MODELS` map into the provider and returns `Object.values(...)` synchronously. [Generated catalog](https://github.com/earendil-works/pi/blob/c6d8371521fc8357958bb21fd43552c15f46c7f4/packages/ai/src/providers/openai-codex.models.ts#L1-L12), [provider wiring](https://github.com/earendil-works/pi/blob/c6d8371521fc8357958bb21fd43552c15f46c7f4/packages/ai/src/providers/openai-codex.ts#L7-L16).

3. **Codex app-server already has the primitive the pager needs.** `model/list` is an app-server request independent of `thread/start`; it returns available models and their ordered reasoning-effort metadata. [Official app-server protocol](https://github.com/openai/codex/blob/800715d201651a2a07c2706dca10400109dae3d3/codex-rs/app-server/README.md#L58-L60). The pager can therefore populate its model selector and choose a provisional/default model before waiting for thread creation.

4. **The supported way to remove a problematic Codex MCP server from startup is `enabled = false`.** Codex's official config schema includes nullable `mcp_servers.<name>.enabled`; the resolved server setting defaults to enabled, while false skips initialization. [Official config schema](https://github.com/openai/codex/blob/800715d201651a2a07c2706dca10400109dae3d3/codex-rs/core/config.schema.json#L32-L34). For the observed server this means an override equivalent to:

   ```toml
   [mcp_servers.openaiDeveloperDocs]
   enabled = false
   ```

   This is preferable to merely shortening the startup timeout: it prevents connection and OAuth discovery rather than failing it sooner. It should only be applied if the pager does not need that server's tools.

5. **Codex exposes MCP startup as state, so the UI should not treat it as model discovery.** App-server emits `mcpServer/startupStatus/updated` with `starting`, `ready`, `failed`, or `cancelled`; an OAuth refresh failure is explicitly reported as `reauthenticationRequired`. [Official event definition](https://github.com/openai/codex/blob/800715d201651a2a07c2706dca10400109dae3d3/codex-rs/app-server/README.md#L206-L208). This supports rendering the model immediately while separately showing MCP startup/degraded status.

## Patterns worth copying

### Model catalog and default seeding

OpenCode's ACP loader fetches providers, agents, commands, skills, and config concurrently, then returns the provider model options and resolved default together. Its first-session default deliberately avoids historical-session scans: configured model, preferred provider, then the highest sorted available model. [Concurrent ACP snapshot](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/opencode/src/acp/service.ts#L720-L774), [deterministic default](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/opencode/src/acp/service.ts#L777-L800).

Its model source is also startup-friendly after the first run: disk cache wins, a bundled snapshot is the next fallback, and only then does it fetch `models.dev`; refresh happens in the background. [Cache/snapshot/fetch order](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/core/src/models-dev.ts#L154-L229), [background refresh](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/core/src/models-dev.ts#L231-L254).

Pi takes the simpler library approach: ship the generated catalog and lazily import OAuth code only when login, refresh, or credential conversion is invoked. [Lazy OAuth helper](https://github.com/earendil-works/pi/blob/c6d8371521fc8357958bb21fd43552c15f46c7f4/packages/ai/src/auth/helpers.ts#L27-L48). Neither pattern waits for a session/network turn before it can display a model name.

### Avoiding repeated OAuth work

OpenCode's **provider OAuth** path stores token expiry and collapses concurrent refresh attempts into one `refreshPromise`. [Single-flight token refresh](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/opencode/src/plugin/openai/codex.ts#L333-L389). This concerns ChatGPT authentication, not MCP OAuth discovery.

OpenCode's separate **MCP OAuth** path is still useful as a design reference: remote MCP config supports both `enabled: false` and `oauth: false`; the latter disables OAuth auto-detection while still attempting an unauthenticated/header-authenticated connection. [Config contract](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/core/src/v1/config/mcp.ts#L44-L59), [auth-provider omission](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/opencode/src/mcp/index.ts#L236-L267). It also stops trying its fallback transport once it has classified the result as authentication-required, avoiding a second equivalent discovery chain. [Auth classification and early stop](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/opencode/src/mcp/index.ts#L289-L333). MCP servers initialize concurrently and disabled entries return before connecting. [Startup loop](https://github.com/anomalyco/opencode/blob/4a760b5743496942fd821eeafaa7d648a5630973/packages/opencode/src/mcp/index.ts#L505-L529).

Codex's current schema does **not** expose an OpenCode-equivalent `oauth = false` switch. It exposes explicit auth modes, configured headers/bearer tokens, and whole-server `enabled`; therefore the safe integration-level fix available now is to disable the unneeded server, or fix its URL/auth configuration, rather than attempting to suppress OAuth discovery in the pager.

## Recommended pager implementation

1. During Codex initialization, call/consume `model/list` and immediately publish the model options plus a provisional selected model to the pager. Selection order should be: explicit requested/configured model, Codex-reported default/current model if available, then first visible model. Never use the literal `Unknown` while a nonempty catalog exists.
2. Start the thread after seeding that state. Reconcile the selected model with the model returned by `thread/start`, because project config or resume state may change it.
3. Treat MCP startup separately. Subscribe to `mcpServer/startupStatus/updated`; show a compact startup/degraded indication without replacing the model label or blocking prompt rendering.
4. Disable `openaiDeveloperDocs` with the official `enabled = false` override when its tools are not required. Make this an explicit adapter/user setting rather than silently rewriting all Codex MCP configuration.
5. If the server is required, preserve it and investigate its endpoint/auth response. The repeated 405/404 discovery chain indicates a server/configuration mismatch; OpenCode and Pi do not provide a directly transferable fix because they never ask Codex app-server to initialize that MCP server.

## Expected effect

Model seeding removes the visible `Unknown` race even if `thread/start` remains slow. Disabling the unneeded MCP server removes the observed network discovery chain from Codex thread startup. These are independent changes and should be timed independently in tests: time-to-model-render, `model/list`, `thread/start`, and each MCP server's startup transition.
