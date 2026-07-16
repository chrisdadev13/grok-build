//! Welcome logo compatibility helpers.
//!
//! The initial TUI no longer renders logo artwork. These helpers remain as
//! no-ops so the standard, blocked, and minimal welcome layouts share the same
//! behavior without carrying logo-specific branches at each call site.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use crate::theme::Theme;

pub fn logo_line_count(_window_height: u16) -> u16 {
    0
}

/// Preserve the welcome column's established minimum width without artwork.
pub fn logo_visual_width(_window_height: u16) -> u16 {
    24
}

pub fn render_logo(_area: Rect, _buf: &mut Buffer, _theme: &Theme, _window_height: u16) {}

pub fn full_logo_line_count() -> u16 {
    0
}

pub fn full_logo_visual_width() -> u16 {
    0
}

pub fn render_full_logo(_area: Rect, _buf: &mut Buffer, _theme: &Theme) {}

pub fn compact_logo_line_count() -> u16 {
    0
}

pub fn render_compact_logo(_area: Rect, _buf: &mut Buffer, _theme: &Theme) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logo_is_absent_from_every_welcome_layout() {
        for height in [0, 22, 26, u16::MAX] {
            assert_eq!(logo_line_count(height), 0);
            assert_eq!(logo_visual_width(height), 24);
        }
        assert_eq!(full_logo_line_count(), 0);
        assert_eq!(full_logo_visual_width(), 0);
        assert_eq!(compact_logo_line_count(), 0);
    }
}
