//! Writing `.srt`.
//!
//! `dec` preserves length (§14.3), so the timecodes anchored to the source audio
//! stay valid against the converted audio — no re-alignment is needed here.

use std::fmt::Write as _;

/// One subtitle cue, in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub start_s: f64,
    pub end_s: f64,
    pub text: String,
}

/// Format seconds as an SRT timestamp: `HH:MM:SS,mmm`.
///
/// Negative input is clamped to zero. Values are rounded to the nearest
/// millisecond, so `59.9996` rolls over to `00:01:00,000`.
pub fn timestamp(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let s = (total_ms / 1000) % 60;
    let m = (total_ms / 60_000) % 60;
    let h = total_ms / 3_600_000;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Render cues as a `.srt` document (1-indexed, blank line separated).
pub fn to_srt(entries: &[Entry]) -> String {
    let mut out = String::new();
    for (i, e) in entries.iter().enumerate() {
        let _ = write!(
            out,
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            timestamp(e.start_s),
            timestamp(e.end_s),
            e.text.trim()
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps() {
        assert_eq!(timestamp(0.0), "00:00:00,000");
        assert_eq!(timestamp(1.5), "00:00:01,500");
        assert_eq!(timestamp(61.25), "00:01:01,250");
        assert_eq!(timestamp(3661.5), "01:01:01,500");
        assert_eq!(timestamp(-3.0), "00:00:00,000");
    }

    #[test]
    fn timestamp_rounds_to_nearest_millisecond() {
        assert_eq!(timestamp(0.0004), "00:00:00,000");
        assert_eq!(timestamp(0.0006), "00:00:00,001");
        assert_eq!(timestamp(59.9996), "00:01:00,000");
    }

    #[test]
    fn srt_layout() {
        let entries = vec![
            Entry {
                start_s: 0.0,
                end_s: 1.0,
                text: "Xin chào.".to_string(),
            },
            Entry {
                start_s: 1.0,
                end_s: 2.5,
                text: "Tôi là Hùng.".to_string(),
            },
        ];
        let srt = to_srt(&entries);
        assert_eq!(
            srt,
            "1\n00:00:00,000 --> 00:00:01,000\nXin chào.\n\n\
             2\n00:00:01,000 --> 00:00:02,500\nTôi là Hùng.\n\n"
        );
    }

    #[test]
    fn empty_entries_yield_empty_document() {
        assert_eq!(to_srt(&[]), "");
    }

    #[test]
    fn cue_text_is_trimmed() {
        let entries = vec![Entry {
            start_s: 0.0,
            end_s: 1.0,
            text: "  padded  ".to_string(),
        }];
        assert!(to_srt(&entries).contains("padded\n"));
    }
}
