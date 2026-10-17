//! Splitting a transcript into caption units at punctuation (§14.5).
//!
//! The pipeline: text split on punctuation gives `N` chunks *and* the number of
//! cut points is known up front — which is what makes this cheaper than scanning
//! the audio for RMS minima (§14.8). Anchoring those chunks to time is a separate
//! step (§14.5.1) and does not happen here.

/// One caption unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Normalised text, including the trailing boundary punctuation (if any).
    pub text: String,
    /// The punctuation character this chunk ended on, if it was closed by one.
    pub end_punct: Option<char>,
}

impl Chunk {
    /// Length used to allocate time/frames proportionally — see `split_frames`.
    pub fn weight(&self) -> f64 {
        self.text.chars().count() as f64
    }
}

/// How to split.
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Characters that terminate a chunk.
    ///
    /// §14.5 research question (a) is still open: exactly which marks produce a
    /// real pause. The default is deliberately broad and configurable.
    pub punctuation: String,
    /// Hard cap on chunk length; longer runs are re-split at the last whitespace
    /// before the cap (§14.5 research question (c)). `0` disables the cap.
    pub max_chars: usize,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            punctuation: ".,!?;:…".to_string(),
            max_chars: 120,
        }
    }
}

/// Split `text` at the default punctuation set.
pub fn split_on_punct(text: &str) -> Vec<Chunk> {
    split_with(text, &SplitConfig::default())
}

/// Split `text` at sentence/clause punctuation.
///
/// A run of consecutive punctuation (`"Wait... really?!"`) is absorbed into a
/// single chunk so it does not produce empty captions. Whitespace runs (including
/// newlines) collapse to a single space, and chunks carrying no alphanumeric
/// content are dropped. When `max_chars` re-splits a long chunk, the trailing
/// punctuation stays on the last piece.
pub fn split_with(text: &str, cfg: &SplitConfig) -> Vec<Chunk> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut cur = String::new();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        cur.push(c);

        if cfg.punctuation.contains(c) {
            let mut last = c;
            let mut j = i + 1;
            while j < chars.len() && cfg.punctuation.contains(chars[j]) {
                cur.push(chars[j]);
                last = chars[j];
                j += 1;
            }
            push_chunk(&mut chunks, &mut cur, Some(last));
            i = j;
            continue;
        }

        i += 1;
    }

    push_chunk(&mut chunks, &mut cur, None);

    if cfg.max_chars > 0 {
        enforce_max_chars(chunks, cfg.max_chars)
    } else {
        chunks
    }
}

/// Trim, collapse internal whitespace, and push if it carries any content.
///
/// A chunk made only of punctuation (`"..."`) is dropped — it is a cut marker,
/// not a caption.
fn push_chunk(chunks: &mut Vec<Chunk>, cur: &mut String, end_punct: Option<char>) {
    let normalised: String = cur.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalised.chars().any(char::is_alphanumeric) {
        chunks.push(Chunk {
            text: normalised,
            end_punct,
        });
    }
    cur.clear();
}

/// Re-split chunks longer than `max` at the last whitespace before the cap.
fn enforce_max_chars(chunks: Vec<Chunk>, max: usize) -> Vec<Chunk> {
    let mut out = Vec::new();

    for chunk in chunks {
        let mut rest = chunk.text;
        let end_punct = chunk.end_punct;

        loop {
            if rest.chars().count() <= max {
                out.push(Chunk {
                    text: rest.trim().to_string(),
                    end_punct,
                });
                break;
            }

            let cut = byte_index_of_char(&rest, max);
            let head = &rest[..cut];

            match head.rfind(char::is_whitespace) {
                Some(p) if p > 0 => {
                    out.push(Chunk {
                        text: rest[..p].trim().to_string(),
                        end_punct: None,
                    });
                    rest = rest[p..].trim_start().to_string();
                }
                // No whitespace to break on — cut at the cap, mid-word.
                _ => {
                    out.push(Chunk {
                        text: head.trim().to_string(),
                        end_punct: None,
                    });
                    rest = rest[cut..].trim_start().to_string();
                }
            }
        }
    }

    out
}

/// Byte offset of the `n`-th character, or `s.len()` if there are fewer.
fn byte_index_of_char(s: &str, n: usize) -> usize {
    s.char_indices().nth(n).map(|(i, _)| i).unwrap_or(s.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(chunks: &[Chunk]) -> Vec<&str> {
        chunks.iter().map(|c| c.text.as_str()).collect()
    }

    #[test]
    fn splits_on_period_and_comma() {
        let chunks = split_on_punct("Xin chào. Tôi là Hùng, rất vui.");
        assert_eq!(texts(&chunks), ["Xin chào.", "Tôi là Hùng,", "rất vui."]);
        assert_eq!(chunks[0].end_punct, Some('.'));
        assert_eq!(chunks[1].end_punct, Some(','));
        assert_eq!(chunks[2].end_punct, Some('.'));
    }

    #[test]
    fn trailing_text_without_punctuation_is_kept() {
        let chunks = split_on_punct("Hello world");
        assert_eq!(texts(&chunks), ["Hello world"]);
        assert_eq!(chunks[0].end_punct, None);
    }

    #[test]
    fn comma_separated_list() {
        let chunks = split_on_punct("a, b, c");
        assert_eq!(texts(&chunks), ["a,", "b,", "c"]);
        assert_eq!(chunks[2].end_punct, None);
    }

    #[test]
    fn consecutive_punctuation_is_absorbed() {
        let chunks = split_on_punct("Wait... really?!");
        assert_eq!(texts(&chunks), ["Wait...", "really?!"]);
        assert_eq!(chunks[0].end_punct, Some('.'));
        assert_eq!(chunks[1].end_punct, Some('!'));
    }

    #[test]
    fn newlines_collapse_to_spaces() {
        let chunks = split_on_punct("Hello\nworld.  Bye");
        assert_eq!(texts(&chunks), ["Hello world.", "Bye"]);
    }

    #[test]
    fn empty_and_whitespace_inputs_yield_no_chunks() {
        assert!(split_on_punct("").is_empty());
        assert!(split_on_punct("   \n\t ").is_empty());
        assert!(split_on_punct("...").is_empty());
    }

    #[test]
    fn custom_punctuation_set() {
        let cfg = SplitConfig {
            punctuation: ".".to_string(),
            max_chars: 0,
        };
        let chunks = split_with("one, two. three", &cfg);
        assert_eq!(texts(&chunks), ["one, two.", "three"]);
    }

    #[test]
    fn long_chunk_is_split_at_whitespace_before_cap() {
        let cfg = SplitConfig {
            punctuation: ".".to_string(),
            max_chars: 10,
        };
        let chunks = split_with("aaaa bbbb cccc dddd", &cfg);
        assert_eq!(texts(&chunks), ["aaaa bbbb", "cccc dddd"]);
        assert!(chunks.iter().all(|c| c.text.chars().count() <= 10));
    }

    #[test]
    fn long_run_without_whitespace_is_hard_cut() {
        let cfg = SplitConfig {
            punctuation: ".".to_string(),
            max_chars: 4,
        };
        let chunks = split_with("abcdefghij", &cfg);
        assert_eq!(texts(&chunks), ["abcd", "efgh", "ij"]);
    }

    #[test]
    fn max_chars_keeps_end_punct_on_final_piece_only() {
        let cfg = SplitConfig {
            punctuation: ".".to_string(),
            max_chars: 5,
        };
        let chunks = split_with("aa bb cc dd.", &cfg);
        assert_eq!(chunks.last().unwrap().end_punct, Some('.'));
        assert!(chunks[..chunks.len() - 1]
            .iter()
            .all(|c| c.end_punct.is_none()));
    }

    #[test]
    fn weight_is_character_count() {
        let chunks = split_on_punct("ab. cdef");
        assert_eq!(chunks[0].weight(), 3.0);
        assert_eq!(chunks[1].weight(), 4.0);
    }
}
