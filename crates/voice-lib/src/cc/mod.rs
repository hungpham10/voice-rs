//! Closed captions (CC) — see `docs/architecture.md` §14.
//!
//! `text` splits a transcript at punctuation; `frames` allocates latent frames
//! across the resulting chunks without drift; `srt` writes the result.

mod frames;
mod srt;
mod text;

pub use frames::{frame_count, split_frames};
pub use srt::{to_srt, timestamp, Entry};
pub use text::{split_on_punct, split_with, Chunk, SplitConfig};
