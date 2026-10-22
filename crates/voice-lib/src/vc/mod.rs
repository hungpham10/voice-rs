//! Voice conversion (VC) — one speaker → one speaker. See `docs/architecture.md` §13.
//!
//! The path has two halves:
//! - `resample` — interpolate *outside* the network: frame-rate nearest mapping
//!   (§13.2.1) and 16 k→22.05 k sample-rate conversion for the loss target.
//! - `adapter` — the only learnable piece (P1, §13.6): HuBERT `768 @ 50 Hz` →
//!   latent `192 @ 86.1328125 Hz` via one `Linear`.

mod adapter;
mod resample;

pub use adapter::{Adapter, IN_DIM, OUT_DIM, PITCH_BINS};
pub use resample::{nearest_index_map, resample_linear};
