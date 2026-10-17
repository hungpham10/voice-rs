//! `voice-lib` — shared library.
//!
//! Holds the pieces that both the runtime pipeline and external consumers need.
//! It is deliberately free of `sources/`, build scripts, and training loops:
//! training and validation live in a separate crate.
//!
//! Modules:
//! - [`cc`] — closed captions: split text at punctuation (§14.5), allocate latent
//!   frames without drift (§14.4), write `.srt`.
//! - [`vc`] — one-speaker→one-speaker voice conversion; currently the frame-rate
//!   interpolation that sits *outside* the network (§13.2.1). The Burn `Adapter`
//!   (P1, §13.6) joins here with the integration step.

pub mod cc;
pub mod vc;
