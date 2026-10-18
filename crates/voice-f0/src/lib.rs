//! voice-f0 — RMVPE pitch extractor via burn-onnx codegen.
//!
//! The ONNX model is expected at `sources/rmvpe_f0.onnx` (E2E-only: mel [128,T] -> salience [360,T]).

use burn::backend::Flex;
use burn::tensor::{Tensor, Device, Float, TensorData};
use std::env;
use std::path::Path;
use thiserror::Error;

// Include the generated model code from OUT_DIR
// Tên module lấy theo tên file ONNX: rmvpe_f0.onnx -> rmvpe_f0.rs
pub mod rmvpe_model {
    include!(concat!(env!("OUT_DIR"), "/model/rmvpe_f0.rs"));
}
use rmvpe_model::Model;

pub mod decode;
pub use decode::{decode, quantize, N_CLASS, THRED};

pub type Backend = Flex;
pub type DeviceType = Device;

/// RMVPE pitch extractor wrapper.
pub struct Rmvpe {
    model: Model,
    device: Device,
}

impl Rmvpe {
    /// Create new pitch extractor with default model (device = `Device::default()`).
    pub fn new() -> Result<Self, F0Error> {
        Self::new_with_device(Device::default())
    }

    /// Create new pitch extractor on the given device (`cpu`/`cuda`).
    pub fn new_with_device(device: Device) -> Result<Self, F0Error> {
        // Build script writes .bpk into OUT_DIR (same directory as generated model code)
        // The ONNX file is rmvpe_f0.onnx, so the model is named rmvpe_f0
        let bpk = Path::new(env!("OUT_DIR")).join("model/rmvpe_f0.bpk");
        if !bpk.exists() {
            return Err(F0Error::ModelNotFound(format!(
                "{} — run `cargo build` so the build script generates it from rmvpe_f0.onnx",
                bpk.display()
            )));
        }
        println!("Loading weights: {}", bpk.display());
        let model = Model::from_file(&bpk, &device);

        Ok(Self { model, device })
    }

    /// Forward pass: mel spectrogram -> salience.
    ///
    /// Args:
    ///     mel: Mel spectrogram [128][T] (from MelConfig::rmvpe())
    ///
    /// Returns:
    ///     Tensor of shape [1, 360, T] with sigmoid output (salience per pitch bin)
    pub fn forward(&self, mel: &[Vec<f32>]) -> Result<Tensor<3, Float>, F0Error> {
        let n_mels = mel.len();
        let n_frames = mel[0].len();

        // Reflect pad to multiple of 32 (U-Net 5 pooling layers)
        let padded = 32 * ((n_frames - 1) / 32 + 1);

        // Flatten mel data: [128][T] -> [1, 128, padded] (layout ONNX expects)
        let mut data = Vec::with_capacity(n_mels * padded);
        for m in 0..n_mels {
            for t in 0..padded {
                if t < n_frames {
                    data.push(mel[m][t]);
                } else {
                    // Reflect padding
                    let idx = 2 * n_frames - 2 - t;
                    data.push(mel[m][idx.clamp(0, n_frames - 1)]);
                }
            }
        }

        // Input tensor [1, 128, padded]
        let input = Tensor::<3, Float>::from_data(TensorData::new(data, [1, n_mels, padded]), &self.device);

        // Run model: input [1, 128, padded] -> output [1, padded, 360] (model outputs time, channels)
        let output = self.model.forward(input);

        // Transpose from [1, padded, 360] to [1, 360, padded]
        let output = output.swap_dims(1, 2);

        // Crop back to original length
        let output = output.slice([0..1, 0..N_CLASS, 0..n_frames]);

        Ok(output)
    }

    /// Forward pass + decode to f0 in Hz.
    pub fn forward_f0(&self, mel: &[Vec<f32>]) -> Result<Vec<f32>, F0Error> {
        let salience = self.forward(mel)?;
        Ok(decode(&salience))
    }

    /// Get the device being used.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[derive(Debug, Error)]
pub enum F0Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Burn error: {0}")]
    Burn(String),
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
}