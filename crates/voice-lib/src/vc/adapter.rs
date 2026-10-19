//! Adapter HuBERT → latent (P1, §13.6).
//!
//! The only learnable piece of the VC path for 1→1. It does two things:
//!
//! 1. **frame-rate reconciliation** — HuBERT runs at 50 Hz, `dec` wants
//!    `86.1328125` Hz. This uses [`nearest_index_map`] (no parameters, §13.2.1)
//!    and is exactly the "interpolate outside the network" half.
//! 2. **projection** — a single `Linear(768 → 192)` that maps a HuBERT frame to
//!    a latent frame of the right width.
//!
//! `t_out` is passed in by the caller, never derived here: the pipeline computes
//! `T_total = round(duration × 22050/256)` once and splits it across chunks
//! (§14.4). This keeps the total frame count drift-free.
//!
//! Layout is channel-major `[1, C, T]` on both sides, matching HuBERT's output
//! and `dec`'s input. `Linear` works on the last dim, so time moves to the end
//! for the projection and back.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Float, Int, Tensor};

use super::resample::nearest_index_map;

/// HuBERT feature width (ContentVec final layer).
pub const IN_DIM: usize = 768;
/// `dec` latent width = `initial_channel` = input channels of `conv_pre`.
pub const OUT_DIM: usize = 192;

#[derive(Module, Debug)]
pub struct Adapter {
    linear: Linear,
}

impl Adapter {
    /// New adapter with the documented widths (768 → 192).
    pub fn new(device: &Device) -> Self {
        Self::with_dims(IN_DIM, OUT_DIM, device)
    }

    /// Custom widths — only used by tests; the pipeline uses [`Adapter::new`].
    pub fn with_dims(in_dim: usize, out_dim: usize, device: &Device) -> Self {
        Self {
            linear: LinearConfig::new(in_dim, out_dim).init(device),
        }
    }

    /// `feats: [1, in_dim, T_h]` at 50 Hz → `z: [1, out_dim, t_out]` at the latent rate.
    ///
    /// `t_out` is the target latent length, supplied from outside (§14.4).
    pub fn forward_to(&self, feats: Tensor<3, Float>, t_out: usize) -> Tensor<3, Float> {
        let [_, in_dim, t_in] = feats.dims();
        debug_assert_eq!(in_dim, self.in_dim(), "adapter input width mismatch");
        assert!(t_in > 0 && t_out > 0, "empty feature/latent sequence");

        let device = feats.device();

        // Nearest frame mapping: output frame `t` copies source frame `index(t)`.
        let idx = nearest_index_map(t_in, t_out);
        let idx = Tensor::<1, Int>::from_ints(idx.as_slice(), &device);

        // [1, in_dim, t_out] — time is dim 2, duplicate frames, still differentiable.
        let up = feats.select(2, idx);

        // Linear projects the last dim: [1, in_dim, t_out] -> [1, t_out, in_dim].
        let x = up.swap_dims(1, 2);
        let x = self.linear.forward(x);
        x.swap_dims(1, 2)
    }

    pub fn in_dim(&self) -> usize {
        self.linear.weight.val().dims()[0]
    }

    /// Param id của weight `[out, in]` (rank 2) và bias `[out]` (rank 1) —
    /// dùng cho grad accumulation ngoài module.
    pub fn param_ids(&self) -> (burn::module::ParamId, Option<burn::module::ParamId>) {
        (
            self.linear.weight.id,
            self.linear.bias.as_ref().map(|b| b.id),
        )
    }

    pub fn out_dim(&self) -> usize {
        self.linear.weight.val().dims()[1]
    }

    /// Lưu trọng số adapter đã huấn luyện ra file record của Burn.
    ///
    /// Sau huấn luyện, `z` suy ra hoàn toàn từ adapter + `dec` đã có — không cần
    /// lưu gì của `dec` (nó chỉ đọc).
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        self.clone()
            .into_record()
            .save(path)
            .map_err(|e| e.to_string())
    }

    /// Nạp trọng số adapter đã lưu bằng [`Adapter::save`] (cùng định dạng
    /// burnpack). `device` quyết định nơi dựng tensor.
    pub fn load(path: &std::path::Path, device: &Device) -> Result<Self, String> {
        Self::new(device)
            .try_load_file(path)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    fn device() -> Device {
        Device::flex() // CPU cố định — feature `cuda` unified trong workspace không được ảnh hưởng tới test
    }

    fn ramp(rows: usize, cols: usize) -> Tensor<3, Float> {
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        Tensor::<3, Float>::from_data(TensorData::new(data, [1, rows, cols]), &device())
    }

    #[test]
    fn output_shape_is_channel_major() {
        let a = Adapter::with_dims(8, 3, &device());
        let feats = ramp(8, 5);
        let z = a.forward_to(feats, 11);
        assert_eq!(z.dims(), [1, 3, 11]);
    }

    #[test]
    fn nearest_gather_matches_index_map() {
        // 1→1 layer: z[t] = w · feats[index(t)] + b. Read w and b from the
        // freshly-initialised module so the expectation is weight-agnostic.
        let a = Adapter::with_dims(1, 1, &device());
        let w = a.linear.weight.val().into_data().as_slice::<f32>().unwrap()[0];
        let b = a.linear.bias.as_ref().unwrap().val().into_data().as_slice::<f32>().unwrap()[0];

        let feats = ramp(1, 5); // [1, 1, 5]
        let z = a.forward_to(feats, 11);
        let got: Vec<f32> = z.into_data().as_slice::<f32>().unwrap().to_vec();
        let want: Vec<f32> = nearest_index_map(5, 11)
            .into_iter()
            .map(|i| w * i as f32 + b)
            .collect();
        assert_eq!(got.len(), want.len());
        for (g, wv) in got.iter().zip(want.iter()) {
            assert!((g - wv).abs() < 1e-5, "got {g}, want {wv}");
        }
    }

    #[test]
    fn real_ratio_6_84s_shape() {
        // 6.84 s: HuBERT 342 frames → latent 589 frames.
        let a = Adapter::new(&device());
        let feats = ramp(IN_DIM, 342);
        let z = a.forward_to(feats, 589);
        assert_eq!(z.dims(), [1, OUT_DIM, 589]);
        assert!(z.into_data().as_slice::<f32>().unwrap().iter().all(|v| v.is_finite()));
    }

    /// `save` → `load` phải cho ra adapter y hệt (hợp đồng checkpoint §13.6.4f).
    #[test]
    fn save_and_load_round_trip() {
        let device = device();
        let a = Adapter::new(&device);
        let path = std::env::temp_dir().join("voice_rs_adapter_roundtrip.bpk");
        a.save(&path).unwrap();
        let b = Adapter::load(&path, &device).unwrap();

        let feats = ramp(IN_DIM, 5);
        let za: Vec<f32> = a
            .forward_to(feats.clone(), 7)
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let zb: Vec<f32> = b
            .forward_to(feats, 7)
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        assert_eq!(za, zb);
        let _ = std::fs::remove_file(&path);
    }
}
