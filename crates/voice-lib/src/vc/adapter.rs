//! Adapter HuBERT → latent (P1, §13.6).
//!
//! The only learnable piece of the VC path for 1→1. It does two things:
//!
//! 1. **frame-rate reconciliation** — HuBERT runs at 50 Hz, `dec` wants
//!    `86.1328125` Hz. This uses [`nearest_index_map`] (no parameters, §13.2.1)
//!    and is exactly the "interpolate outside the network" half.
//! 2. **projection** — `Linear(768 → 192)` followed by 3 dilated residual
//!    blocks (kernel 3, dilation 1/2/4, conv2 zero-init) thêm ngữ cảnh thời
//!    gian cho mỗi latent frame — tinh thần `TextEncoder` của Applio (linear
//!    chiếu trước, conv-stack sau). Blocks bắt đầu bằng 0 ⇒ `z` ban đầu yệt
//!    hẹp từ linear, không phá trọng số đã load.
//!
//! `t_out` is passed in by the caller, never derived here: the pipeline computes
//! `T_total = round(duration × 22050/256)` once and splits it across chunks
//! (§14.4). This keeps the total frame count drift-free.
//!
//! Layout is channel-major `[1, C, T]` on both sides, matching HuBERT's output
//! and `dec`'s input. `Linear` works on the last dim, so time moves to the end
//! for the projection and back.

use burn::module::{Module, ParamId};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Initializer, Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::{activation, Device, Float, Int, Tensor};

use super::resample::nearest_index_map;

/// Một khối residual dilated: `y = x + Conv2(LeakyReLU(Conv1(x)))`.
/// Hai conv cùng chiều `out_dim`, kernel 3, dilation `d`, padding `d` (cùng độ dài).
/// `conv2` khởi tạo bằng 0 → bước đầu khối này tạo ra ≈ 0, `z` yệt toàn bộ từ `linear`.
#[derive(Module, Debug)]
struct ResBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl ResBlock {
    fn new(out_dim: usize, dilation: usize, device: &Device) -> Self {
        let conv1 = Conv1dConfig::new(out_dim, out_dim, 3)
            .with_dilation(dilation)
            .with_padding(PaddingConfig1d::Explicit(dilation, dilation))
            .init(device);
        let conv2 = Conv1dConfig::new(out_dim, out_dim, 3)
            .with_dilation(dilation)
            .with_padding(PaddingConfig1d::Explicit(dilation, dilation))
            .with_initializer(Initializer::Zeros)
            .init(device);
        Self { conv1, conv2 }
    }

    fn forward(&self, x: Tensor<3, Float>) -> Tensor<3, Float> {
        let residual = x.clone();
        let r = self
            .conv2
            .forward(activation::leaky_relu(self.conv1.forward(residual), 0.1));
        x + r
    }
}

/// HuBERT feature width (ContentVec final layer).
pub const IN_DIM: usize = 768;
/// `dec` latent width = `initial_channel` = input channels of `conv_pre`.
pub const OUT_DIM: usize = 192;
/// Dilations của 3 block residual — trường fühl tinh thần `TextEncoder` của Applio
/// (linear chiếu trước, stack convolution sau để có ngữ cảnh thời gian).
const TCN_DILATIONS: [usize; 3] = [1, 2, 4];

#[derive(Module, Debug)]
pub struct Adapter {
    linear: Linear,
    block1: ResBlock,
    block2: ResBlock,
    block3: ResBlock,
}

impl Adapter {
    /// New adapter with the documented widths (768 → 192).
    pub fn new(device: &Device) -> Self {
        Self::with_dims(IN_DIM, OUT_DIM, device)
    }

    /// Custom widths — only used by tests; the pipeline uses [`Adapter::new`].
    pub fn with_dims(in_dim: usize, out_dim: usize, device: &Device) -> Self {
        let linear = LinearConfig::new(in_dim, out_dim).init(device);
        let block1 = ResBlock::new(out_dim, TCN_DILATIONS[0], device);
        let block2 = ResBlock::new(out_dim, TCN_DILATIONS[1], device);
        let block3 = ResBlock::new(out_dim, TCN_DILATIONS[2], device);
        Self {
            linear,
            block1,
            block2,
            block3,
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

        // Linear projects the last dim: [1, in_dim, t_out] -> [1, t_out, in_dim] -> [1, out_dim, t_out].
        let mut z = self.linear.forward(up.swap_dims(1, 2)).swap_dims(1, 2);
        // residual TCN thêm ngữ cảnh thời gian (tinh thần TextEncoder của Applio:
        // linear chiếu trước, conv-stack sau). Blocks bắt đầu bằng 0 ⇒ z đầu tiên yệt hẹp từ linear.
        z = self.block1.forward(z);
        z = self.block2.forward(z);
        z = self.block3.forward(z);
        z
    }

    pub fn in_dim(&self) -> usize {
        self.linear.weight.val().dims()[0]
    }

    /// Param id + rank (2 = weight, 1 = bias) của mọi tensor học: linear + 3 block.
    /// Dùng cho grad accumulation bên ngoài module (`train.rs`).
    pub fn param_ids(&self) -> Vec<(ParamId, usize)> {
        let mut ids: Vec<(ParamId, usize)> = Vec::with_capacity(8);
        ids.push((self.linear.weight.id, 2));
        if let Some(b) = &self.linear.bias {
            ids.push((b.id, 1));
        }
        for b in [&self.block1, &self.block2, &self.block3] {
            ids.push((b.conv1.weight.id, 2));
            if let Some(bi) = &b.conv1.bias {
                ids.push((bi.id, 1));
            }
            ids.push((b.conv2.weight.id, 2));
            if let Some(bi) = &b.conv2.bias {
                ids.push((bi.id, 1));
            }
        }
        ids
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
    ///
    /// Checkpoint v2.2.23 và trước chỉ chứa `linear` (record cũ thiếu
    /// `block1/2/3`) → nạp mới thất bại, ta trả về adapter khởi tạo lại:
    /// blocks zero-init (z ban đầu yệt hẹp từ linear), chỉ cần huấn luyện lại.
    pub fn load(path: &std::path::Path, device: &Device) -> Result<Self, String> {
        match Self::new(device).try_load_file(path) {
            Ok(adapter) => Ok(adapter),
            Err(e) => {
                eprintln!(
                    "⚠️  adapter '{}' không khớp định dạng mới ({e}); \
                     dùng adapter khởi tạo mới (TCN blocks zero-init, huấn luyện lại từ đầu)",
                    path.display()
                );
                Ok(Self::new(device))
            }
        }
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

    #[test]
    fn tcn_conv2_zero_init() {
        let a = Adapter::new(&device());
        for b in [&a.block1, &a.block2, &a.block3] {
            let data = b.conv2.weight.val().into_data();
            let w = data.as_slice::<f32>().unwrap();
            assert!(
                w.iter().all(|&v| v == 0.0),
                "conv2 của block phải zero-init để z đầu tiên yệt hẹp từ linear"
            );
        }
    }

    #[test]
    fn tcn_param_ids_cover_linear_and_blocks() {
        let a = Adapter::new(&device());
        let ids = a.param_ids();
        // linear (weight+bias) + 3 block × 2 conv × (weight+bias) = 2 + 12 = 14
        assert_eq!(
            ids.len(),
            14,
            "param_ids phải bao gồm linear + TCN blocks: {ids:?}"
        );
        assert!(ids.iter().all(|(_, r)| *r == 1 || *r == 2));
    }

    #[test]
    fn tcn_forward_equals_linear_at_init() {
        // Với conv2 zero-init, mỗi block trả về x + 0 ⇒ forward_to == linear thuần.
        let a = Adapter::new(&device());
        let feats = ramp(IN_DIM, 9);
        let z_tcn = a.forward_to(feats.clone(), 13);

        // Tính toán linear thuần theo cùng đường pipeline, không qua block.
        let device = device();
        let [_, in_dim, t_in] = feats.dims();
        assert_eq!(in_dim, IN_DIM);
        let idx = nearest_index_map(t_in, 13);
        let idx = Tensor::<1, Int>::from_ints(idx.as_slice(), &device);
        let up = feats.select(2, idx);
        let x = up.swap_dims(1, 2);
        let x = a.linear.forward(x);
        let z_linear = x.swap_dims(1, 2);

        let zt: Vec<f32> = z_tcn
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let zl: Vec<f32> = z_linear
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        assert_eq!(zt.len(), zl.len());
        for (a_, b_) in zt.iter().zip(zl.iter()) {
            assert!(
                (a_ - b_).abs() < 1e-6,
                "TCN chưa zero-init: {a_} vs {b_}"
            );
        }
    }
}
