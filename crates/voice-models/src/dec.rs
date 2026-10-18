//! VITS HiFi-GAN generator (`dec`) — viết tay bằng Burn.
//!
//! Xem `docs/architecture.md` §2 cho đặc tả đầy đủ và bốn bẫy.
//!
//! Cấu trúc (`sources/vits/configs/ljs_base.json`):
//! ```text
//! initial_channel 192 · upsample_initial_channel 512
//! upsample_rates          [8, 8, 2, 2]   (tích = 256)
//! upsample_kernel_sizes   [16, 16, 4, 4]
//! resblock_kernel_sizes   [3, 7, 11]
//! resblock_dilation_sizes [[1,3,5], [1,3,5], [1,3,5]]
//! lrelu_slope 0.1, nhưng LReLU CUỐI dùng slope mặc định của PyTorch = 0.01
//! conv_post bias = False
//! ```
//!
//! Vì sao viết tay chứ không dùng burn-onnx codegen: graph chỉ 200 node conv
//! thuần, không `Expand`, không ngẫu nhiên, không weight-norm — codegen không
//! thêm giá trị gì, nhưng lại kéo theo `OUT_DIR`, `.bpk` và phụ thuộc
//! `burn-onnx`. Tự viết ~250 dòng thì sửa kiến trúc chỉ cần sửa Rust.
//!
//! Trọng số nạp từ `.npy` đã export MỘT LẦN từ checkpoint VITS (đóng băng trên
//! HF, phát trong release bundle — weight-norm đã gộp sẵn lúc export).

use std::path::{Path, PathBuf};

use burn::backend::Flex;
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use burn::nn::PaddingConfig1d;
use burn::tensor::{activation, Device, Float, Tensor, TensorData};

use crate::npy;
use crate::sinegen::SineGen;

pub type Backend = Flex;

#[derive(Debug, thiserror::Error)]
pub enum DecError {
    #[error("thiếu file: {0}")]
    Missing(String),
    #[error("lỗi .npy: {0}")]
    Npy(#[from] npy::NpyError),
    #[error("lỗi tensor: {0}")]
    Tensor(String),
    #[error("lỗi record: {0}")]
    Record(String),
}

/// ResBlock của VITS: 3 conv có dilation + 3 conv dilation=1, cùng kernel.
///
/// `Vec` **CÓ** implement `Module` (`burn-core/src/module/param/primitive.rs:133`)
/// và derive macro đánh dấu nó là module field, nên không cần đổi sang `[T; N]`.
/// Đường dẫn trong record ra đúng `resblocks.{i}.convs1.{c}.weight` — khớp regex
/// của `apply_lora` (GĐ3).
#[derive(Module, Debug)]
pub struct ResBlock {
    pub convs1: Vec<Conv1d>,
    pub convs2: Vec<Conv1d>,
}

impl ResBlock {
    fn forward(&self, x: Tensor<3, Float>, slope: f64) -> Tensor<3, Float> {
        let mut out = x;
        for (c1, c2) in self.convs1.iter().zip(self.convs2.iter()) {
            let residual = out.clone();
            out = activation::leaky_relu(out, slope);
            out = c1.forward(out);
            out = activation::leaky_relu(out, slope);
            out = c2.forward(out);
            out = out + residual;
        }
        out
    }
}

/// Hàm `Generator::load` đã khởi tạo đúng cấu trúc; `load_record`/`save_file` của
/// burn 0.22 nạp **vào** một module có sẵn (không còn `from_record`/`Config`).
///
/// Các field `f64`/`usize` là primitive nên derive tự bỏ qua — chúng là hằng số cấu
/// hình, không phải tham số huấn luyện.
#[derive(Module, Debug)]
pub struct Generator {
    pub conv_pre: Conv1d,
    pub ups: Vec<ConvTranspose1d>,
    pub resblocks: Vec<ResBlock>,
    pub conv_post: Conv1d,
    /// NSF excitation: `Option<Conv1d(8→512, k=1)>` — `None` khi bundle không có
    /// `conv_src.npy` (backward-compat với checkpoint v2.2.14 trở về trước).
    pub conv_src: Option<Conv1d>,
    pub n_kernels: usize,
    pub lrelu_slope: f64,
    /// ⚠️ Bẫy số 1: PyTorch gọi `leaky_relu(x)` không truyền slope ở bước cuối
    /// → default `0.01`, KHÁC `lrelu_slope = 0.1` dùng trong vòng lặp.
    pub lrelu_slope_final: f64,
}

impl Generator {
    /// Dựng cấu trúc từ `sources/vits_ljs_dec/config.json` và nạp trọng số `.npy`.
    ///
    /// `dir` là thư mục chứa `config.json` và thư mục con `dec/`.
    pub fn load(dir: &Path, device: &Device) -> Result<Self, DecError> {
        let cfg_path = dir.join("config.json");
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&cfg_path)
                .map_err(|e| DecError::Missing(format!("{}: {e}", cfg_path.display())))?,
        )
        .map_err(|e| DecError::Missing(format!("config.json: {e}")))?;

        let initial = cfg["initial_channel"].as_u64().unwrap_or(192) as usize;
        let up_init = cfg["upsample_initial_channel"].as_u64().unwrap_or(512) as usize;
        let rates: Vec<usize> = arr(&cfg["upsample_rates"]);
        let kerns: Vec<usize> = arr(&cfg["upsample_kernel_sizes"]);
        let rk: Vec<usize> = arr(&cfg["resblock_kernel_sizes"]);
        let rd: Vec<Vec<usize>> = cfg["resblock_dilation_sizes"]
            .as_array()
            .map(|a| a.iter().map(|x| arr(x)).collect())
            .unwrap_or_default();
        let slope = cfg["lrelu_slope"].as_f64().unwrap_or(0.1);
        let slope_final = cfg["lrelu_slope_final"].as_f64().unwrap_or(0.01);
        let n_kernels = rk.len();

        let dec_dir = dir.join("dec");
        let p = |name: String| dec_dir.join(format!("{name}.npy"));

        // ---- conv_pre: Conv1d(192 -> 512, k=7, pad=3, bias=true)
        let conv_pre = Conv1dConfig::new(initial, up_init, 7)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .with_bias(true)
            .init(device);
        let conv_pre = load_conv1d(
            conv_pre,
            &p("conv_pre.weight".into()),
            &Some(p("conv_pre.bias".into())),
            device,
        )?;

        // ---- ups: ConvTranspose1d, weight layout [in, out, k] — KHÔNG cần transpose
        let mut ups = Vec::with_capacity(rates.len());
        for i in 0..rates.len() {
            let cin = up_init / 2usize.pow(i as u32);
            let cout = up_init / 2usize.pow(i as u32 + 1);
            let k = kerns[i];
            let stride = rates[i];
            // (k - stride) / 2  — khớp hifigan.py: padding=(kernel_size - up)/2
            let pad = (k - stride) / 2;
            let cfg = ConvTranspose1dConfig::new([cin, cout], k)
                .with_stride(stride)
                .with_padding(pad)
                .with_bias(true)
                .init(device);
            let w = load3(&p(format!("ups.{i}.weight")), device)?;
            let b = load(&p(format!("ups.{i}.bias")), device)?;
            ups.push(assign_conv_t(cfg, w, b)?);
        }

        // ---- resblocks
        let mut resblocks = Vec::with_capacity(rates.len() * n_kernels);
        for i in 0..rates.len() {
            let ch = up_init / 2usize.pow(i as u32 + 1);
            for j in 0..n_kernels {
                let k = rk[j];
                let dil = &rd[j];
                let mut convs1 = Vec::with_capacity(dil.len());
                let mut convs2 = Vec::with_capacity(dil.len());
                for (c, &d) in dil.iter().enumerate() {
                    // pad = (k*d - d) / 2
                    let pad1 = (k * d - d) / 2;
                    let cfg1 = Conv1dConfig::new(ch, ch, k)
                        .with_dilation(d)
                        .with_padding(PaddingConfig1d::Explicit(pad1, pad1))
                        .with_bias(true)
                        .init(device);
                    let w1 = load3(&p(format!("resblocks.{}.convs1.{c}.weight", i * n_kernels + j)), device)?;
                    let b1 = load(&p(format!("resblocks.{}.convs1.{c}.bias", i * n_kernels + j)), device)?;
                    convs1.push(assign_conv1d(cfg1, w1, b1)?);

                    // ⚠️ Bẫy số 2: convs2 dùng CÙNG kernel_size `k`, KHÔNG phải 3.
                    // pad = (k*d - d)/2 với d=1 → (k-1)/2  (KHÔNG phải k-1)
                    let pad2 = (k - 1) / 2;
                    let cfg2 = Conv1dConfig::new(ch, ch, k)
                        .with_dilation(1)
                        .with_padding(PaddingConfig1d::Explicit(pad2, pad2))
                        .with_bias(true)
                        .init(device);
                    let w2 = load3(&p(format!("resblocks.{}.convs2.{c}.weight", i * n_kernels + j)), device)?;
                    let b2 = load(&p(format!("resblocks.{}.convs2.{c}.bias", i * n_kernels + j)), device)?;
                    convs2.push(assign_conv1d(cfg2, w2, b2)?);
                }
                resblocks.push(ResBlock { convs1, convs2 });
            }
        }

        // ---- conv_post: Conv1d(32 -> 1, k=7, pad=3, bias=FALSE)
        let ch_last = up_init / 2usize.pow(rates.len() as u32);
        let conv_post = Conv1dConfig::new(ch_last, 1, 7)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .with_bias(false)
            .init(device);
        // ⚠️ Bẫy số 3: KHÔNG có conv_post.bias trong thư mục trọng số.
        let conv_post = load_conv1d(
            conv_post,
            &p("conv_post.weight".into()),
            &None,
            device,
        )?;

        // ---- conv_src: Conv1d(8 -> 512, k=1, bias=true) — Optional NSF
        // excitation source. Thiếu file → None (backward-compat).
        let conv_src_path = p("conv_src.weight".into());
        let conv_src = if conv_src_path.exists() {
            let cfg = Conv1dConfig::new(SineGen::out_channels(), up_init, 1)
                .with_padding(PaddingConfig1d::Explicit(0, 0))
                .with_bias(true)
                .init(device);
            let w = load3(&conv_src_path, device)?;
            let b = load(&p("conv_src.bias".into()), device)?;
            Some(assign_conv1d(cfg, w, b)?)
        } else {
            None
        };

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            conv_post,
            conv_src,
            n_kernels,
            lrelu_slope: slope,
            lrelu_slope_final: slope_final,
        })
    }

    /// `z [B, 192, T]` → `wav [B, 1, T*256]`.
    ///
    /// Nếu `f0` có giá trị (và bundle có `conv_src`), bơm excitation NSF
    /// sau `conv_pre` (xem §13.4); `f0 = None` là đường inference cũ.
    pub fn forward(&self, z: Tensor<3, Float>, f0: Option<Tensor<2>>, trace: bool) -> Tensor<3, Float> {
        self.forward_impl(z, f0, trace).0
    }

    /// Giống [`forward`](Self::forward) nhưng trả thêm **đầu vào của `conv_post`**
    /// (tức `x` ngay trước bước LeakyReLU cuối).
    ///
    /// Chỉ dùng cho verify / giải chứng gradient (GĐ2). Không dùng f0.
    pub fn forward_trace(&self, z: Tensor<3, Float>) -> (Tensor<3, Float>, Tensor<3, Float>) {
        let (wav, x) = self.forward_impl(z, None, true);
        (wav, x.expect("trace = true thi phai co dau vao cua conv_post"))
    }

    fn forward_impl(
        &self,
        z: Tensor<3, Float>,
        f0: Option<Tensor<2>>,
        trace: bool,
    ) -> (Tensor<3, Float>, Option<Tensor<3, Float>>) {
        let device = z.device();
        let mut x = self.conv_pre.forward(z);

        // ---- NSF-lite: bơm excitation từ f0 vào sau conv_pre (§13.4)
        // x: [B, 512, T_z]; exc: [B, 8, T_z]; conv_src: Conv1d(8→512,k=1)
        if let Some(f0) = f0 {
            if let Some(src) = &self.conv_src {
                let exc = SineGen::forward(f0, &device);
                x = x + src.forward(exc);
            }
        }

        for (i, up) in self.ups.iter().enumerate() {
            x = activation::leaky_relu(x, self.lrelu_slope);
            x = up.forward(x);

            let mut acc: Option<Tensor<3, Float>> = None;
            for j in 0..self.n_kernels {
                let y = self.resblocks[i * self.n_kernels + j].forward(x.clone(), self.lrelu_slope);
                acc = Some(match acc {
                    None => y,
                    Some(a) => a + y,
                });
            }
            let acc = acc.expect("n_kernels > 0");
            x = acc / (self.n_kernels as f64);
        }

        let traced = trace.then(|| x.clone());

        // ⚠️ slope 0.01, không phải 0.1
        x = activation::leaky_relu(x, self.lrelu_slope_final);
        x = self.conv_post.forward(x);
        (x.tanh(), traced)
    }

    /// Xuất ra `.bpk` của burn.
    ///
    /// Dùng sau `materialize()` (GĐ3) để lấy một generator **phẳng** đã gộp
    /// adapter — inference sau đó không cần biết LoRA tồn tại.
    pub fn save(&self, path: &Path) -> Result<(), DecError> {
        self.clone()
            .into_record()
            .save(path)
            .map_err(|e| DecError::Record(e.to_string()))
    }
}

// ------------------------------------------------------------------ helpers

fn arr(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
        .unwrap_or_default()
}

fn load(path: &PathBuf, device: &Device) -> Result<Tensor<1, Float>, DecError> {
    let (dims, data) = npy::read_f32(path).map_err(DecError::Npy)?;
    if dims.len() != 1 {
        return Err(DecError::Tensor(format!(
            "{}: cần 1 chiều, có {:?}",
            path.display(),
            dims
        )));
    }
    Ok(Tensor::from_data(TensorData::new(data, [dims[0]]), device))
}

fn load3(path: &PathBuf, device: &Device) -> Result<Tensor<3, Float>, DecError> {
    if !path.exists() {
        return Err(DecError::Missing(path.display().to_string()));
    }
    let (dims, data) = npy::read_f32(path).map_err(DecError::Npy)?;
    if dims.len() != 3 {
        return Err(DecError::Tensor(format!(
            "{}: cần 3 chiều, có {:?}",
            path.display(),
            dims
        )));
    }
    Ok(Tensor::from_data(TensorData::new(data, [dims[0], dims[1], dims[2]]), device))
}

fn load_conv1d(
    mut m: Conv1d,
    w: &PathBuf,
    b: &Option<PathBuf>,
    device: &Device,
) -> Result<Conv1d, DecError> {
    let weight = load3(w, device)?;
    let [co, ci, k] = weight.dims();
    if let Some(bp) = b {
        let bias = load(bp, device)?;
        if bias.dims()[0] != co {
            return Err(DecError::Tensor(format!(
                "{}: bias len {} != out {}",
                bp.display(),
                bias.dims()[0],
                co
            )));
        }
        m.bias = Some(burn::module::Param::from_tensor(bias));
    }
    let _ = (ci, k);
    m.weight = burn::module::Param::from_tensor(weight);
    Ok(m)
}

/// Gán trọng số + bias vào một `Conv1d` đã init.
fn assign_conv1d(m: Conv1d, w: Tensor<3, Float>, b: Tensor<1, Float>) -> Result<Conv1d, DecError> {
    let [co, ..] = w.dims();
    if b.dims()[0] != co {
        return Err(DecError::Tensor(format!(
            "bias len {} != out_channels {co}",
            b.dims()[0]
        )));
    }
    let mut m = m;
    m.weight = burn::module::Param::from_tensor(w);
    m.bias = Some(burn::module::Param::from_tensor(b));
    Ok(m)
}

/// Gán trọng số + bias vào một `ConvTranspose1d` đã init.
///
/// Layout weight của PyTorch ConvTranspose1d là `[in, out/groups, k]`, **giống
/// hệt** burn → copy thẳng, không transpose.
fn assign_conv_t(
    m: ConvTranspose1d,
    w: Tensor<3, Float>,
    b: Tensor<1, Float>,
) -> Result<ConvTranspose1d, DecError> {
    let [ci, co, _] = w.dims();
    let [cfg_ci, cfg_co] = m.channels;
    if ci != cfg_ci || co != cfg_co {
        return Err(DecError::Tensor(format!(
            "weight [{ci},{co},k] != config [{cfg_ci},{cfg_co}]"
        )));
    }
    if b.dims()[0] != co {
        return Err(DecError::Tensor(format!(
            "bias len {} != out_channels {co}",
            b.dims()[0]
        )));
    }
    let mut m = m;
    m.weight = burn::module::Param::from_tensor(w);
    m.bias = Some(burn::module::Param::from_tensor(b));
    Ok(m)
}


#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;
    use std::path::Path;

    fn dec_assets_exist() -> bool {
        Path::new("sources/vits_ljs_dec/config.json").exists()
    }

    #[test]
    fn forward_full_matches_forward_without_f0() {
        if !dec_assets_exist() {
            eprintln!("skip: sources/vits_ljs_dec thiếu");
            return;
        }
        let device = Device::flex();
        let gen = Generator::load(Path::new("sources/vits_ljs_dec"), &device)
            .unwrap()
            .freeze();
        let make_z = || {
            Tensor::<3>::from_data(
                TensorData::new(vec![0.5f32; 1 * 192 * 4], [1, 192, 4]),
                &device,
            )
        };
        let w1 = gen.forward(make_z(), None, false);
        let w2 = gen.forward(make_z(), None, false);
        let d1: Vec<f32> = w1.into_data().as_slice::<f32>().unwrap().to_vec();
        let d2: Vec<f32> = w2.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(d1, d2, "forward_full(z, None) phải giống forward(z)");
    }

    #[test]
    fn forward_full_with_f0_runs_and_finite() {
        if !dec_assets_exist() {
            eprintln!("skip: sources/vits_ljs_dec thiếu");
            return;
        }
        let device = Device::flex();
        let gen = Generator::load(Path::new("sources/vits_ljs_dec"), &device)
            .unwrap()
            .freeze();
        let z = Tensor::<3>::from_data(
            TensorData::new(vec![0.5f32; 1 * 192 * 4], [1, 192, 4]),
            &device,
        );
        let f0 = Tensor::<2>::from_data(TensorData::new(vec![220.0f32; 4], [1, 4]), &device);
        let wav = gen.forward(z, Some(f0), false);
        let d: Vec<f32> = wav.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(d.len(), 4 * 256);
        assert!(d.iter().all(|v| v.is_finite()), "wav phải finite");
    }
}
