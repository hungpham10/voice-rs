//! `train overfit` — GĐ4: overfit `Adapter` trên **một câu** (§13.6.2).
//!
//! Đây là hành vi của bin `train_adapter` cũ, nay gom vào subcommand của
//! `voice-train`. Mục đích là **integration test**: xác nhận `Adapter` +
//! `dec` (đóng băng) + loss log-mel chạy đúng, loss phải **giảm**.
//!
//! `dec` (VITS HiFi-GAN) **đóng băng**; chỉ `Linear(768→192)` của `Adapter` học.
//! Loss = `L1` giữa log-mel của wav sinh ra và wav đích, cả hai ở 22.05 kHz
//! (wav đích 16 kHz được resample lên ngoài mạng, §13.2.1).

use std::path::PathBuf;

use burn::module::Module;
use burn::tensor::{Device, Tensor, TensorData};
use voice_lib::vc::{resample_linear, Adapter};
use voice_mel::MelConfig;
use voice_models::dec::Generator;

use crate::{data, mel};

/// 6.84 s × 22050/256, làm tròn — §14.4.
pub const DEFAULT_T_OUT: usize = 589;

/// Cấu hình `train overfit`.
#[derive(Debug, Clone)]
pub struct OverfitConfig {
    pub dec_dir: PathBuf,
    pub wav: PathBuf,
    pub feats: PathBuf,
    pub out_wav: PathBuf,
    pub out_adapter: PathBuf,
    /// Số frame latent đem ra huấn luyện (tự co theo dữ liệu thật).
    pub seg: usize,
    pub steps: usize,
    pub lr: f64,
    /// `0` = tắt; `>0` = cứ mỗi N bước lưu `checkpoint.bpk` (cạnh `out_adapter`).
    pub checkpoint_every: usize,
}

/// Kết quả `train overfit`.
#[derive(Debug, Clone)]
pub struct OverfitSummary {
    pub first: f64,
    pub last: f64,
    /// `first / last` (>1 nghĩa là loss giảm).
    pub ratio: f64,
    pub t_out: usize,
    pub out_wav: PathBuf,
    pub out_adapter: PathBuf,
    /// File checkpoint định kỳ (nếu `checkpoint_every > 0`).
    pub checkpoint_path: Option<PathBuf>,
}

/// Huấn luyện overfit. `device` là CPU/… **trước** autodiff (hàm tự bật).
pub fn train_overfit(
    cfg: &OverfitConfig,
    device: &Device,
) -> Result<OverfitSummary, Box<dyn std::error::Error>> {
    if !cfg.dec_dir.join("config.json").exists() {
        return Err(format!(
            "{} chưa có — chạy ./scripts/fetch_assets.sh",
            cfg.dec_dir.display()
        )
        .into());
    }
    if !cfg.feats.exists() {
        return Err(format!(
            "{} chưa có — chạy `embed <wav> <out.npy>` trước",
            cfg.feats.display()
        )
        .into());
    }

    let device = device.clone().autodiff();

    // `dec` đóng băng: `freeze()` tắt require_grad toàn cây → không gradient nào
    // chảy vào 14.33 M tham số của nó (khớp GĐ2).
    let gen = Generator::load(&cfg.dec_dir, &device)?.freeze();
    // Chỉ adapter học. `unfreeze()` đặt require_grad = true cho `Linear`.
    let mut adapter = Adapter::new(&device).unfreeze();
    println!(
        "dec (đóng băng) = {:.2} M tham số · adapter (học) = {} tham số\n",
        gen.num_params() as f64 / 1e6,
        adapter.in_dim() * adapter.out_dim() + adapter.out_dim()
    );

    // ---- dữ liệu
    let mel_cfg = MelConfig::ljs();
    let basis = mel::mel_basis(&mel_cfg, &device);

    let feats_full = data::features_channel_major(&cfg.feats, &device)?; // [1,768,T_h]
    println!("features {:?}", feats_full.dims());

    let (mut target_wav, sr) = data::read_wav_mono(&cfg.wav)?;
    if sr != 16000 {
        eprintln!("⚠️  wav nguồn {sr} Hz, kỳ vọng 16000 Hz");
    }
    target_wav = resample_linear(&target_wav, sr, 22050);

    // `seg` = số frame latent để huấn luyện (mặc định cả câu). Luôn co lại theo
    // lượng dữ liệu thật (wav đích + features) để không panic khi audio ngắn.
    let t_wav_max = target_wav.len() / 256;
    let t_feat_max = (feats_full.dims()[2] as f64 * 86.1328125 / 50.0).floor() as usize;
    let t_out = cfg.seg.min(t_wav_max).min(t_feat_max).max(1);
    let t_h = ((t_out as f64 * 50.0 / 86.1328125).round() as usize).min(feats_full.dims()[2]);
    let feats = feats_full.narrow(2, 0, t_h); // [1,768,t_h]
    let n = t_out * 256; // wav sinh ra có đúng t_out·256 mẫu
    target_wav.truncate(n);
    let target = Tensor::<2>::from_data(TensorData::new(target_wav, [1, n]), &device);
    println!("huấn luyện {t_out} frame latent ({t_h} frame HuBERT) · đích {n} mẫu\n");

    // ---- huấn luyện
    let checkpoint_path = if cfg.checkpoint_every > 0 {
        let p = cfg.out_adapter.with_file_name("checkpoint.bpk");
        Some(p)
    } else {
        None
    };

    let mut optim = burn::optim::AdamConfig::new().init();
    let mut first = f64::NAN;
    let mut last = f64::NAN;

    for step in 0..cfg.steps {
        let z = adapter.forward_to(feats.clone(), t_out); // [1,192,t_out]
        let wav = gen.forward(z).reshape([1, n]); // [1,n]
        let loss = mel::mel_l1(wav, target.clone(), &mel_cfg, &basis); // Tensor<1>

        let lv: f64 = loss.clone().into_scalar::<f32>() as f64;
        if step == 0 {
            first = lv;
        }
        last = lv;

        let grads = burn::optim::GradientsParams::from_grads(loss.backward(), &adapter);
        adapter = optim.step(cfg.lr, adapter, grads);

        if let Some(path) = &checkpoint_path {
            if (step + 1) % cfg.checkpoint_every == 0 {
                adapter.save(path)?;
                println!("  ↳ checkpoint step {} → {}", step + 1, path.display());
            }
        }

        if step % 25 == 0 || step == cfg.steps - 1 {
            println!("step {step:>3}  mel-L1 = {lv:.5}");
        }
    }

    let ratio = first / last;
    println!("\nmel-L1: {first:.5} → {last:.5}  (giảm {ratio:.2}×)");

    // ---- nghe thử: sinh wav cuối bằng adapter đã học
    let z = adapter.forward_to(feats, t_out);
    let wav = gen
        .forward(z)
        .reshape([1, n])
        .into_data()
        .as_slice::<f32>()?
        .to_vec();
    data::write_wav_mono(&cfg.out_wav, &wav, 22050)?;
    adapter.save(&cfg.out_adapter)?;
    println!(
        "đã ghi {} và {}",
        cfg.out_wav.display(),
        cfg.out_adapter.display()
    );

    Ok(OverfitSummary {
        first,
        last,
        ratio,
        t_out,
        out_wav: cfg.out_wav.clone(),
        out_adapter: cfg.out_adapter.clone(),
        checkpoint_path,
    })
}
