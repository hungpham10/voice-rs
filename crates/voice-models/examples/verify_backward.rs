//! GĐ2 — chứng minh **backward thật sự chạy** trên generator VITS.
//!
//! §13.9 của `docs/architecture.md` mới chỉ *đọc source* khẳng định mọi op đều có
//! backward. Ví dụ này chạy thật để kiểm chứng, đồng thời chặn NaN (bẫy §8.2).
//!
//! Run: cargo run --release -p voice-models --example verify_backward
//!
//! ## Vì sao `dec.rs` không cần sửa cho ví dụ này
//!
//! Burn 0.22 đã bỏ tham số backend khỏi `Tensor`
//! (`burn-tensor/src/tensor/api/base.rs:76`):
//!
//! ```text
//! pub struct Tensor<const D: usize, K = Float> { primitive: BridgeTensor, .. }
//! ```
//!
//! Backend được chọn **lúc chạy** từ `Device`. Nên `Generator` viết
//! `Tensor<3, Float>` chạy được trên cả device thường lẫn device đã bật autodiff —
//! chỉ khác `Device::default().autodiff()`.

use std::path::PathBuf;

use burn::module::{Module, Param};
use burn::tensor::{Device, Gradients, Tensor, TensorData};
use voice_models::dec::Generator;

mod common;
use common::read_npz;

const DIR: &str = "sources/vits_dec";
const T: usize = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(DIR);
    if !dir.join("ref_input.npz").exists() {
        return Err(format!("{DIR}/ref_input.npz chưa có — chạy scripts/export_vits_dec.py").into());
    }

    // ⚠️ `Device::default()` KHÔNG bật autodiff. Thiếu `.autodiff()` thì `backward()`
    // panic với "requires a tracked autodiff tensor".
    let device = Device::default().autodiff();
    assert!(device.is_autodiff(), "device chưa bật autodiff");
    println!("device autodiff = {}\n", device.is_autodiff());

    let gen = Generator::load(&dir, &device)?;
    println!(
        "num_params = {}   ({:.2} M)",
        gen.num_params(),
        gen.num_params() as f64 / 1e6
    );

    // ---- input lấy từ ground truth, scale nhỏ lại cho gradient khỏi bão hòa
    let ref_z = read_npz(&dir.join("ref_input.npz"))?;
    let (z_flat, _) = ref_z
        .get(&format!("T{T}"))
        .ok_or_else(|| format!("khong co khoa T{T} trong ref_input.npz"))?;

    let z = Tensor::from_data(TensorData::new(z_flat.clone(), [1usize, 192, T]), &device)
        .mul_scalar(0.03125);

    // ---- forward
    let wav = gen.forward(z);
    let [b, c, n] = wav.dims();
    assert_eq!((b, c, n), (1, 1, T * 256), "sai shape sau forward");
    println!("forward OK: wav [1, 1, {n}]");

    // ---- loss = Σ wav²
    //
    // Chọn L2 vì gradient = 2·wav — biết trước độ lớn nên kiểm tra chéo được bằng
    // tay, thay vì chỉ "khác 0".
    let loss = wav.clone().powf_scalar(2.0).sum();
    let loss_val: f64 = loss.clone().into_scalar::<f32>().into();
    println!("loss = Σ wav² = {loss_val:.6e}");
    assert!(loss_val.is_finite() && loss_val > 0.0, "loss hong");

    // ---- backward
    let mut grads = loss.backward();
    println!("backward OK\n");

    // ---- kiểm tra gradient: khác 0 VÀ finite
    println!(
        "{:<34} {:>13} {:>13}  {}",
        "tham so", "|grad|", "|grad|max", "ket qua"
    );
    let mut all_pass = true;

    all_pass &= check(&gen.conv_pre.weight, "conv_pre.weight", &mut grads).is_some();
    all_pass &= check(&gen.ups[0].weight, "ups.0.weight", &mut grads).is_some();
    all_pass &= check(
        &gen.resblocks[0].convs1[0].weight,
        "resblocks.0.convs1.0.weight",
        &mut grads,
    )
    .is_some();
    all_pass &= check(
        &gen.resblocks[11].convs2[2].weight,
        "resblocks.11.convs2.2.weight",
        &mut grads,
    )
    .is_some();
    all_pass &= check(
        gen.conv_pre.bias.as_ref().expect("conv_pre co bias"),
        "conv_pre.bias",
        &mut grads,
    )
    .is_some();
    // Giữ lại chuẩn của conv_post để in ra — `grad_remove` chỉ lấy được **một**
    // lần, nên không thể gọi lần hai.
    let g_post = check(&gen.conv_post.weight, "conv_post.weight", &mut grads);
    all_pass &= g_post.is_some();
    let _ = g_post;

    // ---- kiểm tra chéo bằng công thức tường minh
    //
    // conv_post là `Conv1d(32 → 1, k=7, pad=3)`, tức tương quan chéo có đệm trái 3:
    //     out[t] = Σ_{k=0..6} w[k] · x[t + k − 3]
    // loss = Σ out[t]², và out = tanh(conv_post(·)) cho:
    //     g[t]  = dLoss/dout[t] = 2·out[t]·(1 − out[t]²)
    //     dLoss/dw[k] = Σ_t g[t] · x[t + k − 3]
    // Không có xấp xỉ nào — so trực tiếp với gradient burn.
    //
    // ⚠️ Trước đây tôi dùng ngưỡng `2·‖wav‖` — sai, vì gradient quy về `x` (32 kênh)
    // chứ không quy về `wav`.
    let dev1 = Device::default().autodiff();
    let gen1 = Generator::load(&dir, &dev1)?;
    let (z1_flat, _) = ref_z.get("T1").ok_or("thieu khoa T1")?;
    let z1 =
        Tensor::from_data(TensorData::new(z1_flat.clone(), [1usize, 192, 1]), &dev1).mul_scalar(0.03125);

    let (wav1, x1) = gen1.forward_trace(z1);
    let [_, xc, xt] = x1.dims();
    let (out1, x1v) = (
        wav1.clone().into_data().as_slice::<f32>()?.to_vec(),
        x1.into_data().as_slice::<f32>()?.to_vec(),
    );
    assert_eq!(out1.len(), xt);

    let loss1 = wav1.powf_scalar(2.0).sum();
    let mut grads1 = loss1.backward();
    let g = gen1
        .conv_post
        .weight
        .grad_remove(&mut grads1)
        .expect("khong co gradient conv_post.weight")
        .into_data()
        .as_slice::<f32>()?
        .to_vec();

    // g[t] = 2·out·(1 − out²)
    let gl: Vec<f64> = out1
        .iter()
        .map(|o| 2.0 * (*o as f64) * (1.0 - (*o as f64).powi(2)))
        .collect();

    // ⚠️ `x1` là giá trị **trước** bước LeakyReLU cuối. `conv_post` thực ra nhận
    // `leaky_relu(x, 0.01)` — bỏ bước này thì mọi phần tử âm lệch đúng 100×.
    let slope = gen1.lrelu_slope_final;
    let xr: Vec<f64> = x1v
        .iter()
        .map(|v| {
            let v = *v as f64;
            if v > 0.0 { v } else { slope * v }
        })
        .collect();

    // grad w[c][k] = Σ_t gl[t] · x[c][t + k − 3]
    let mut worst = 0.0f64;
    for (idx, got) in g.iter().enumerate() {
        let c = idx / 7;
        let k = idx % 7;
        let mut acc = 0.0f64;
        for (t, gv) in gl.iter().enumerate() {
            let ti = t + k - 3;
            if ti >= 0 && ti < xt {
                acc += gv * xr[c * xt + ti];
            }
        }
        worst = worst.max((*got as f64 - acc).abs());
    }

    let cross_ok = worst < 1e-6;
    println!(
        "\nkiem tra cheo conv_post: max|grad - tong tich tay| = {worst:.3e}  ({xc} kenh, xt={xt})  {}",
        if cross_ok { "PASS" } else { "FAIL" }
    );
    all_pass &= cross_ok;

    println!("\n{}", if all_pass { "PASS tat ca" } else { "CO FAIL" });
    if !all_pass {
        std::process::exit(1);
    }
    Ok(())
}

/// Lấy gradient của một `Param`, kiểm nó **khác 0** *và* **finite**.
///
/// Trả về `Some(chuẩn)` khi đạt. `grad_remove` chỉ lấy được gradient **một lần**,
/// nên hàm này vừa kiểm vừa trả về giá trị cho bước kiểm tra chéo.
///
/// ⚠️ Không dùng `max().max(NaN)` để tổng hợp — `f32::max(NaN)` trả về số cũ,
/// khiến test báo ✅ giả (bẫy §8.2). Ở đây kiểm `is_finite` **trước**, rồi mới tính
/// chuẩn bằng `sum` + `sqrt`.
fn check<const D: usize>(
    param: &Param<Tensor<D>>,
    name: &str,
    grads: &mut Gradients,
) -> Option<f64> {
    let Some(g) = param.grad_remove(grads) else {
        println!("{name:<34} {:>13} {:>13}  FAIL (khong co gradient)", "-", "-");
        return None;
    };
    let vals = g.into_data().as_slice::<f32>().expect("f32").to_vec();

    if !vals.iter().all(|v| v.is_finite()) {
        println!("{name:<34} {:>13} {:>13}  FAIL (NaN/Inf — bay 8.2)", "-", "-");
        return None;
    }
    let norm = vals.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let max = vals.iter().fold(0.0f64, |a, v| a.max((*v as f64).abs()));
    if norm == 0.0 {
        println!("{name:<34} {norm:>13.3e} {max:>13.3e}  FAIL (bang 0)");
        return None;
    }
    println!("{name:<34} {norm:>13.3e} {max:>13.3e}  PASS");
    Some(norm)
}