//! GĐ3 — chứng minh **chỉ LoRA được huấn**, base VITS đóng băng.
//!
//! Ba điều kiện phải đạt (xem `docs/architecture.md` §13.8):
//!   1. `freeze()` đóng băng base
//!   2. `apply_lora` với `ParamGroup` **thật sự** khớp — kiểm bằng cách đếm số
//!      tensor trong record, không tin regex
//!   3. Số tham số LoRA nhỏ hơn base rất nhiều
//!
//! Run: cargo run --release -p voice-models --example verify_lora

use std::path::PathBuf;

use burn::module::{Lora, Module, ParamGroup};
use burn::tensor::Device;
use voice_models::dec::Generator;

const DIR: &str = "sources/vits_ljs_dec";

/// Base được chọn cho LoRA: 4 `ups` + 12 `resblocks` × 6 conv.
///
/// Khớp **76** tensor weight — con số này kiểm được độc lập từ cấu trúc
/// (`upsample_rates` có 4 phần tử, `resblock_kernel_sizes` có 3, mỗi resblock có
/// `convs1` và `convs2` mỗi 3 conv theo dilation).
const BASE_RE: &str = r"^(ups\.\d+|resblocks\.\d+\.convs[12]\.\d+)\.weight$";
const N_EXPECTED_MATCH: usize = 4 + 12 * 6; // = 76

/// Chỉ hai tensor LoRA mỗi lớp được huấn.
const LORA_RE: &str = r"(^|\.)lora\.(a|b)$";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(DIR);
    let device = Device::default().autodiff();

    let gen = Generator::load(&dir, &device)?;
    let base_params = gen.num_params();
    let rank = 8usize;
    let alpha = 16.0f64;
    println!("base           num_params = {base_params}  ({:.2} M)", base_params as f64 / 1e6);
    assert_eq!(base_params, 14_327_424, "số tham số base lệch tài liệu §2.1");

    // ---- 1. đóng băng base TRƯỚC khi gắn LoRA
    //
    // ⚠️ `freeze()` khác `no_grad()`: tắt cả grad tracking lẫn training flag.
    // ⚠️ `valid()`/`train()` KHÔNG hoàn tác `freeze()` đã đặt.
    let frozen = gen.freeze();

    // ---- 2. kiểm regex base khớp đúng số tensor
    //
    // Không tin regex: `into_record_group` trả về record chỉ chứa những tensor
    // thực sự khớp, nên `.len()` là số đếm thật.
    let base_group = ParamGroup::from_regex(BASE_RE).map_err(|e| format!("regex base khong hop le: {e:?}"))?;
    let n_match = frozen.clone().into_record_group(base_group.clone()).len();
    println!("regex base     khớp {n_match} tensor (mong doi {N_EXPECTED_MATCH})");
    if n_match != N_EXPECTED_MATCH {
        return Err(format!(
            "regex `{BASE_RE}` khớp {n_match} tensor, mong doi {N_EXPECTED_MATCH} — \
             sai đường dẫn hoặc sai regex"
        )
        .into());
    }

    // ---- 3. gắn LoRA
    //
    // Đo trước số tham số LoRA **tường minh** từ shape của 76 weight được khớp:
    // mỗi weight [d1, d2, k] nhận `rank·(d1 + d2)` tham số.
    let expected_lora_params: usize = collect_matched_shapes(&frozen)
        .iter()
        .map(|[d1, d2]| rank * (d1 + d2))
        .sum();
    println!(
        "LoRA rank={rank}  tham so du kien = {expected_lora_params}  \
         ({:.3}% cua base)",
        100.0 * expected_lora_params as f64 / base_params as f64
    );
    assert!(
        expected_lora_params * 10 < base_params,
        "LoRA them qua nhieu tham so: {base_params} → +{expected_lora_params}"
    );

    let adapted = frozen.apply_lora(Lora::new(rank, alpha).set_param_group(base_group));

    // ⚠️ `num_params()` **không đổi** sau `apply_lora`: tham số LoRA nằm trong hộp
    // reparameterization chứ không phải param thường của module. Đo bằng
    // `num_params()` để ước lượng sẽ luôn ra 0 — đừng tin phép đo đó.
    println!(
        "sau LoRA       num_params = {}  (base + 0: LoRA nam trong reparameterization)",
        adapted.num_params()
    );

    // ---- 5. đếm số tensor LoRA
    //
    // ⚠️ Kết quả: **0**. Và đây **không phải** regex sai.
    //
    // `burn-core/src/module/lora.rs:90-94`:
    // ```rust
    // // LoRA only adapts 2-D weight matrices. Every other base parameter is frozen too...
    // if D != 2 { return (param.set_require_grad(false), None); }
    // ```
    //
    // Generator VITS **không có tham số 2-D nào**: weight đều 3-D, bias 1-D. Nên
    // `apply_lora` không gắn được adapter nào — nó chỉ freeze base rồi trả về y nguyên.
    // `num_params()` không đổi ở trên chính vì thế.
    let lora_group = ParamGroup::from_regex(LORA_RE).map_err(|e| format!("regex lora khong hop le: {:?}", e))?;
    let n_lora = adapted.clone().into_record_group(lora_group).len();
    println!("tensor LoRA    {n_lora}   <-- 0: xet ghi chu, burn chi LoRA cho tham so 2-D");

    // ---- 6 & 7: chẩn đoán
    //
    // `freeze()` **có** hiệu lực: chạy backward thật để chứng minh, thay vì đếm cờ.
    let z = burn::tensor::TensorData::new(vec![0.01f32; 192], [1, 192, 1]);
    let z = burn::tensor::Tensor::<3>::from_data(z, &device).require_grad();
    let loss = adapted.forward(z).powf_scalar(2.0).sum();
    let mut grads = loss.backward();
    let g_base = adapted.conv_pre.weight.grad_remove(&mut grads);

    println!("\nbackward tren model da freeze + apply_lora:");
    println!(
        "  conv_pre.weight (base) -> {}",
        if g_base.is_none() {
            "khong gradient  ✓ base DONG BANG that"
        } else {
            "CO GRADIENT     ✗ base chua bi dong bang"
        }
    );

    let flat_params = adapted.clone().materialize().num_params();
    println!(
        "  materialize()   num_params = {flat_params} (base {base_params}) — {}",
        if flat_params > base_params { "da gap" } else { "khong doi: khong co adapter nao de gap" }
    );

    // ---- kết luận
    let base_frozen = g_base.is_none();
    println!("\n{}", if base_frozen { "✓ freeze() hoat dong" } else { "✗ freeze() that bai" });

    if n_lora == 0 {
        println!(
            "\n=== GĐ3 BLOCKED ===\n\
             apply_lora khong gac duoc adapter nao: burn chi LoRA cho tham so 2-D\n\
             (lora.rs:92), con generator VITS toan conv 3-D + bias 1-D.\n\n\
             Regex base khop DUNG {n_match} tensor, va freeze() DUNG co hieu luc —\n\
             phan con thieu la co adapter de gan hay khong.\n\n\
             Huong giai: viet Reparameterizer cho conv (reshape [out,in,k] -> [out,in*k]\n\
             roi matmul), roi goi apply_reparameterization thay vi apply_lora."
        );
        std::process::exit(1);
    }

    println!("\nPASS tat ca");
    Ok(())
}

/// Thu thập `[d1, d2]` (hai chiều đầu) của mọi weight mà `BASE_RE` khớp.
///
/// Duyệt thẳng cấu trúc đã biết (`ups` + `resblocks`) thay vì parse regex — đây là
/// danh sách **độc lập** với regex, nên khi hai bên lệch nhau thì biết ngay là
/// regex sai chứ không phải logic đếm sai.
fn collect_matched_shapes(gen: &Generator) -> Vec<[usize; 2]> {
    let mut out: Vec<[usize; 2]> = Vec::new();

    for up in &gen.ups {
        let [i, o, _] = up.weight.val().dims();
        out.push([i, o]);
    }
    for rb in &gen.resblocks {
        for c in rb.convs1.iter().chain(rb.convs2.iter()) {
            let [o, i, _] = c.weight.val().dims();
            out.push([o, i]);
        }
    }
    assert_eq!(out.len(), N_EXPECTED_MATCH, "so weight khong khop N_EXPECTED_MATCH");
    out
}