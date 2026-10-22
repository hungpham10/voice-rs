use std::path::PathBuf;
use voice_train::dataset::{Chunk, Dataset, Utterance};

fn make_ds(chunk_frames: usize) -> Dataset {
    let mut utts = vec![Utterance {
        speaker: 0,
        utt_id: "short".into(),
        t_h: 1,
        c: 768,
        feats: vec![0.0f32; 1 * 768],
        wav: vec![0.1f32; 1 * 256],
        f0: None,
    }];
    for i in 0..5usize {
        utts.push(Utterance {
            speaker: 0,
            utt_id: format!("long{i}"),
            t_h: 40,
            c: 768,
            feats: vec![0.1f32; 40 * 768],
            wav: vec![0.1f32; 69 * 256],
            f0: Some(vec![100.0f32; 50]),
        });
    }
    let mut chunks = Vec::new();
    let mut per_speaker_chunks = vec![Vec::new()];
    for (u_idx, utt) in utts.iter().enumerate() {
        let total = Dataset::latent_len_of_utt(utt);
        if total >= chunk_frames {
            let cidx = chunks.len();
            chunks.push(Chunk {
                utt: u_idx,
                offset_lat: 0,
                len_lat: chunk_frames,
                chunk_id: format!("{}#0", utt.utt_id),
            });
            per_speaker_chunks[0].push(cidx);
        }
    }
    Dataset {
        speakers: vec!["spk0".to_string()],
        utts,
        per_speaker: vec![vec![0, 1, 2, 3, 4, 5]],
        chunks,
        per_speaker_chunks,
    }
}

fn main() {
    let device = burn::tensor::Device::flex();
    let train_ds = make_ds(32);
    let val_ds = make_ds(32);
    let cfg = voice_train::train::TrainConfig {
        dec_dir: PathBuf::from("sources/vits_ljs_dec"),
        out_dir: std::env::temp_dir().join("pitch_smoke_out"),
        chunk_frames: 32,
        batch: 2,
        grad_accum: 2,
        steps: 2,
        lr: 1e-4,
        val_every: 50,
        seed: 42,
        warmup: 1,
        clip_norm: 1.0,
        checkpoint_every: 0,
        nsf: false,
        pitch_embed: true,
    };
    match voice_train::train::run_training(&train_ds, &val_ds, &cfg, &device) {
        Ok(s) => println!("OK first={:.4} last={:.4} best={:.4}", s.first, s.last, s.best_val),
        Err(e) => println!("ERR: {e}"),
    }
}
