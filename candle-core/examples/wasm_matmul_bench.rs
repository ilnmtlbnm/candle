use candle_core::quantized::{k_quants, GgmlType};
use candle_core::{Device, Tensor};
use std::time::Instant;

fn bench_f32_matmul(m: usize, k: usize, n: usize, label: &str, iters: u32) {
    let device = Device::Cpu;
    let lhs = Tensor::randn(0f32, 1.0, (m, k), &device).unwrap();
    // NT layout: rhs created as (n, k) then transposed → strides (1, k)
    let rhs = Tensor::randn(0f32, 1.0, (n, k), &device)
        .unwrap()
        .t()
        .unwrap();

    for _ in 0..3 {
        let _ = lhs.matmul(&rhs).unwrap();
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = lhs.matmul(&rhs).unwrap();
    }
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_micros() / iters as u128;

    println!(
        "{label:>20} ({m:>2}x{k:>4}) x ({k:>4}x{n:>4}): {per_iter_us:>6} us/iter  ({iters} iters)",
    );
}

fn bench_q8_0_matmul(m: usize, k: usize, n: usize, label: &str, iters: u32) {
    let k = (k / 32) * 32;
    let k_in_blocks = k / 32;

    let lhs: Vec<f32> = (0..m * k)
        .map(|i| 0.1 + 2.0 * (i as f32 * 0.01).cos())
        .collect();
    let rhs_f32: Vec<f32> = (0..n * k)
        .map(|i| 0.1 + 2.0 * (i as f32 * 0.007).sin())
        .collect();

    let mut rhs_q8: Vec<k_quants::BlockQ8_0> = vec![k_quants::BlockQ8_0::zeros(); n * k_in_blocks];
    for col in 0..n {
        k_quants::BlockQ8_0::from_float(
            &rhs_f32[col * k..(col + 1) * k],
            &mut rhs_q8[col * k_in_blocks..(col + 1) * k_in_blocks],
        );
    }

    let mut dst = vec![0f32; m * n];

    for _ in 0..3 {
        k_quants::matmul((m, k, n), &lhs, &rhs_q8, &mut dst).unwrap();
    }

    let start = Instant::now();
    for _ in 0..iters {
        k_quants::matmul((m, k, n), &lhs, &rhs_q8, &mut dst).unwrap();
    }
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_micros() / iters as u128;

    println!(
        "{label:>20} ({m:>2}x{k:>4}) x ({k:>4}x{n:>4}): {per_iter_us:>6} us/iter  ({iters} iters)",
    );
}

fn main() {
    // =====================================================================
    // Real Mimi codec dimensions (from mimi-rs MimiConfig::v202601):
    //   n_filters=64, ratios=[4,5,6] (reversed for encoder: [6,5,4])
    //   compress=2, kernel_size=7, residual_kernel_size=3
    //   dimension=512, transformer_d_model=512, transformer_dim_feedforward=2048
    //
    // SEANet encoder conv1d layers (im2col → f32 matmul):
    //   init:  in=1,   out=64,  ks=7, stride=1    → K=7,   N=64
    //   res0:  in=64,  out=32,  ks=3, dil=1       → K=192, N=32
    //   res0:  in=32,  out=64,  ks=1              → K=32,  N=64
    //   down0: in=64,  out=128, ks=12, stride=6   → K=768, N=128
    //   res1:  in=128, out=64,  ks=3, dil=1       → K=384, N=64
    //   res1:  in=64,  out=128, ks=1              → K=64,  N=128
    //   down1: in=128, out=256, ks=10, stride=5   → K=1280,N=256
    //   res2:  in=256, out=128, ks=3, dil=1       → K=768, N=128
    //   res2:  in=128, out=256, ks=1              → K=128, N=256
    //   down2: in=256, out=512, ks=8,  stride=4   → K=2048,N=512
    //   final: in=512, out=512, ks=3              → K=1536,N=512
    //
    // Mimi Q8_0 transformer (d_model=512, ffn=2048, 2 layers):
    //   in_proj:  (M, 512)  x (512, 1536)   [qkv fused]
    //   out_proj: (M, 512)  x (512, 512)
    //   ffn_up:   (M, 512)  x (512, 2048)
    //   ffn_down: (M, 2048) x (2048, 512)
    //
    // Pocket TTS FlowLM (d_model=1024, ffn=4096, 6 layers):
    //   in_proj:  (M, 1024) x (1024, 3072) [qkv fused]
    //   out_proj: (M, 1024) x (1024, 1024)
    //   ffn_up:   (M, 1024) x (1024, 4096)
    //   ffn_down: (M, 4096) x (4096, 1024)
    // =====================================================================

    println!("=== f32 matmul: Mimi SEANet encoder (real dimensions) ===");
    println!("M=1 is streaming mode (one frame at a time)");
    println!();

    // The hot conv1d layers (larger K, more compute)
    let f32_configs: &[(usize, usize, usize, &str, u32)] = &[
        // Streaming mode (M=1)
        (1, 768, 128, "down0", 100),
        (1, 1280, 256, "down1", 50),
        (1, 2048, 512, "down2", 50),
        (1, 1536, 512, "final", 50),
        // Residual blocks
        (1, 192, 32, "res0-3x3", 100),
        (1, 384, 64, "res1-3x3", 100),
        (1, 768, 128, "res2-3x3", 100),
    ];

    for &(m, k, n, label, iters) in f32_configs {
        bench_f32_matmul(m, k, n, label, iters);
    }

    println!();
    println!("=== Q8_0 matmul: Mimi transformer (real dimensions) ===");
    println!("d_model=512, ffn=2048, 2 layers");
    println!();

    let mimi_q8_configs: &[(usize, usize, usize, &str, u32)] = &[
        (1, 512, 1536, "mimi-qkv", 50),
        (1, 512, 512, "mimi-out", 50),
        (1, 512, 2048, "mimi-ffn-up", 50),
        (1, 2048, 512, "mimi-ffn-down", 50),
    ];

    for &(m, k, n, label, iters) in mimi_q8_configs {
        bench_q8_0_matmul(m, k, n, label, iters);
    }

    println!();
    println!("=== Q8_0 matmul: Pocket TTS FlowLM (real dimensions) ===");
    println!("d_model=1024, ffn=4096, 6 layers");
    println!();

    let pocket_q8_configs: &[(usize, usize, usize, &str, u32)] = &[
        (1, 1024, 3072, "pocket-qkv", 50),
        (1, 1024, 1024, "pocket-out", 50),
        (1, 1024, 4096, "pocket-ffn-up", 50),
        (1, 4096, 1024, "pocket-ffn-down", 50),
    ];

    for &(m, k, n, label, iters) in pocket_q8_configs {
        bench_q8_0_matmul(m, k, n, label, iters);
    }
}
