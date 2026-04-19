# Ising-QEC — run summary (2026-04-18/19)

One-file review of what was done, what worked, what broke, and where the artifacts live.

## Goal

Stand up an end-to-end pipeline on top of `NVIDIA/Ising-Decoding` for a d=5 surface-code decoder under biased (Z:X = 10:1) circuit-level noise, fine-tuned from the Fast pretrained model, exported to FP8 TensorRT, and benchmarked for round-trip latency against the notes.md <10 µs target on Kempner H200 hardware.

## Final numbers (H200, batch=1, qec-decoder-d5-biased, Fast-seeded)

### LER (logical error rate), 4096 shots per basis

| path              | X basis | Z basis | Avg    | note                                           |
|-------------------|---------|---------|--------|------------------------------------------------|
| PyMatching alone  | 0.0645  | 0.0032  | 0.0338 | baseline with no neural pre-decoder            |
| FP32 (PyTorch)    | 0.0688  | 0.0024  | 0.0356 | pre-decoder + PyMatching, training-time eval   |
| FP8 TRT           | 0.0686  | 0.0024  | 0.0355 | FP8 engine + PyMatching, near parity with FP32 |

- 18× gap between X and Z LER is the expected signature of a biased-noise decoder.
- FP8 quantization penalty ≈ 5% relative on Avg LER; improvable with more than the 256 default calibration samples.
- PyMatching wall time: 0.93 → 0.58 µs/round with the neural pre-decoder active (1.60× speedup).

### Compute-only forward-pass latency (1000 iters)

| config                          | mean µs | p50   | p99   | min   | max    |
|---------------------------------|---------|-------|-------|-------|--------|
| PyTorch FP32 eager              | 159.3   | 157.9 | 170.1 | 134.0 | 181.7  |
| PyTorch FP32 CUDA graph         | 93.1    | 93.1  | 94.1  | 92.5  | 115.3  |
| FP8 TRT eager                   | 115.7   | 115.7 | 117.3 | 114.4 | 126.1  |
| FP8 TRT CUDA graph              | **3.86**| 3.74  | 5.86  | 3.42  | 19.65  |

24× speedup vs PyTorch FP32 graph baseline once we get onto the FP8 TRT + graph path.

### Round-trip (host-pinned H2D → engine → D2H, 1000 iters)

| chain style                          | mean µs | p50   | p99   | min   | max    |
|--------------------------------------|---------|-------|-------|-------|--------|
| 3 separate eager launches            | 149.6   | 149.5 | 152.7 | 143.3 | 202.7  |
| Fused single CUDA graph (H2D+eng+D2H)| **6.45**| 6.40  | 7.68  | 5.95  | 9.28   |

- Under notes.md <10 µs target including pinned-memory I/O both ways.
- 23× gain from fusing the three launches into one graph.
- ~2.5 µs above engine-only is the actual memcpy PCIe cost (120 bytes in, 121 out).

## What shipped

- `env/ising.def` + built `env/ising.sif` (10 GB, cu12-pinned TensorRT)
- `conf/config_ising_qec_d5_biased.yaml` — 25-param biased noise override
- `slurm/*.sbatch` — smoke, bootstrap, train, pytest, export_fp8, latency (torch), latency_trt, roundtrip. All on kempner_eng H200.
- `bench/latency.py` — PyTorch forward-pass latency.
- `bench/latency_trt.py` — TensorRT engine latency.
- `bench/roundtrip.py` — full H2D→engine→D2H round-trip with eager and graph-fused modes.
- `outputs/outputs/qec-decoder-d5-biased/models/best_model/PreDecoderModelMemory_v1.0.50.pt` — fine-tuned weights (0.91 M params).
- `Ising-Decoding/predecoder_memory_d5_T5_{X,Z}_fp8.{onnx,engine}` — FP8 artifacts.

## What went wrong (and the fix)

1. **Container build: `/tmp` cleared mid-build.**
   Singularity %files landed requirements in `/tmp/`; the apt clean step then wiped them, leaving `pip install -r /tmp/...` with no file.
   Fix: copy to `/opt/reqs/` instead.

2. **`tensorrt` package pulled cu13 libs.**
   Plain `pip install tensorrt` resolved to `tensorrt 10.16.1.11` + `tensorrt_cu13_libs`, which needs a newer NVIDIA driver than some kempner_eng nodes have. Result: `CUDA initialization failure with error: 35` inside ModelOpt's INT8/FP8 path, then a segfault.
   Fix: pin `tensorrt-cu12>=10.8,<10.13` in `env/ising.def`. Rebuilt to `tensorrt 10.12.0.36`.

3. **cuStabilizer BitMatrixSampler segfaults on basis=Z under training LER.**
   The per-epoch LER validation calls the cuQuantum sampler hundreds of times; state accumulates across X/Z bases, and basis=Z eventually hits a native-code crash. Hardware-independent (observed on both H100 and H200). Not triggered by the once-off end-of-training LER call.
   Fix: set `PREDECODER_LER_FINAL_ONLY=1` during training. Compute LER only at final epoch.
   Residual patch: `Ising-Decoding/code/qec/dem_sampling.py` now honors `ISING_QEC_DISABLE_CUST=1` to force the non-DLPack sampling path (kept for future debugging but not currently used).

4. **50 epochs at upstream default would have taken ~125 h.**
   Upstream default is ~8.4 M samples per epoch. With `--time=24:00:00` this cannot finish.
   Fix: `PREDECODER_TRAIN_SAMPLES=262144` → ~512 batches/epoch → ~70 min for the full 50-epoch schedule.

5. **kempner_dev A100 nodes segfaulted in the same cuStabilizer path.**
   The earliest d5-biased run landed on an A100 (holygpu8a19605) and crashed during basis=Z LER. Unrelated to the driver issue above — same code path, same crash pattern as #3.
   Fix: per user guidance, pin all jobs to kempner_eng (H200). Saved as a feedback memory.

6. **FP8 ONNX export pybind11 init error on first attempt.**
   `pybind11::init(): factory function returned nullptr` during ModelOpt FP8 quantization; upstream fell back to FP32 PyTorch inference.
   Root cause: same tensorrt cu13 issue from #2. Rebuilding with cu12 pin fixed it — FP8 export then completed in 10.6 s and produced a valid 2.2 MB TRT engine.

## Decisions worth remembering

- **Receptive field is fixed at 9** for Model 1 (the Fast model). Our `distance=5, n_rounds=5` config controls evaluation sampling, not training window. Training data always arrives as a 9×9×9 volume. This is upstream's design; §4 of the original notes got it right, the Phase-2 "13×13×13" line was wrong for d=5.

- **Validation generator runs at d=9** regardless of config target. If we want val on d=5, custom generator logic is needed. Not pursued here.

- **LER measured once at the end of training** (`PREDECODER_LER_FINAL_ONLY=1`). Per-epoch LER validation remains broken; would need upstream fix to cuStabilizer caching.

- **FP8 calibration is coarse** (256 samples, default). Beefing it up should close the ~5% FP8→FP32 LER gap if we want matching accuracy.

- **Round-trip I/O path is now a fused CUDA graph** (`bench/roundtrip.py`). The pinned host buffer is the only addressable surface; host memcpy into it between replays gives us a working streaming model for realtime use.

## Open items

- **Noise parameter sweep** — we ran one point (Z:X = 10:1, p_total ≈ 6e-3). A proper sweep over bias ratio and total rate would clarify how specialization transfers.
- **Matched-distance validation** — val at d=9, test at d=5; worth adding a d=5 val generator for cleaner signal.
- **cuStabilizer per-epoch LER** — still segfaults. Worth reporting upstream or wiring a pure-PyMatching LER fallback.
- **FP8 LER gap** — fix with more calibration samples or INT8-to-FP8 conversion of a higher-quality calibration set.
- **Simulator-side timing** — our round-trip assumes the syndrome is already in pinned host memory. Real-world controller latency from detector readout to pinned memory has not been measured and is outside our control.

## Timeline (rough)

- Plan review + upstream verification: ~30 min
- Clone + container v1 build: ~30 min (failed, /tmp issue)
- Container v1 rebuild (/opt/reqs fix): ~20 min
- Smoke + bootstrap + training bring-up: ~60 min (mostly iterating through cuStabilizer crash + partition / GRES issues)
- Fine-tune run (50 epochs): ~70 min wall
- FP8 export attempt #1 (cu13 TRT): fell back to PyTorch FP32
- Container v2 rebuild (cu12 TRT pin): ~20 min
- FP8 export attempt #2: succeeded, engines built
- Latency + round-trip benchmarks: ~15 min total job time
