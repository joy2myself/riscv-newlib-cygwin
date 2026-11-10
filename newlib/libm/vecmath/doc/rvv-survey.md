# RISC-V Vector (RVV 1.0) math-library and performance-engineering survey

**Audience:** `libvecmath` maintainers (newlib, array API, 41 f64 + 41 f32 entries, static LMUL, VLA strip loop, ≤ 1 ulp).
**Date of survey:** 2026-11-11. **Scope:** public RVV 1.0 math libraries, measured RVV instruction costs, VLA programming guidance, toolchain pitfalls, precision practice.

**Evidence tags used throughout**

| tag | meaning |
|---|---|
| **[M]** | Number reported/measured in the cited external source, with that source's hardware. |
| **[M-ours]** | Measured by our project (repo README, `test/results/`, or the task brief). |
| **[I]** | My inference or engineering judgement — not directly measured anywhere cited. |
| **[C]** | Vendor/community claim, accepted but not independently verified here. |

Every performance number below states the **core, VLEN, and LMUL**. Where a source measures *throughput* (independent instructions), that is stated; it is **not** latency.

---

## 0. Bottom line for our two boards

1. **Do not raise LMUL to fix VLEN=128.** On the a210 core (XuanTie C920v2, VLEN=128), measured RVV instruction throughput scales **linearly** with LMUL for divide/sqrt/FMA/add/mul: `vfdiv.vv e64m1 = 20 cyc`, `m2 = 40`, `m4 = 80`, `m8 = 160` — i.e. the **per-element cost is constant** across LMUL. LMUL buys loop-overhead amortization and slightly better load efficiency, not arithmetic throughput. It cannot repair a latency-bound kernel. Raising LMUL only adds register pressure — consistent with our measured regression at LMUL 4. **[M]** (rvv-bench, C920v2) + **[M-ours]**.
2. **The a210 outliers are mostly codegen, not arithmetic.** `atan2` at 316 ns/element is ≈600 cycles/element at 1.9 GHz; a 2-division + polynomial kernel should be ~50–100 cycles/element even at VLEN=128. GCC Bugzilla PR 121018 documents GCC spilling `v0–v8, v24–v31` around a call between RVV intrinsics, and PR 121051 documents generally bad RVV register allocation ("half of the libjpeg-turbo RVV kernels spill in GCC, while they have very neat codegen in clang"). Audit the asm of the 13 sub-1.0x entries before touching their math. **[M]** (GCC PRs) + **[I]**.
3. **Keep static LMUL 2 as the default, but specialize per function at compile time:** LMUL 1 for the 13 a210 sub-1.0x entries (listed in R1) and LMUL 2 for the rest; LMUL 4/8 never. This is the same shape of policy as the only public RVV libm with per-function LMUL data (`rivosinc/veclibm`: 58 functions at m2, 20 at m1, 4 at m4). **[M]**.
4. **Remove `vfdiv.vv` from critical paths.** On C920v2/e64: `vfdiv.vv` 20 cyc (10 cyc/element), `vfdiv.vf` 6.37 cyc (3.2/element), `vfrec7.v` 0.5–1 cyc (0.25–0.5/element). One `vfrec7` plus 2–3 Newton FMAs is cheaper than one vector/vector divide and is the single largest lever for `erfc`/`atan2`/`atanpi` at VLEN=128. **[M]** (instruction costs) + **[M-ours]** (our erfc 181→120 ns/element from replacing three divisions with one reciprocal).
5. **Get the `vfirst` + out-of-line fixup call out of the strip loop.** Accumulate a "special lanes" mask with `vmor` across strips and run one scalar fixup pass after the loop. This removes the vector→scalar op, the branch, and (critically) the function call that triggers the GCC spill in PR 121018. Our measured 51% regression from a `vfirst` guard in a long chain is the same phenomenon. **[M-ours]** + **[M]** (PR 121018) + **[I]** (design).
6. **For latency-bound kernels, interleave two independent strips rather than widening LMUL.** The cores are in-order (X60 is explicitly "in-order" in the literature); with LMUL linear scaling, two m1 strips give two independent chains at the register cost of one m2 group, and hide divider latency. **[I]**, supported by **[M]** (linear LMUL scaling, in-order cores).
7. **Use `vrgather` only at LMUL 1/2, never 4/8.** X60 (VLEN=256): `vrgather.vv e64m1 = 4 cyc` (1 cyc/element), `m2 = 16` (2/element), `m4 = 64` (4/element), `m8 = 256` (8/element) — quadratic. C920v2 (VLEN=128): 0.5 / 2.36 / 8 / 32 cyc. `vrgather.vx/vi` is much cheaper than `.vv` on X60 (2 vs 4 cyc at m1). **[M]**.
8. **Keep `-mrvv-vector-bits=scalable`** (the default). GCC's own documentation says `zvl` code "is NOT portable between the core with different VLEN"; it is the one flag guaranteed to break the single-binary requirement. **[M]** (GCC 14 changes).
9. **Build the remaining f32 widen/narrow wrappers natively where the a210 scalar baseline is strong** — `cbrtf` (0.73x), `erfcf` (0.76x), `atan2f` (0.62x), `cospif` (0.82x). A native f32 `cbrt` with LMUL 1 and a short polynomial should beat the f64 widen/narrow path at VLEN=128. **[M-ours]** + **[I]**.
10. **Do not wait for SLEEF or glibc.** SLEEF's RVV backends are now marked **"Unmaintained"** in SLEEF master (4.0.0-dev), after being added experimentally in 3.6 (LMUL 1/2 only). glibc has **no** RISC-V libmvec (RFC only, May 2024; `sysdeps/riscv/rvv/` contains only `memset.S`). LLVM libc's new `mathvec` has no RVV backend as of its March 2026 RFC. `rivosinc/veclibm` is the best algorithm/LMUL reference, but it is f64-only, MIT, and has no published ulp contract. **[M]**.

---

## 1. Hardware and measurement context

| board / core | ISA | VLEN | DLEN | clock | notes | sources |
|---|---|---|---|---|---|---|
| **muse-pi**: SpacemiT X60 (K1) | RVV 1.0 + Zvfh | 256 | 256 | 1.6 GHz (BPI-F3) / 1.8 GHz (Jupiter) | 8 cores, in-order, 32 KB L1D, 2×512 KB L2; full-VLEN FMA per cycle | **[M]** rvv-bench X60 page; **[M]** arXiv 2605.10860, 2608.28097 |
| **a210**: XuanTie C920v2 (Zhihe A210) | RVV 1.0 + Zvfh | 128 | 256 (2×128-bit pipes; rvv-bench reports VLEN=128) | 1.9 GHz | the A210 board has 4× C920v2 + 4× C908 (assumed to be our a210); full-VLEN FMA per cycle | **[M]** rvv-bench C920v2 page; **[M]** arXiv 2608.28097; **[I]** board identity |
| XuanTie C908 (K230) | RVV 1.0 | 128 | 128 | 1.6 GHz | useful proxy for "small RVV" | **[M]** rvv-bench C908 page |
| SiFive P870 (LX500) | RVV 1.0 | 128 | — | 2.2–2.7 GHz | 6-wide OoO; poor `vfdiv.vv` throughput | **[M]** rvv-bench P870 page |
| SiFive X280 (Blackhole) | RVV 1.0 | 512 | 256 | — | full VLEN takes 2 cycles | **[M]** arXiv 2608.28097; rvv-bench |

**Important caveat on the rvv-bench tables used below.** They are "measured cycle averages when unrolling and looping over the given instruction, which should estimate the **instruction throughput**"; operands are independent (dependency broken). They are **not latency** and they use `vl = VLMAX`, `ta/ma`, on the stated core. A dependent chain will be slower. The X60 page's floating-point ALU/divide/sqrt cells are **blank** in the published table (only integer/permute/load rows are populated); I therefore use C920v2/C908/P870 for FP instruction costs and the X60 `veclibm` benchmark for end-to-end function throughput. Do not silently extrapolate C920v2 divide costs to X60. **[M]** (page structure) + **[I]** (caution).

---

## 2. Catalogue of RVV math implementations

### 2.1 SLEEF — has an RVV backend, but it is experimental and now unmaintained

| item | finding |
|---|---|
| Version | RISC-V libm support added in **SLEEF 3.6** (released 2024-02-15; PR #477 "Integrate RISC-V support", merged 2023-11-20). Current master is 4.0.0-dev. **[M]** |
| Backends | `RVVM1` (LMUL 1) and `RVVM2` (LMUL 2) only. CMake options `SLEEF_ENABLE_RVVM1`/`RVVM2` are marked "experimental feature … may be removed without notice"; `COSTOVERRIDE_RVVM1/2 = 3`. **[M]** |
| Status now | SLEEF master `README.adoc` supported-architecture table lists `RISC-V · RVVM1 · **Unmaintained**` and `RVVM2 · **Unmaintained**`, with "Unmaintained features may be removed without notice." **[M]** |
| VLEN assumptions | `src/arch/helperrvv.h` has `CONFIG == 1/2` = "RISC-V Vector Extension with Min. VLEN" (runtime `vsetvlmax`, VLA), and `CONFIG == 7..11` = fixed **128/256/512/1024/2048-bit** builds (`SLEEF_RVV_VLEN`, `LOG2VECTLENDP`). So SLEEF can build either a VLA binary or a fixed-VLEN binary; the fixed builds are VLEN-specific. **[M]** |
| Algorithms | Same portable C sources as the other backends (`sleefsimddp.c`/`sleefsimdsp.c`), including the double-double helpers in `src/common/dd.h`; FMA path enabled for RVV (`ENABLE_FMA_DP/SP`) when `CONFIG != 2`. **[M]** (source) |
| Accuracy claim | SLEEF's `_u10` variants are its 1.0-ULP tier (the suffix is the ULP bound); `_u35` is the faster 3.5-ULP tier. No RVV-specific accuracy report is published; the same algorithms are compiled, so the bound is expected to carry over. **[M]** for naming/tier; **[I]** for RVV carry-over. |
| Measured RVV performance | **No official SLEEF RVV performance numbers found** (SLEEF's own benchmark pages are x86/AArch64). The rvv-bench `veclibm` benchmark is *not* SLEEF; it is `rivosinc/veclibm` (§2.2). Do not attribute its numbers to SLEEF. **[M]** absence of evidence. |

**Takeaway:** SLEEF is a useful algorithmic reference (and its dd helpers are a model for 1-ulp tails), but it is not a maintained RVV dependency, and it deliberately stops at LMUL 2.

### 2.2 `rivosinc/veclibm` — the most relevant public RVV libm

- **What:** MIT-licensed, C RVV-intrinsic vector math library, MKL-style API, **double precision only** at present; every function in its own file, with per-function LMUL configuration. README: "We sometimes provide multiple implementations for a single function as each may perform differently on different hardware implementations of the RISC-V vector ISA … one can manually define one … allowing one to determine the highest performing choice experimentally." **[M]** ([README](https://github.com/rivosinc/veclibm))
- **Static LMUL policy [M]:** from `include/rvvlm.h`, of 82 double-precision entry points, **58 use `rvvlm_fp64m2.h` (LMUL 2)**, **20 use `rvvlm_fp64m1.h` (LMUL 1)**, and **4 use `rvvlm_fp64m4.h` (LMUL 4)**. The LMUL-1 set is the long-chain/special set: `cdfnorm`, `cdfnorminv`, `erfc`, `erfinv`, `erfcinv`, `exp10`, `expint1`, `pow`, `lgamma`, `tgamma`; the LMUL-4 set is `cbrt` and `exp` (plus their strided variants).
- **Algorithms [M]:** FMA-heavy minimax polynomials; for `exp` a degree-11 polynomial (`PSTEP`, `rvvlm_expD.inc.h`) with four compile-time variants — `STD`, `STD_R_EXTRA`, `STD_EPSIM`, `TBL64` (64-entry table). Trig has `rvvlm_2ovpi_tbl.c`; `atan`/`atan2` use FMA corrections of the quotient (`tmp3 = vfmacc(divide, tmp3, PI_LO, tmp1)`), i.e. the same "one reciprocal + residual" idea we use.
- **Precision practice [M]:** no formal ULP contract; tests compare against **long double** (`test_funcs.c`), which the README calls "deemed sufficient". No double-double type is used in the kernels I inspected; accuracy comes from FMA and careful splitting (not from dd).
- **Measured throughput [M]** — rvv-bench `veclibm` page, **SpacemiT X60, VLEN=256, 1.6 GHz, fp64, elements/cycle, best of runs; speedup vs scalar libc**:

| function | scalar libc el/cyc | veclibm el/cyc | speedup | | function | scalar | veclibm | speedup |
|---|---|---|---|---|---|---|---|---|
| exp | 0.010041 | 0.036663 | 3.65x | | asin | 0.006330 | 0.015950 | 2.51x |
| exp2 | 0.010880 | 0.031866 | 2.92x | | acos | 0.007101 | 0.014722 | 2.07x |
| expm1 | 0.006542 | 0.025765 | 3.93x | | atan | 0.006554 | 0.012272 | 1.87x |
| log | 0.011769 | 0.025071 | 2.13x | | sinh | 0.003711 | 0.020924 | 5.63x |
| log10 | 0.006865 | 0.023795 | 3.46x | | cosh | 0.005534 | 0.024105 | 4.35x |
| log2 | 0.012316 | 0.025622 | 2.08x | | tanh | 0.004238 | 0.017362 | 4.09x |
| log1p | 0.005046 | 0.029617 | 5.86x | | asinh | 0.003840 | 0.013434 | 3.49x |
| sqrt | 0.026011 | 0.087398 | 3.36x | | acosh | 0.014037 | 0.015503 | 1.10x |
| cbrt | 0.003617 | 0.021652 | 5.98x | | atanh | 0.003320 | 0.021419 | 6.45x |
| sin | 0.007350 | 0.025246 | 3.43x | | erf | 0.008981 | 0.011269 | 1.25x |
| cos | 0.007662 | 0.024844 | 3.24x | | **erfc** | 0.009433 | 0.005403 | **0.57x** |
| tan | 0.005325 | 0.015874 | 2.98x | | tgamma | 0.002126 | 0.003782 | 1.77x |
| | | | | | **lgamma** | 0.004234 | 0.003266 | **0.77x** |

  Per-function LMUL for this table (from `rvvlm.h`): `exp`/`cbrt` = **m4**; `erfc`/`pow`/`lgamma`/`tgamma` = **m1**; the remaining functions = **m2**. The benchmark calls `rvvlm_<name>` with the library's default configuration. These are end-to-end array measurements (function-call per element, strip-mined) and are the closest public analogue to our harness. Note that **even a dedicated RVV libm loses to scalar on `erfc` and `lgamma` at VLEN=256 on X60** — the same special-function family that is weak for us on a210 (`erfc` 0.71x) and on muse-pi (`erfc` 1.24x; `lgamma` 1.39x). This is external validation that these functions are chain-bound, not that our implementation is uniquely bad. **[M]** + **[I]**.

### 2.3 glibc libmvec — no RISC-V port

- libmvec exists for **x86_64** (glibc 2.22) and **AArch64** (glibc 2.40); glibc `NEWS` mentions libmvec only for those. **[M]**
- **RISC-V:** an RFC series "[RFC V4] Enable libmvec support for RISC-V" by Yulong Shi (ISCAS/PLCT) was posted in April–May 2024, demonstrating vector `cos` and experimenting with EMUL=2; review suggested Rivos' `veclibm` as the algorithm source and questioned the ABI details. It was **not merged**. **[M]** ([RFC V4 thread](https://sourceware.org/pipermail/libc-alpha/2024-May/156644.html))
- glibc master `sysdeps/riscv/rvv/` contains **only `memset.S`**; there is no RISC-V `sysdeps/riscv/rvv/fpu` or libmvec directory. **[M]** (GitHub API listing, 2026-11-11)
- LLVM/Clang has a patch series "[TargetLibraryInfo] Add libmvec support for risc-v" that renames `LIBMVEC_X86` to `LIBMVEC` and adds `RVVM1/2/4/8` to `VFISAKind` so the vectorizer can emit `_ZGV…` calls with an LMUL suffix. This is compiler-side plumbing, not a library; status is patch-series, not a shipped RVV libmvec. **[M]** ([cfe-commits patch](https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg509884.html))

**Consequence for us:** `-fveclib=glibc`/OpenMP `declare simd` cannot be relied on for RISC-V in 2026; the array API + VFABI aliases we already export are the right interface. **[I]**

### 2.4 LLVM libc / `mathvec` — generic f32 only, no RVV backend yet

- LLVM libc's new `libc/src/mathvec` tree (RFC "[RFC][MATHVEC] Introduce the LLVM's mathvec library as yet another `-fveclib=` functions provider", **March 2026**) contains **generic vectorized f32** implementations of 31 functions (`acosf`, `acoshf`, `asinf`, `atanf`, `atanhf`, `cbrtf`, `cosf`, `coshf`, `erff`, `expf`, `exp2f`, `exp10f`, `expm1f`, `logf`, `log1pf`, `log2f`, `rsqrtf`, `sinf`, `sinhf`, `sqrtf`, `tanf`, `tanhf`, `tanpif`, …). **[M]**
- Optimized backends: **AArch64 NEON `expf`** only; exported symbols are x86_64 `_ZGVdN8v_*` and AArch64 `_ZGVnN4v_*`. **No RISC-V/RVV mention** in the RFC. **[M]**
- No ULP claims in the RFC; it is a `-fveclib` provider, not a 1-ulp contract library. **[M]** ([RFC thread](https://discourse.llvm.org/t/rfc-mathvec-introduce-the-llvms-mathvec-library-as-yet-another-veclib-functions-provider/92591))

### 2.5 OpenCV / DNN — RVV universal intrinsics, not a libm

- OpenCV has had length-agnostic RVV support through its **universal intrinsics** since 2022, and a **HAL `riscv-rvv`** backend introduced for RVV 1.0 (blog: "Introducing HAL riscv-rvv: Unleashing the power of RISC-V CPUs with RVV 1.0", by Yuantao Feng, OpenCV China). SpacemiT K1 (VLEN=256) is the reference board. **[C]** ([OpenCV blog](https://opencv.org/introducing-hal-riscv-rvv-unleashing-the-power-of-risc-v-cpus-with-rvv-1-0/), [wiki](https://github.com/opencv/opencv/wiki/OpenCV-RISC-V))
- Scope is image/arithmetic/DNN operators, not scalar-C99-compatible transcendentals; no per-function ULP contract; fast-math-style approximations. Numbers are reported as operator/frame speedups, not ns/element with VLEN/LMUL, so they are **not comparable** to our harness. **[C]** + **[I]**
- Relevant engineering practice: OpenCV's RVV backend is a *single VLA binary* (no per-VLEN dispatch) built on universal intrinsics — the same portability model we use. **[C]**

### 2.6 Vendor libraries

| vendor | public RVV math library? | what exists | accuracy / measured numbers |
|---|---|---|---|
| **XuanTie / T-Head** | **No public `xt-math`** found (GitHub org `XUANTIE-RV`, 73 repos, none named math/xt-math). | `csi-nn2` (NN operator library), optimized OpenBLAS for C910/C920/C930 (vendor page), glibc/newlib forks, and public **RVV intrinsic manuals** (RVM-0.3, RVV-1.0). | No public per-element transcendental accuracy or VLEN/LMUL throughput. OpenBLAS numbers are GEMM-level. **[M]** (absence) + **[C]** (vendor page) |
| **Andes** | `andestech/libvec` — "Andes Vector Library" | vector/matrix basic ops, filtering, image resize, transforms; **pure-C** build (`build_purec_lib.sh`), not a transcendental libm. | No ULP claims; no RVV instruction-level numbers. **[M]** ([libvec](https://github.com/andestech/libvec)) |
| **SiFive** | **No public RVV transcendental library** found (`org:sifive math` GitHub search = 0). | `freedom-metal` is a bare-metal HAL; X280/P870 are measured in rvv-bench for generic kernels. | No libm numbers. **[M]** (absence) |
| **Rivos** | `rivosinc/veclibm` (§2.2) | the only public, per-function-LMUL, RVV-native libm with published X60 throughput. | long-double reference only. **[M]** |

### 2.7 PLCT / LLVM RVV work

- PLCT's public report archive (`plctlab/PLCT-Open-Reports`) has RVV/LLVM talks from 2020–2022 (RVV 0.8/0.9 era); no RVV math library. **[M]**
- Current RVV codegen work is upstream LLVM; the RISC-V Summit Europe 2025 poster "Improvements to RISC-V Vector code generation in LLVM" (LAU) describes llvm-exegesis support for RVV, scheduling models, VLEN=32, and `zvzip`/`zvabd` experiments. No libm numbers. **[M]** ([poster PDF](https://riscv-europe.org/summit/2025/media/proceedings/2025-05-15-RISC-V-Summit-Europe-P1.4.03-LAU-poster.pdf))

### 2.8 GitHub repos implementing vector sin/cos/exp/log/pow for RVV

Beyond `veclibm` and SLEEF, no actively maintained, accuracy-documented RVV transcendental library surfaced. The rvv-bench ecosystem (below) is the main public source of *measurements*, not a library. **[M]** (search results) + **[I]**

---

## 3. Measured RVV instruction costs (the numbers to design against)

All values: **cycles per instruction, throughput** (independent instructions, `vl=VLMAX`, `ta/ma`), from [rvv-bench-results](https://camel-cdr.github.io/rvv-bench-results/). "cyc/el" is my division by elements per instruction at that VLEN.

### 3.1 XuanTie C920v2 (a210), VLEN=128

| instruction | e64m1 (2 el) | e64m2 (4 el) | e64m4 (8 el) | e64m8 (16 el) | e32m1 (4 el) | e32m4 (16 el) |
|---|---|---|---|---|---|---|
| `vfdiv.vv` | **20.00** (10/el) | 40.03 | 80.10 | 160.23 | 12.11 | 48.64 |
| `vfdiv.vf` | **6.37** (3.2/el) | 12.00 | 24.00 | 48.02 | 10.47 | 42.26 |
| `vfrdiv.vf` | 6.37 | 12.00 | 24.00 | 48.00 | 10.31 | 42.08 |
| `vfsqrt.v` | **16.96** (8.5/el) | 33.47 | 66.84 | 134.38 | 11.79 | 48.16 |
| `vfsqrt.v` masked | 10.42 | 26.11 | 50.01 | 98.20 | 10.92 | 40.00 |
| `vfrec7.v` | **0.50** (0.25/el) | 1.00 | 4.00 | 8.00 | 0.50 | 4.00 |
| `vfrsqrt7.v` | 0.50 (0.25/el) | 1.00 | 4.00 | 8.00 | 0.50 | 4.00 |
| `vfmacc.vv` | **0.55** (0.28/el) | 1.00 | 2.00 | 4.00 | 0.50 | 2.00 |
| `vfmul.vv` | 1.00 (0.5/el) | 2.00 | 4.00 | 8.00 | 1.00 | 2.00 |
| `vfadd.vv` | 0.50 (0.25/el) | 1.00 | 2.00 | 4.00 | 0.50 | 2.00 |
| `vrgather.vv` | **0.50** (0.25/el) | 2.36 | 8.00 | 32.00 | 0.50 | 8.00 |
| `vrgather.vx/vi` | 0.50 | 3.00 | 6.00 | 12.00 | 0.50 | 6.00 |
| `vrgatherei16.vv` | 1.36 | 2.36 | 8.00 | 32.00 | 1.36 | 8.00 |
| `vfirst.m` | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `vcpop.m` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `vmv.x.s` | 0.65 | 0.62 | 0.65 | 0.65 | 0.65 | 0.65 |
| `vfmv.f.s` | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| `vle64.v` | 1.25 (0.63/el) | 2.45 | 4.40 | 8.50 | 1.40 | — |
| `vse64.v` | 1.35 | 2.41 | 5.02 | 10.46 | 1.25 | — |
| `vfmerge.vfm` | 0.50 | 1.00 | 2.00 | 4.00 | 0.50 | 2.00 |

*Reading note:* the published HTML has empty `e64mf2` cells, so values are mapped by header order. I cross-checked the FMA rows against arXiv 2608.28097's "full-VLEN FMA per cycle" statement; the divide/sqrt/gather/load rows are unambiguous and are the ones used for the recommendations.

Key ratios **[M]**:
- `vfdiv.vv` / `vfdiv.vf` ≈ **3.1x** at e64m1 (20 vs 6.37).
- `vfdiv.vv` / `vfrec7` ≈ **40x** at e64m1 (20 vs 0.5).
- `vfsqrt` / `vfrec7` ≈ **34x** (16.96 vs 0.5).
- `vfdiv.vv` e64m1 costs **10 cycles/element**, i.e. ~5.3 ns/element at 1.9 GHz, *per divide in a throughput-bound loop*.
- Masked `vfdiv.vv` is **faster** than unmasked (13.00 vs 20.00 at e64m1); masked `vfsqrt` is faster (10.42 vs 16.96). Don't assume masking costs more on this core.

### 3.2 SpacemiT X60 (muse-pi), VLEN=256

FP ALU/divide/sqrt rows are **blank** on the published X60 page. Available rows **[M]**:

| instruction | e64m1 (4 el) | e64m2 (8 el) | e64m4 (16 el) | e64m8 (32 el) |
|---|---|---|---|---|
| `vrgather.vv` | **4.00** (1/el) | 16.00 (2/el) | 64.05 (4/el) | 256.25 (8/el) |
| `vrgather.vx/vi` | 2.00 | 4.00 | 8.00 | 16.00 |
| `vrgatherei16.vv` | 4.00 | 16.00 | 64.05 | 256.25 |
| `vmerge.vvm` | 2.00 | 4.00 | 8.00 | 16.00 |
| `vfirst.m` | 2.00 | 2.00 | 2.00 | 2.00 |
| `vcpop.m` | 2.00 | 2.00 | 2.00 | 2.00 |
| `vmv.x.s` | 2.00 | 2.00 | 3.00 | 6.00 |
| `vle64.v` | 2.40 (0.60/el) | 4.20 | 8.49 | 16.51 (0.52/el) |
| `vse64.v` | 2.82 | 4.31 | 8.58 | 19.82 |
| `vle64ff.v` | 5.45 | 7.12 | 11.32 | 19.56 |

Key facts **[M]**:
- `vrgather.vv` is **quadratic** in LMUL: per-element cost doubles each LMUL step (1 → 2 → 4 → 8 cycles/element).
- `vrgather.vx/vi` (scalar index) is **2x cheaper** than `.vv` at every LMUL (2 vs 4 at m1). Use `.vx/.vi` for fixed table indices.
- Loads: m1 0.60 cyc/el vs m8 0.52 cyc/el — only ~13% better per element, not a reason to raise LMUL.
- `vfirst`/`vcpop` throughput is 2 cycles; `vmv.x.s` grows to 6 cycles at m8.
- X60 page's own observations: "Maximizing LMUL is a good strategy for computation and load/stores, but **not for permutation instructions**"; "The vl doesn't impact performance, so a vl=1 and a vl=VLMAX instruction with LMUL=8 will take the same amount of time"; "When a gather operation doesn't need to cross register boundaries, multiple LMUL=1 or LMUL=2 gathers are faster." **[M]**

### 3.3 XuanTie C908 (VLEN=128) and SiFive P870 (VLEN=128)

| instruction | C908 e64m1 | C908 e64m2 | P870 e64m1 | P870 e64m2 | P870 e64m4 |
|---|---|---|---|---|---|
| `vfdiv.vv` | **41.0** (20.5/el) | 82.0 | **37.00** (18.5/el) | 74.02 | 148.12 |
| `vfdiv.vf` | 12.3 (6.2/el) | 24.8 | 10.00 (5/el) | 20.00 | 40.00 |
| `vfsqrt.v` | 27.0 (13.5/el) | 53.7 | 30.27 (15.1/el) | 58.66 | 120.42 |
| `vfrec7.v` | 2.0 (1/el) | 4.0 | 0.50 (0.25/el) | 1.00 | 2.00 |
| `vfmacc.vv` | 1.0 | 2.0 | 0.62 | 1.25 | 2.50 |
| `vrgather.vv` | 4.0 (2/el) | 16.4 | 1.00 (0.5/el) | 10.00 | 14.00 |
| `vfirst.m` | 2.0 | 2.0 | 2.47 | 2.47 | 2.47 |
| `vmv.x.s` | 2.0 | 2.0 | 2.41 | 2.41 | 2.41 |

**[M]**. P870 is a 6-wide OoO core yet has *worse* `vfdiv.vv` per element than C920v2 (18.5 vs 10 cycles/element) — vector division is a weak spot across RVV implementations, not just on in-order cores.

### 3.4 What the instruction tables imply for our kernels

1. **One `vfdiv.vv e64m1` occupies the divider for ~20 cycles in throughput (2 elements), and a dependent chain serializes them.** Three divisions in `erfc`'s tail therefore cost ≥60 cycles per 2-element strip (≈30 cycles/element, more if latency-bound), i.e. ~16 ns/element at 1.9 GHz, before polynomial work. Our `erfc` was 181 ns/element before replacing those divisions and 120 after; the divide share is plausible. **[M]** + **[M-ours]** + **[I]**.
2. **`vfdiv.vf` (scalar denominator) is 3x cheaper than `.vv`** on C920v2/C908, and `vfrec7` is ~40x cheaper on C920v2 (20 vs 0.5 cycles per e64m1 instruction). Any quotient whose denominator is loop-invariant or can be formed once should never use `.vv`. **[M]**
3. **LMUL does not change cycles/element** for div/sqrt/FMA (the reciprocal estimate steps at m4, §3.1); it changes cycles/instruction proportionally. Therefore:
   - LMUL 2 halves the number of `vsetvl`s and loop branches per element (good for throughput-bound short kernels),
   - but does **not** shorten the dependency chain per element (no help for latency-bound kernels),
   - and doubles the live-register footprint (bad when the chain already needs many temporaries).
   This is the quantitative explanation of our "LMUL 4 is worse on both boards" observation. **[M]** + **[M-ours]** + **[I]**.
4. **`vrgather` at m1/m2 is cheap enough for table-based range reduction** (0.25–0.6 cycles/element on C920v2), but at m4 it costs 1–2 cycles/element and at m8 4–8, so table lookup must be done at low LMUL or in a separate pass. **[M]**

---

## 4. RVV performance papers and reports

### 4.1 "Closer in the Gap: Towards Portable Performance on RISC-V Vector Processors" (arXiv 2605.10860, 2026)

- Testbed: **Milk-V Jupiter and Banana Pi BPI-F3, SpacemiT X60, 8 cores, "2 × 256-bit" RVV 1.0, 1.6 GHz (BPI-F3) / 1.8 GHz (Jupiter), 32 KB L1I/L1D, 2×512 KB L2, 16 GB LPDDR4/4X**. **[M]**
- Method: assembly microbenchmarks, dependency-free loops, 10^8 iterations, `vl=VLMAX`, `vta/vma` agnostic; perf-counter calibration on RVV. **[M]**
- Findings **[M]**:
  - **Predication masks cost ~35% vs unmasked `vsetvl` for handling tail elements.**
  - **Strided vector loads/stores cost up to 4x unit-stride.**
  - **`vdiv.vv` and scalar `div` are both very low throughput (0.22–1.97 Gops/s)**; the authors explicitly recommend replacing division with shift/multiplication where possible.
  - **"Default LMUL selection proves near-optimal in most cases"**, though GCC 15 has more headroom from larger LMUL tuning than Clang 21.
  - GCC 15 vs Clang 21 across six HPC/ML proxy apps: GCC 15 wins 4/6; Clang wins SGEMM/DGEMM via more aggressive instruction reduction; STREAM/SpMV get no autovec benefit.
  - GCC 15 introduced **dynamic LMUL selection**, early-break vectorization, misaligned-access fixes; LLVM 21 added hardware-specific scheduling models.
  - Manual intrinsics still beat compiler autovec for irregular patterns; Qsim (intrinsics) shows GCC 15 > LLVM 21.
- Relevance to us: the 35% predication result argues for **`vsetvl`-based tail handling** (which our strip loop already does) over mask-based tail folding; the division result reinforces §3; the "default LMUL near-optimal" result is about autovec, not hand-written intrinsics, so it does not contradict our per-function LMUL specialization. **[I]**

### 4.2 "Great Expectations: Benchmarking the Real-World Performance of RVV 1.0 in HPC" (arXiv 2608.28097, Aug 2026)

- Platforms: **X60/K1 (VLEN 256, DLEN 256, 12.8 GFLOP/s FP64 per core), C920v2/SG2044 (VLEN 128, 20.8 GFLOP/s/core), X280 (VLEN 512, 14 GFLOP/s/core), X100/A100 (K3), Grace (SVE2) reference**. **[M]**
- "The X60, X100 and C920v2 cores have a SIMD-like setup, with a theoretical throughput of a full-VLEN RVV FMA per cycle"; X280 and A100 take 2 and 4 cycles per full-VLEN instruction. **[M]**
- FMA-chain throughput with `vle` inserted every N FMAs (FLOP/cycle, FP64) **[M]**:

| platform | baseline (no load) | 1 load per chain | 4 loads | 14 loads | 26 loads |
|---|---|---|---|---|---|
| X60 (K1) | 7.999 | **2.6624** | 5.8561 | 7.4184 | 7.4176 |
| C920v2 (SG2044) | 7.999 | **1.9989** | 6.6409 | 7.9238 | 7.9238 |
| X280 | 7.9871 | 3.2171 | 7.9873 | 7.9875 | 7.9873 |
| Grace | 15.9999 | 9.9426 | 15.9999 | 15.9999 | 15.9999 |

- Interpretation for us: **one load per FMA chain destroys 67% (X60) / 75% (C920v2) of peak**; four loads per chain already recovers most. Our kernels do one load per element plus one store, so they are not in this regime, but it explains why adding a table load per element is expensive at VLEN=128. **[I]**
- The paper's HPC library results (BLAS/FFTW/HPL) are GEMM/FFT-level, not per-element libm; not directly comparable. **[M]**

### 4.3 "A Performance Analysis of Autovectorization on RVV RISC-V Boards" (PDP 2025)

- Scope: autovectorization quality on RVV boards; I could only retrieve the abstract/pages (PDF text extraction unavailable in this environment). It is relevant as corroboration that compiler autovec on RVV is still uneven, but I do not cite specific numbers from it. **[M]** (paper exists) **[I]** (no numbers used). ([PDF](https://cosenza.eu/papers/CarpentieriPDP25.pdf))

### 4.4 rvv-bench per-core observations (community measurements)

The most useful published engineering notes, all **[M]**:
- **X60:** maximize LMUL for computation/load-store, not permutation; `vl` does not affect instruction time; fault-only-first loads slightly slower; gathers that don't cross register boundaries are faster as multiple m1/m2 gathers; manual tail handling not worth it; manual unrolling helps very small loops; source/dest alignment didn't matter in streaming; masked arithmetic slower for LMUL<8 only due to dependency, not intrinsic cost.
- **C920v2 / A210:** VLEN=128, 1.9 GHz; the page is the source of §3.1.
- **P870:** VLEN=128, 6-wide OoO; §3.3.

---

## 5. Vector-length-agnostic (VLA) programming practice

### 5.1 The single-binary rule and the one flag that breaks it

- GCC's RVV model: scalable vector types; `-mrvv-vector-bits=scalable` (default) treats the `-march` VLEN as a **minimum**; `-mrvv-vector-bits=zvl` treats it as **exact**. GCC 14's release notes say the `zvl` option and `riscv_rvv_vector_bits` attribute generate code that **"is NOT portable between the core with different VLEN"**. **[M]** ([GCC 14 changes](https://gcc.gnu.org/gcc-14/changes.html), [GCC RISC-V options](https://gcc.gnu.org/onlinedocs/gcc/RISC-V-Options.html))
- LLVM equivalent: `-mrvv-vector-bits=zvl` or `-mllvm -riscv-v-vector-bits-max=VLEN`; LLVM's docs state that exact VLEN must be known at compile time for fixed-length `insert_subvector`/`extract_subvector` legalization, and that `vscale = VLEN/64`. **[M]** ([LLVM RVV docs](https://llvm.org/docs/RISCV/RISCVVectorExtension.html))
- **For us:** one static configuration that must serve VLEN=128 and 256 ⇒ keep `scalable`; never `zvl`. A `zvl128b` build would run on X60 but only use 128 of 256 bits per strip, halving throughput — exactly the failure mode we are trying to avoid. **[I]** (direct consequence).

### 5.2 Recommended loop shape (already ours, keep it)

The canonical VLA strip loop from the RVV intrinsics spec (v1.0-ratified) is:

```c
for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
    vl = __riscv_vsetvl_e64m2(n);   /* one vsetvl per strip, AVL = remaining */
    vfloat64m2_t vx = __riscv_vle64_v_f64m2(x, vl);
    /* ... kernel transforms vx ... */
    __riscv_vse64_v_f64m2(y, vx, vl);
}
```

Properties that matter **[M]** + **[I]**:
- `vsetvl` computes `vl = min(AVL, VLMAX)` and sets `vtype`; with a constant LMUL/SEW the compiler can keep `vtype` in a register and only update `vl`. This is the cheapest tail handling; the alternative (mask off lanes with `v0.t`) costs **~35%** in the arXiv microbenchmarks. **[M]**
- One `vsetvl` per strip is fine; do not put a `vfirst`/`vcpop` guard in the loop (§6.3). LLVM's `RISCVVLOptimizer` exists specifically to reduce AVL and avoid `vsetvli` toggles; GCC has no equivalent pass, so source-level discipline matters more on GCC. **[M]** (LLVM docs) + **[I]**.
- `-mrvv-max-lmul` controls **autovectorization only**; intrinsics carry their LMUL in the type (`vfloat64m2_t`), so it does not constrain our kernels. It does affect scalar fallback loops and any autovectorized helper. **[M]** (GCC docs) + **[I]**.

### 5.3 Does the compiler auto-select LMUL?

- **GCC:** default `-mrvv-max-lmul=m1` ("for compatibility with existing hardware that does not support the other options"); `dynamic` selects LMUL under register pressure; `-madjust-lmul-cost` (off by default) multiplies instruction latency by LMUL in the cost model. **[M]** (GCC docs)
- **LLVM:** `-mrvv-max-lmul=m1|m2|m4|m8`; default LMUL choice in the vectorizer; hardware scheduling models in LLVM 21. **[M]** (LLVM docs; arXiv 2605.10860)
- **Published measurement:** default LMUL selection is near-optimal for autovectorized HPC/ML apps on X60, with GCC 15 gaining more from explicit larger LMUL than Clang 21. **[M]** (2605.10860)
- **For hand-written intrinsics:** the compiler does not choose; our `#define VM_LMUL` does. The published per-function policy from `veclibm` (m2 default, m1 for long chains, m4 only for exp/cbrt) is the best available template. **[M]** + **[I]**

### 5.4 Function calls, vector arguments, and the calling convention

- The RISC-V psABI defines a **Standard Vector Calling Convention Variant**: mask args in `v0`, vector data/tuple args in `v8–v23`, return values like the first arg; **`v1–v7` and `v24–v31` must be preserved across the call**. C/C++ functions opt in with `__attribute__((riscv_vector_cc))`; GCC 14 added support ("Support for the standard vector calling convention variant", Lehua Ding/RiVAI). **[M]** ([psABI riscv-cc.adoc](https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-cc.adoc), GCC 14 changes)
- Even with `riscv_vector_cc`, a call inside a strip is expensive: GCC PR 121018 shows a function taking one `vuint8m1_t`, calling `foo()`, and returning it spills **`v0–v8, v24–v31`** although only `v8` is live. That is 17 vector registers of stack traffic per call. **[M]** ([PR 121018](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121018))
- **Recommendation:** no calls in the hot strip. If a rare-lane fixup is needed, accumulate a mask and call once after the loop; if a helper must be called, keep it `noinline` + `riscv_vector_cc` and pass **pointers**, not vector values. A conforming callee may clobber `v0` and `v8–v23`; it must preserve `v1–v7` and `v24–v31` — but GCC PR 121018 shows the caller spills those preserved registers anyway. **[I]**, grounded in **[M]**.

---

## 6. Toolchain codegen pitfalls (documented) and workarounds

### 6.1 GCC RVV register allocation and spills are a known, open problem

| report | what it documents | status/date |
|---|---|---|
| **GCC PR 121018** | "RVV codegen: spills unused vector registers on function call between RVV intrinsics." Minimal repro `vuint8m1_t bar(vuint8m1_t x){ foo(); return x; }` spills `v0–v8, v24–v31`; only `v8` is needed. | UNCONFIRMED, 2025-07-09, camel-cdr **[M]** |
| **GCC PR 121051** | "RVV intrinsics: unnecessary spilling and bad register allocation." Reporter: "GCC codegen for RVV intrinsics is often a lot worse than LLVM … redundant spills and moves"; follow-up 2026-02-14: "**Half of the libjpeg-turbo RVV kernels spill in gcc, while they have very neat codegen in clang**." | UNCONFIRMED, 2025-07-13; follow-up 2026-02-14 **[M]** |
| **LLVM issue 105661** | "[RISCV][MI] Support partial spill/reload for vector registers" — LLVM currently spills whole vector registers, not sub-registers; open. | open **[M]** |

These are exactly the symptoms behind our a210 `atan2` 316 ns/element and the GCC 14 vs 15 3.2x spread: a spill of a few LMUL-2 groups per strip multiplies the cost of a 50-instruction kernel. The GCC 14→15 difference is not a single documented bug; it is the aggregate effect of RA/cost-model changes (GCC 15 added dynamic LMUL, early-break vectorization, misaligned-access fixes). Treat **compiler version as part of the kernel ABI** and record it with every benchmark. **[M]** (PRs) + **[M-ours]** (316 vs 99) + **[I]**.

### 6.2 LMUL ≥ 4 and register pressure

- 32 vector registers total; an `m8` value occupies all 32; an `m4` value 16. A kernel with 3–4 live `m4` groups has no headroom. LLVM's register allocator has four classes (`VR`, `VRM2`, `VRM4`, `VRM8`) and allocates vectors before scalars, then inserts `vsetvli`; GCC's allocator is the one reported to spill. **[M]** (LLVM docs; PR 121051)
- Published policy avoids high LMUL for math: SLEEF builds only RVVM1/RVVM2; `veclibm` uses m4 for just 4 of 82 entries; rvv-bench X60 notes "maximizing LMUL … not for permutation". **[M]**
- **Our measurement** that static LMUL 4 is worse on both boards is consistent with this, and with the linear cycles/element scaling in §3. **[M-ours]** + **[I]**

### 6.3 `vfirst` / `vmv.x.s` in a dependency chain

- Throughput: `vfirst.m` = 1 cycle (C920v2), 2 cycles (X60/C908), 2.47 (P870); `vmv.x.s` = 0.65 (C920v2), 2–6 (X60), 2.41 (P870). These look cheap, but they are **vector→scalar transfers**: they require the vector pipeline to produce a scalar operand, and on in-order cores they serialize with the surrounding vector chain. **[M]** (throughput) + **[I]** (latency/serialization).
- **Our measurement:** guarding a small block inside a long dependency chain with `vfirst` made one kernel **51% slower on muse-pi**. **[M-ours]**
- **Workaround:** two-pass specials. In the main strip loop, OR each "needs fixup" mask into an accumulator (`vmor.mm`); after the loop, do one `vfirst`/`vcpop` and, only if non-zero, run a scalar pass over the array. Common case: one mask OR per strip, no transfer, no branch, no call. Rare case: one extra array pass, which is cheaper than a call per strip. **[I]** (design), supported by **[M-ours]** and **[M]** PR 121018.
- Alternative if a second pass is unacceptable: hoist the guard to a coarse granularity (e.g. check every 8th strip) and accept that a rare special lane may be processed by the vector path first; only valid if the vector path is safe (no spurious `FE_INVALID`). **[I]**

### 6.4 `vsetvl` placement and toggling

- LLVM's `RISCVVLOptimizer` reduces AVL to the minimum demanded by users and "avoids unnecessary `vsetvli` toggles"; register allocation is split so `RISCVInsertVSETVLI` runs after vector RA but before scalar RA. **[M]** (LLVM docs)
- GCC has no equivalent pass; each intrinsic call may re-emit `vsetvli` if the compiler cannot prove the vtype is unchanged. In intrinsics code, keep one `vsetvl` per strip and use the **same** SEW/LMUL for all ops in the strip; avoid mixing `e64m2` and `e32m1` in the same loop body. **[I]**
- `vsetvl` cost itself is not separately published in rvv-bench (the suite lists it as a TODO: "cycle count for vsetvl"). The 35% predication result suggests `vsetvl` is cheaper than masking for tails. **[M]** (rvv-bench README TODO; 2605.10860) + **[I]**

### 6.5 Documented compiler-version sensitivity and how others avoid it

- `veclibm` ships **multiple implementations per function** and selects by macro; README explicitly says this is because implementations perform differently on different hardware and "the manual process can be automated, allowing one to determine the highest performing choice experimentally." That is the strongest published statement that RVV codegen is hardware/compiler-sensitive. **[M]**
- SLEEF isolates RVV in separate LMUL-1/LMUL-2 backends and separate per-VLEN configs; it does not attempt one kernel for all. **[M]**
- LLVM's RVV docs list the machinery needed to get good code (VL optimizer, split RA, pseudo-instructions per LMUL, `vsetvli` insertion after RA); GCC's equivalent machinery is younger. **[M]**
- **Practical workarounds [I]:**
  - Pin the compiler per board and record the exact version in benchmark reports (our 316 vs 99 ns/element is otherwise irreproducible).
  - Keep kernels small and branch-free; large inlined expressions with many live groups are what spill.
  - Prefer `m1`/`m2`; if a kernel spills at m2, drop to m1 before changing the algorithm.
  - Use `__attribute__((noinline))` + `riscv_vector_cc` for cold helpers, and never pass vector types by value.
  - If a single TU is pathological only under GCC 14, consider compiling that TU with GCC 15 (the objects are ABI-compatible) or with `-mrvv-max-lmul=m1` (for autovec parts) — but verify the whole-library mean, because GCC 15's mean on a210 is 2.26x vs 2.79x for GCC 14 (our measurement), i.e. it fixes `atan2` but loses elsewhere. **[M-ours]**

---

## 7. Precision and accuracy practice

### 7.1 Do RVV math libraries use double-double?

- **SLEEF:** yes. `src/common/dd.h` provides the `vdouble2` type and `ddadd2`/`ddmul`-style helpers; the 1-ulp (`u10`) functions use dd tails where a single double cannot hold the last bits (exp/log/pow/trig reduction). The same C sources compile for RVV, so the RVV backend inherits dd. **[M]** (source) + **[I]** (RVV inherits).
- **`veclibm`:** no dd in the kernels inspected. It uses FMA-heavy minimax polynomials (exp: degree 11), split constants (`P_HI`/`P_LO`), and a 64-entry table variant; accuracy is validated only against `long double`. **[M]**
- **glibc/LLVM libc:** glibc's x86/AArch64 libmvec is built from the same scalar-derived algorithms and some routines use double-double internally **[I]**; LLVM `mathvec` is f32-only and has no published ulp target **[M]**.
- **Ours:** dd tail only where the framework shows it is needed (`erf`, `erfc`, `lgamma`, `tgamma`, `hypot`, hyperbolic/inverse-hyperbolic), plain double elsewhere. **[M-ours]** (README)

### 7.2 What dd costs on RVV, versus a division

Counting vector FP ops. On C920v2 the published `e64m1` timings for ALU ops are ~0.5–1.0 cycle per instruction (the core has a 256-bit datapath built from 2×128-bit pipes), i.e. ~0.25–0.5 cycles/element; I use the conservative 0.5 cycles/element below:

| operation | vector FP ops | cycles/element (C920v2, e64) | vs `vfdiv.vv` (10 cyc/el) |
|---|---|---|---|
| `two_prod` (Dekker/FMA) | 2 (`vfmul` + `vfmacc`) | ~0.5–1.0 | **10–20x cheaper** |
| `two_sum` (fast2sum) | 3 | ~0.75–1.5 | **7–13x cheaper** |
| `two_sum` (full 6-op) | 6 | ~1.5–3.0 | **3–7x cheaper** |
| dd add (hi/lo) | ~7 | ~2–3.5 | **3–5x cheaper** |
| dd multiply | ~5–7 | ~1.5–3.5 | **3–7x cheaper** |
| `vfdiv.vv` | 1 instruction | **10** | 1x |
| `vfsqrt.v` | 1 instruction | **8.5** | 0.85x |

*Caveat:* the published C920v2 FP-ALU/reciprocal cells have empty `e64mf2` columns and some noisy values; the `vfdiv`/`vfsqrt`/`vrgather`/load rows are the load-bearing measurements. The dd conclusion is robust to the ALU cost being 2x higher or lower, because the comparison is against a 10-cycle/element divide.

**[M]** for instruction costs; **[I]** for the op-count mapping (standard Dekker/FMA sequences, as in SLEEF `dd.h`).

**Implication:** on RVV, double-double is **cheap relative to division**. A kernel that replaces one `vfdiv.vv` with a reciprocal + dd residual is likely a win even if it adds 5–10 vector FMAs. Our own measurements agree: `erfc` 181→120 ns/element by replacing three divisions with one reciprocal and a dd multiply; `atan` 73.5→54.5 ns/element with the same trick. **[M-ours]** (README)

### 7.3 Accuracy vs speed tradeoffs published for RVV

- SLEEF: `_u10` (1.0 ULP) vs `_u35` (3.5 ULP) variants; the fast variants trade ~3.5x the error bound for fewer ops. No RVV-specific speed table published. **[M]** (naming) + **[M]** (absence).
- `veclibm`: four variants per function (`STD`, `STD_R_EXTRA`, `STD_EPSIM`, `TBL64`) selected at compile time; no published ulp-vs-speed curve. **[M]**
- rvv-bench `veclibm` table is the only public per-function RVV accuracy-adjacent performance data, but it does not report ulp. **[M]**
- **No published RVV "accuracy vs speed" Pareto data** was found for vector transcendentals. Our own MPFR-gated 1-ulp contract plus per-function ulp maxima is therefore a differentiator, not a catch-up item. **[I]**

### 7.4 Precision practice recommendation

- Keep the 1-ulp contract. The cheap path to it on RVV is **FMA + split constants + a dd tail only where the polynomial's rounding demands it**, not dd everywhere. **[I]**
- Never put a `vfdiv.vv` inside a dd sequence. Form one reciprocal (`vfrec7` + Newton) and use dd multiplies; on C920v2 that is ~3 cycles/element vs 10+ for one divide. **[M]** + **[I]**
- `vfrec7` alone is only ~7 bits; one Newton step gives ~14, two ~28, three ~56. For f64 1-ulp tails, 3 steps (6 FMAs) are needed unless the reciprocal feeds a polynomial that tolerates the error; for f32, 2 steps. **[I]** (standard Newton doubling; `vfrec7` precision per RVV spec is 7 bits).
- Beware `FE_INEXACT`: our README already documents that dd arithmetic makes it meaningless. Any new dd tail must preserve the existing per-lane flag contract for `FE_INVALID`/`FE_DIVBYZERO`/`FE_OVERFLOW`/`FE_UNDERFLOW`. **[M-ours]**

---

## 8. Concrete recommendations for `libvecmath`

Ordered by expected effect. Each is tagged with the evidence and the measurement I would use to accept/reject it.

### R1. Per-function static LMUL: m1 for long chains, m2 for the rest, never m4

- **What:** keep `VM_LMUL 2` as the default, but instantiate the 13 a210 sub-1.0x entries at `VM_LMUL 1`. The exact 13 from `bench-a210-gcc14.md`: `asinh`, `atan2`, `atan2f`, `atan2pi`, `atanpi`, `cbrtf`, `cospif`, `erfc`, `erfcf`, `expm1`, `log1p`, `log2`, `log2p1`. (`veclibm`'s m1 set — `pow`, `lgamma`, `tgamma`, `erfc`, `exp10`, … — differs because its algorithms differ; use its policy, not its exact list.)
- **Why:** C920v2 cycles/element is constant across LMUL for div/sqrt/FMA (§3.1), so m1 loses nothing in arithmetic throughput and halves register pressure; `veclibm` uses m1 for a similar long-chain/special class. Our LMUL-4 regression is the same effect at the other end. **[M]** + **[M-ours]**
- **How to measure:** a210 GCC 14.2, per-entry ns/element at m1 vs m2 with `n=65536`; accept if the m1 build is ≥5% faster or within noise while reducing spills (check `objdump` for `vs1r/vl1r` count).
- **Risk:** m1 doubles the number of strips and `vsetvl`s; for short arrays this may lose. Use m2 for simple throughput kernels (exp/log/sin/cos).

### R2. Replace `vfdiv.vv` with `vfrec7` + Newton (or `vfdiv.vf` when the denominator is scalar)

- **What:** for every quotient on a critical path, compute one reciprocal and multiply; keep a dd/Newton residual for the final rounding. For `atan2`, `atanpi`, `erfc`, `log1p`, `expm1`, `cbrtf` this is the main lever.
- **Why:** C920v2 e64m1: `vfdiv.vv` 20 cyc (10/el), `vfdiv.vf` 6.37 (3.2/el), `vfrec7` 0.5–1 (0.25–0.5/el); P870 37 vs 10 vs 0.5; C908 41 vs 12.3 vs 2.0. **[M]**
- **Expected:** our `erfc` went 181→120 ns/element and `atan` 73.5→54.5 with one reciprocal serving three divisions on muse-pi; the a210 gain should be larger because its `vfdiv.vv` is relatively worse. **[M-ours]** + **[I]**
- **Risk:** raw `vfrec7` is 7-bit; Newton steps add FMAs. Validate ulp maxima (must stay ≤1.0) and per-lane flags.

### R3. Two-pass specials: no `vfirst` and no call inside the strip

- **What:** in the strip loop, accumulate `special |= mask` (`vmor.mm`); after the loop, if `vfirst(special) >= 0`, run a scalar pass over the array that recomputes only flagged lanes. Replace the current per-strip `vfirst` + out-of-line `k_fixup` call.
- **Why:** `vfirst` in a chain cost us **51% on muse-pi**; GCC PR 121018 shows a call between RVV intrinsics spills 17 vector registers even with no vector args. Removing both from the hot loop is the highest-confidence structural fix for the a210 outliers. **[M-ours]** + **[M]**
- **Expected:** the 13 a210 outliers are 0.31–1.0x; if spills/calls are the cause, this should move most of them above 1.0x. `atan2` is the acid test (316 → target < 97 ns/element).
- **Risk:** the scalar pass is O(n); it only runs when a special lane exists. Choose the "special" predicate conservatively (rare) and ensure the vector path cannot raise spurious `FE_INVALID` on those lanes (use `vfclass`, not ordered compares, as our header already does). **[M-ours]**

### R4. Interleave two independent strips for latency-bound kernels (instead of raising LMUL)

- **What:** process strips `i` and `i+1` in the same loop body with two independent chains, at m1 or m2. The compiler sees two independent dependency chains and can overlap the divider/sqrt latency; register cost is 2× the per-strip groups, still ≤ an m4 group.
- **Why:** both cores are in-order; LMUL scaling is linear, so widening does not shorten the chain. Two chains at m1 give 2 independent element streams and preserve low pressure. This is classic software pipelining, not runtime dispatch. **[I]**, grounded in **[M]** (linear LMUL, in-order) and **[M]** (rvv-bench "manual unrolling helps very small loops").
- **How to measure:** a210 and muse-pi, the 13 outliers + `erfc`/`atan2`; accept if ≥10% and no register-spill regression. Start with the 4 worst a210 entries, not all 82.
- **Risk:** doubles code size and may increase spills if the kernel already has many live values; use it only for kernels that are provably chain-bound (per-element time roughly independent of n and of LMUL).

### R5. Fix the GCC 14 a210 outliers by code shape first, flags second

- **What:** for each sub-1.0x entry, dump asm and count `vs1r.v`/`vl1r.v` (vector spills) and calls inside the strip. Then, in order: (a) remove vector-by-value helper calls; (b) mark cold helpers `noinline` + `riscv_vector_cc` and pass pointers; (c) drop that kernel to m1; (d) only then try compiler flags/versions.
- **Why:** the 316 ns/element `atan2` is ~600 cycles/element, far above the arithmetic budget; GCC PR 121051 documents systemic RA spills. GCC 15.1 on the same board gives `atan2` 1.03x (99 ns/element) vs 0.31x, proving the source is not the limit. **[M-ours]** + **[M]**
- **Do not** globally switch a210 to GCC 15.1: our measured mean drops from 2.79x to 2.26x (GCC 15 fixes `atan2` but loses other entries). Per-TU/object selection is the pragmatic middle ground, and is still a static build, not runtime dispatch. **[M-ours]**
- **Risk:** mixing objects from two GCC versions must respect the ABI (it does for C, same `lp64d`, no LTO across versions); record the mapping in the build system.

### R6. Table lookups at m1 with `vrgather.vx/vi`; keep polynomials at m2

- **What:** if a kernel uses a table (e.g. `log2p1`'s `c/(u ln2)` or exp's 64-entry variant), do the gather at m1 and broadcast/merge, or use `vrgather.vx/vi` with scalar indices. Do not use `vrgather.vv` at m2+.
- **Why:** X60: `vrgather.vv` 4/16/64/256 cycles at m1/m2/m4/m8 (1/2/4/8 cycles per element); `vrgather.vx` 2/4/8/16. C920v2: 0.5/2.36/8/32. The X60 page explicitly says max LMUL is bad for permutation. **[M]**
- **Expected:** table-based range reduction can be cheaper than a long polynomial at VLEN=128, but only at m1. **[I]**
- **Risk:** gather index range and mask semantics; a gather at m1 costs a separate strip/pass and may lose more on loop overhead than it gains.

### R7. Native f32 kernels for the remaining widen/narrow entries

- **What:** implement f32 `cbrt`, `erfc`, `atan2`, `cospi` natively (f32 coefficients + one f32 fast path + f64 fallback per lane), as was already done for exp/log/trig.
- **Why:** a210 GCC 14.2: `cbrtf` 31.5 vs scalar 23.0 (0.73x), `erfcf` 37.3 vs 28.3 (0.76x), `atan2f` 65.1 vs 40.6 (0.62x), `cospif` 31.8 vs 26.2 (0.82x). These are exactly the entries still wrapping f64 kernels. **[M-ours]**
- **Expected:** native f32 halves the vector work and removes the widen/narrow conversions; the f32 `atan2f` gain should be large because it also removes the f64 divide. **[I]**
- **Risk:** the per-lane f64 fallback must preserve the existing special-value/flag contract; f32 ulp gate (≤1.0, 0.5 where possible) is tighter in absolute terms.

### R8. Keep `scalable`, keep one binary, and tune for the weaker core

- **What:** build with `-march=rv64gcv_zvfh` (or the board's full ISA), `-mrvv-vector-bits=scalable` (default), no `zvl`. For `-mtune`, GCC 15 offers `spacemit-x60`, `xt-c920v2`, `generic-ooo`; one binary cannot be tuned for both, so test `generic-ooo` vs `xt-c920v2` and pick the one that does not regress muse-pi.
- **Why:** `zvl` is explicitly non-portable across VLEN (GCC 14 notes); scalable is the only single-binary option. `-mtune` affects scheduling/cost model, and the a210 is the weaker target (13 entries ≤1.0x), so tuning for it is the safer default. **[M]** + **[I]**
- **Risk:** `generic-ooo` may schedule for OoO and hurt the in-order X60; measure both boards, not just a210.

### R9. Small-n scalar threshold (static, no CPU dispatch)

- **What:** for `n` below a small threshold (e.g. `n < 2 * VLMAX`), call the scalar libm in a simple loop instead of entering the vector strip loop.
- **Why:** a210's scalar FPU is fast (`atan2` 97 ns/element scalar vs 316 vector; `cbrtf` 23 vs 31.5); for short arrays the vector prologue, `vsetvl`, and (currently) fixup dominate. The threshold is a compile-time constant, not runtime LMUL/VLEN dispatch, so it respects the constraint. **[M-ours]** + **[I]**
- **Risk:** on muse-pi the vector path may already win at small n; choose the threshold from both boards (likely 8–32 elements) and keep it per-function only if needed.

### R10. Treat compiler version as part of the kernel ABI

- **What:** record GCC/Clang version, `-march`, `-mtune`, `-O` level, and the exact commit in every benchmark report; add an asm-spill check (`vs1r`/`vl1r` count) to CI for the 13 problem kernels.
- **Why:** the same source moved `atan2` by 3.2x between GCC 14.2 and 15.1 on the same board; GCC PR 121051 shows this is systemic, not a one-off. Without recording the toolchain, the numbers are not reproducible. **[M-ours]** + **[M]**
- **Risk:** CI asm checks are toolchain-specific; keep them advisory, not gating, until the pattern is stable.

---

## 9. Risks and pitfalls (with citations)

| risk | evidence | mitigation |
|---|---|---|
| **GCC spills around calls inside a strip** | PR 121018: `v0–v8, v24–v31` spilled for one live `v8` **[M]** | No calls in strip; two-pass fixup; pointers not vector values; `noinline`+`riscv_vector_cc` for cold code. |
| **GCC RVV RA generally poor** | PR 121051 + 2026-02-14 comment: half of libjpeg-turbo RVV kernels spill **[M]** | Small kernels, m1/m2, asm audit, per-TU compiler choice. |
| **LLVM whole-register spills** | LLVM issue 105661 (partial spill/reload not supported) **[M]** | Same: keep live groups few; avoid m4+. |
| **`-mrvv-vector-bits=zvl` breaks the single binary** | GCC 14 notes: "NOT portable between the core with different VLEN" **[M]** | Keep `scalable`; never `zvl` in shipped builds. |
| **LMUL ≥4 pressure** | 32 regs; m8 = all; SLEEF m1/m2 only; veclibm m4 for 4/82; our m4 regression **[M]** + **[M-ours]** | Cap at m2; m1 for long chains. |
| **`vrgather` quadratic** | X60 4/16/64/256; C920v2 0.5/2.36/8/32 **[M]** | Gather at m1; `.vx/.vi`; avoid at m4+. |
| **`vfirst`/`vmv.x.s` serialization** | 51% regression (ours); 1–2.5 cycle throughput but vector→scalar **[M-ours]** + **[M]** | Two-pass specials; hoist guards. |
| **Predication overhead** | ~35% vs unmasked `vsetvl` for tails (X60, 2605.10860) **[M]** | Strip with `vsetvl`; masks only for data-dependent lanes. |
| **Vector divide/sqrt throughput** | 10 cyc/el (C920v2), 18.5 (P870), 20.5 (C908) **[M]** | Reciprocal + Newton; `.vf` when possible; dd residual. |
| **Compiler-version fragility** | 316 vs 99 ns/element (ours); PR 121051; GCC 15 dynamic LMUL **[M]** | Pin/record toolchain; per-TU selection; asm CI. |
| **SLEEF RVV unmaintained** | master README "Unmaintained" **[M]** | Do not depend on it; borrow algorithms only. |
| **glibc libmvec absent on RISC-V** | `sysdeps/riscv/rvv/` only `memset.S`; RFC unmerged **[M]** | Keep our VFABI aliases; no `-fveclib` reliance. |
| **LLVM `mathvec` has no RVV backend** | RFC list: x86_64 + AArch64 NEON only **[M]** | Not a dependency. |
| **`vfrec7` accuracy** | ~7-bit estimate per RVV spec **[M]** | 2–3 Newton steps; validate ulp. |
| **dd + division interaction** | 10 cyc/el per divide vs ~1 cyc/el per dd FMA **[M]** + **[I]** | Never divide inside dd; one reciprocal + dd multiply. |
| **`FE_INEXACT` semantics with dd** | our README documents it as meaningless **[M-ours]** | Keep contract wording; verify other flags per lane. |
| **rvv-bench X60 FP cells blank** | published table **[M]** | Use C920v2/C908 for FP costs, X60 `veclibm` for end-to-end; do not extrapolate. |
| **Throughput ≠ latency** | rvv-bench page says "estimate the instruction throughput" **[M]** | Do not size chains from these tables alone; measure dependent loops. |

---

## 10. What is measured vs. inferred (explicit ledger)

**Measured in external sources [M]:**
- All rvv-bench instruction timings in §3.1–3.3, including the linear LMUL scaling of `vfdiv`/`vfsqrt`/`vfmacc`, the quadratic `vrgather`, and `vfirst`/`vmv.x.s` throughput.
- X60 `veclibm` elements/cycle and speedups in §2.2 (VLEN=256, 1.6 GHz).
- SLEEF versions, LMUL-1/2-only backends, "Unmaintained" status, per-VLEN configs.
- `veclibm` per-function LMUL counts (58 m2 / 20 m1 / 4 m4), algorithms, long-double reference.
- glibc libmvec absence on RISC-V; RFC V4; `sysdeps/riscv/rvv/memset.S` only.
- LLVM `mathvec` scope and absence of RVV.
- GCC 14 `-mrvv-max-lmul` default m1, `dynamic`, vector CC support, `zvl` non-portability.
- LLVM scalable-vector model, `vscale = VLEN/64`, `RISCVVLOptimizer`, split RA.
- psABI vector calling convention rules (`v0`, `v8–v23`, preserve `v1–v7`/`v24–v31`).
- GCC PR 121018 / 121051 text and dates; LLVM issue 105661.
- arXiv 2605.10860 (35% predication, 4x strided, 0.22–1.97 Gops/s div, default LMUL near-optimal, GCC15 dynamic LMUL).
- arXiv 2608.28097 hardware table and FMA-chain table.

**Measured by our project [M-ours]:**
- muse-pi mean 3.08x, all 82 >1.0x; a210 mean 2.79x (GCC 14.2), 13 entries ≤1.0x; GCC 15.1 mean 2.26x; `atan2` 316 vs 99 ns/element; LMUL 4 worse on both; `vfirst` guard +51% on muse-pi; latency-bound cost doubles at VLEN=128 while throughput-bound barely changes; `erfc` 181→120; `atan` 73.5→54.5; per-entry a210 numbers in R1 and `test/results/board-a210/bench-a210-gcc14.md`.
- The README's per-entry ulp maxima and speedups (FULL tier on muse-pi).

**Inference / engineering judgement [I]:**
- That the a210 outliers are dominated by spills/calls rather than arithmetic (supported by cycle arithmetic and GCC PRs, not proven by a profiler here).
- That two-pass specials will recover most of the a210 gap (design proposal; needs measurement).
- That two-strip interleaving will help latency-bound kernels at VLEN=128 (standard software pipelining; not measured in this survey).
- That native f32 `cbrt`/`erfc`/`atan2` will beat the widen/narrow path on a210 (based on the cost of conversions and the strong scalar baseline; not measured).
- That m1 will be neutral-to-better for the 13 outliers (based on linear LMUL scaling and veclibm policy; not measured on our kernels).
- The dd op-count → cycles/element mapping in §7.2 (standard sequences; instruction costs are measured).
- All statements about what "should" happen on X60 FP divides (its table is blank).

**Claims not used / deliberately excluded:**
- OpenCV's operator-level speedups (no VLEN/LMUL/per-element numbers; not comparable).
- Vendor slide numbers without a reproducible harness.
- Any SLEEF RVV performance number (none found).
- PDP25 specific numbers (PDF text not extractable here).

---

## 11. Sources

1. rvv-bench results, SpacemiT X60: <https://camel-cdr.github.io/rvv-bench-results/spacemit_x60/index.html> (and `veclibm.html`)
2. rvv-bench results, XuanTie C920v2 (Zhihe A210): <https://camel-cdr.github.io/rvv-bench-results/xuantie_c920v2/index.html>
3. rvv-bench results, XuanTie C908 (K230): <https://camel-cdr.github.io/rvv-bench-results/canmv_k230/index.html>
4. rvv-bench results, SiFive P870: <https://camel-cdr.github.io/rvv-bench-results/sifive_p870/index.html>
5. rvv-bench source and README: <https://github.com/camel-cdr/rvv-bench>
6. `rivosinc/veclibm`: <https://github.com/rivosinc/veclibm> (README, `include/rvvlm.h`, `include/rvvlm_expD.inc.h`)
7. SLEEF: <https://github.com/shibatch/sleef> (README.adoc supported-architecture table; `src/arch/helperrvv.h`; `src/common/dd.h`; CMake RVVM1/RVVM2; releases 3.6/3.6.1)
8. SLEEF PR #477 "Integrate RISC-V support": <https://github.com/shibatch/sleef/pull/477>
9. glibc libmvec RISC-V RFC V4: <https://sourceware.org/pipermail/libc-alpha/2024-May/156644.html>
10. glibc `sysdeps/riscv/rvv/` (only `memset.S`): <https://github.com/bminor/glibc/tree/master/sysdeps/riscv/rvv>
11. LLVM libmvec RISC-V patch: <https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg509884.html>
12. LLVM libc `mathvec` RFC: <https://discourse.llvm.org/t/rfc-mathvec-introduce-the-llvms-mathvec-library-as-yet-another-veclib-functions-provider/92591>
13. OpenCV RVV HAL blog: <https://opencv.org/introducing-hal-riscv-rvv-unleashing-the-power-of-risc-v-cpus-with-rvv-1-0/>; wiki: <https://github.com/opencv/opencv/wiki/OpenCV-RISC-V>
14. Andes `libvec`: <https://github.com/andestech/libvec>
15. XuanTie GitHub org: <https://github.com/XUANTIE-RV>; compute-library page: <https://www.xrvm.cn/soft-tools/application/algorithm-library>
16. PLCT open reports: <https://github.com/plctlab/PLCT-Open-Reports>
17. RISC-V Summit Europe 2025 LLVM RVV poster: <https://riscv-europe.org/summit/2025/media/proceedings/2025-05-15-RISC-V-Summit-Europe-P1.4.03-LAU-poster.pdf>
18. arXiv 2605.10860, "Closer in the Gap: Towards Portable Performance on RISC-V Vector Processors": <https://arxiv.org/abs/2605.10860>
19. arXiv 2608.28097, "Great Expectations: Benchmarking the Real-World Performance of RVV 1.0 in HPC": <https://arxiv.org/abs/2608.28097>
20. PDP 2025, "A Performance Analysis of Autovectorization on RVV RISC-V Boards": <https://cosenza.eu/papers/CarpentieriPDP25.pdf>
21. GCC 14 RISC-V changes: <https://gcc.gnu.org/gcc-14/changes.html>
22. GCC RISC-V options (`-mrvv-vector-bits`, `-mrvv-max-lmul`, `-mtune`): <https://gcc.gnu.org/onlinedocs/gcc/RISC-V-Options.html>
23. LLVM RISC-V Vector Extension docs: <https://llvm.org/docs/RISCV/RISCVVectorExtension.html>
24. RISC-V psABI calling convention (vector variant): <https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-cc.adoc>
25. RVV intrinsics spec v1.0-ratified: <https://github.com/riscv-non-isa/riscv-rvv-intrinsic-doc/releases/tag/v1.0-ratified>
26. GCC PR 121018: <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121018>
27. GCC PR 121051: <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121051>
28. LLVM issue 105661: <https://github.com/llvm/llvm-project/issues/105661>
29. Our project: `newlib/libm/vecmath/README.md`, `newlib/libm/vecmath/test/results/bench-final.md`, `newlib/libm/vecmath/test/results/board-a210/bench-a210-gcc14.md`, `newlib/libm/vecmath/test/results/board-a210/bench-a210-gcc15.md`.

---

### Appendix A. Quick asm-audit recipe for the a210 outliers

```sh
# per entry, count vector spills in the hot loop
riscv64-linux-gnu-objdump -d --no-show-raw-insn libvecmath.a \
  | awk '/<vatan2>:/,/^$/' | grep -cE '\b(vs1r|vl1r|vs2r|vl2r|vs4r|vl4r|vs8r|vl8r)\.v'
# and count calls inside the function
riscv64-linux-gnu-objdump -d --no-show-raw-insn libvecmath.a \
  | awk '/<vatan2>:/,/^$/' | grep -cE '\b(call|jal|jalr)\b'
```

Acceptance: zero vector spills and zero calls in the strip loop of every entry that is ≤1.0x on a210. If a spill exists, reduce LMUL or live range before changing the math. **[I]** (method), based on **[M]** PR 121018/121051.

### Appendix B. Suggested measurement matrix for the next tuning round

| build | board | entries | metric |
|---|---|---|---|
| m2 (current) | muse-pi | 82 | ns/el, mean |
| m2 (current) | a210 GCC 14.2 | 82 | ns/el, mean |
| per-function m1 (R1) | both | 13 outliers + erfc/atan2 | ns/el, spill count |
| two-pass specials (R3) | both | 13 outliers | ns/el, asm calls |
| 2-strip interleave (R4) | both | 4 worst | ns/el, spill count |
| native f32 (R7) | a210 | cbrtf/erfcf/atan2f/cospif | ns/el, ulp |

Run with the existing harness (pinned core, board lock, best-of-9×3, `n=65536`) so the numbers are comparable to `test/results/board-a210/`. **[M-ours]**
