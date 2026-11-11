# libvecmath — RISC-V RVV vector math library

A double- and single-precision vector math library for RISC-V RVV 1.0,
built as part of newlib. It is compiled **only** when newlib is
configured with `--enable-vecmath`, and it produces a separate
`libvecmath.a` — nothing is added to `libm.a`.

All 41 functions exist at both precisions. Every entry is within 1 ulp
of the infinitely precise result against an MPFR reference (exact figures
per function in the table below), and all 82 entries are faster than
the scalar loop an untuned caller writes — mean 3.06x on a real RVV 1.0
board, slowest 1.06x.

## API

```c
#include <vecmath.h>
void vsin  (double *restrict y, const double *restrict x, size_t n);
void vsinf (float  *restrict y, const float  *restrict x, size_t n);
```

Array-style, VLA: every entry processes `n` elements and stripmines over
the hardware VLEN internally, so one binary runs on any RVV 1.0
implementation (VLEN 128 / 256 / 512 / 1024 / …). The single-precision
entries take the C99 `f` suffix and `float` arrays. Link with:

```
-lvecmath -lm -lc
```

### Contract

* `y` (and any second output) must not overlap the inputs.
* Pointers may have any alignment; `n == 0` is legal.
* **`errno` is never set** (libmvec / VFABI semantics). Floating-point
  exception flags in `fcsr` *are* raised per lane exactly as the scalar
  C99 functions would — except `FE_INEXACT`, which accumulates
  legitimately from the internal double-double arithmetic and is
  therefore not meaningful.
* Results are correctly rounded or within 1 ulp of the infinitely precise
  result. Signed zeros, infinities and NaN propagation follow IEEE-754 /
  C99 / SYCL.
* **Bare-metal users must enable the vector unit** (`sstatus.VS != Off`)
  before calling these functions; Linux user space does this
  automatically. The test program does it with
  `li t0, 0x2200; csrs sstatus, t0`.
* Weak symbols do not pull archive members: linking statically against
  `libvecmath.a` from a bare-metal image needs
  `-Wl,--whole-archive -lvecmath -Wl,--no-whole-archive`.

### Functions

Every entry below exists as `v<name>` (binary64, `double` arrays) and as
`v<name>f` (binary32, `float` arrays).

| group | entries |
|---|---|
| roots / power | `vsqrt` `vrsqrt` `vcbrt` `vpow` `vhypot` |
| exponential | `vexp` `vexp2` `vexp10` `vexpm1` |
| logarithmic | `vlog` `vlog2` `vlog10` `vlog1p` `vlog2p1` |
| trigonometric | `vsin` `vcos` `vtan` `vsincos` |
| pi-scaled trigonometric | `vsinpi` `vcospi` `vtanpi` `vsincospi` |
| inverse trigonometric | `vatan` `vatan2` `vatanpi` `vatan2pi` `vasin` `vacos` `vasinpi` `vacospi` |
| inverse hyperbolic | `vasinh` `vacosh` `vatanh` |
| hyperbolic | `vsinh` `vcosh` `vtanh` |
| special | `verf` `verfc` `vtgamma` `vlgamma` `vlgamma_r` |

Greenfield functions (those with no scalar C analogue) follow SYCL
semantics, which for the pi-scaled ones are the OpenCL C 3.0 edge cases.
With `n` an integer:

* `sinpi(±0) = ±0`, `sinpi(±n) = ±0` (`copysign(0, x)`), `sinpi(±inf) = NaN`
  with `FE_INVALID`
* `cospi(±0) = 1`, `cospi(n) = (-1)^n`, `cospi(n + 1/2) = +0`
* `tanpi(±0) = ±0`, `tanpi(n) = copysign(0, x)*(-1)^n`,
  `tanpi(n + 1/2) = (-1)^n * +Inf` with `FE_DIVBYZERO`
* `rsqrt(±0) = +Inf` with `FE_DIVBYZERO`, `rsqrt(+inf) = +0`
* `exp10(x) = 10^x`; `log2p1(x) = log2(1+x)`, never forming `1+x`

## How the single-precision entries are computed

The binary32 entries split into three shapes.

**Native binary32 kernels** (exp, exp2, exp10, expm1, log, log2, log10,
log1p, log2p1, sin, cos, tan, the pi-scaled trig, asin, acos, atan, atan2,
the pi-scaled inverse trig, sinh, cosh, tanh, asinh, acosh, atanh, hypot,
erf, erfc, tgamma, lgamma, lgamma_r). These evaluate in binary32
arithmetic — fdlibm's float coefficients, or a binary32 fast path — with
the value carried as a (hi, lo) double-float pair where a single binary32
rounding cannot reach the last bit, and one final narrowing. Every kernel
also has a **per-lane fallback**: lanes outside the fast range (huge,
tiny, subnormal, Inf/NaN, or a reduction that cancels too far) are
recomputed with the binary64 kernel through the widen/narrow path and
merged in by mask. Accuracy, special values and per-lane exception flags
therefore come from the wide path where it matters, while the common case
runs at binary32 cost. `scripts/domain-check.sh` verifies that the fast
path alone is still inside the contract.

**Widen/narrow wrappers** (sqrt, rsqrt, cbrt, pow). These are already
1.5-7x the scalar baseline, so they are left as the exact widening:
`f32mK` widens exactly into `f64m2K` (every binary32 value is a binary64
value), the binary64 kernel runs, and one correctly-rounded conversion
narrows back. Since a binary64 reference is accurate to a few binary64
ulps, which is under `2^-29` of a binary32 ulp, the result is correctly
rounded to within `0.5 ulp + 2^-29 ulp`. `vsqrtf` is native for the same
reason: `vfsqrt.v` on binary32 is correctly rounded on its own.

**Dual-output** (sincosf, sincospif, lgamma_rf) follow the same rules
with two results per lane; `lgamma_rf`'s sign is carried as an `int`
vector.

The binary32 underflow carrier is worth calling out: three entries
(`vexpm1f`, `vlog1pf`, `vpowf`) short-circuit the tiny region and return
the input (or, for `pow`, an exactly subnormal result) verbatim. For a
binary32 argument that is exact even where the binary32 operation itself
would have had to round a correction onto the much coarser binary32
subnormal grid, so the narrowing conversion stays silent; those entries
raise `FE_UNDERFLOW` explicitly on the lanes whose result is a nonzero
subnormal, with the same masked multiply-and-add device the binary64
`log2p1` kernel uses.

## Accuracy and speed

Both are measured with the library's own harness on real RVV 1.0
hardware (SpaceMiT X60, VLEN 256, 8x1.6 GHz) and under qemu:

* **accuracy** — `test/runner` against an MPFR reference, FULL tier
  (100 057 records per function), VLEN 256; the same PASS set at
  VLEN 128/512/1024. Reports: `test/results/acc-final.md`.
* **speed** — `test/bench/bench.c` vs the scalar loop an untuned caller
  writes (`-O2 -static`, no `-ffast-math`, errno-raising libm calls),
  same input array, n = 65 536, best of 9 reps x 3 runs, pinned core,
  board-side lock. Reports: `test/results/bench-final.md`.

Every one of the 82 entries is within 1 ulp (no lane over 1 ulp, no
special-class failure, no per-lane flag failure, sweep-stable across
strip lengths, and identical at every VLEN), and all 82 are faster than
the scalar baseline. The mean speedup is 3.93x; the slowest entry is
`lgamma` at 1.46x and the fastest `log1p` at 7.39x.

| function | max ulp | speedup | | function | max ulp | speedup |
|---|---|---|---|---|---|---|
| sqrt | 0.5000 | 4.43x | asin | 0.6383 | 1.69x |
| rsqrt | 0.8612 | 2.96x | acos | 0.6409 | 1.77x |
| cbrt | 0.6645 | 3.54x | atan | 0.9181 | 2.87x |
| exp | 0.5030 | 3.78x | atan2 | 0.9054 | 4.74x |
| exp2 | 0.5038 | 3.62x | asinpi | 1.0000 | 1.51x |
| exp10 | 0.5034 | 3.74x | acospi | 0.6624 | 1.57x |
| expm1 | 0.8130 | 3.57x | atanpi | 1.0000 | 3.15x |
| log | 0.7220 | 3.56x | atan2pi | 1.0000 | 6.54x |
| log2 | 0.6132 | 3.05x | sinh | 0.6446 | 4.57x |
| log10 | 0.9091 | 2.80x | cosh | 0.5645 | 3.27x |
| log1p | 0.6723 | 7.39x | tanh | 0.5230 | 3.88x |
| log2p1 | 1.0000 | 2.19x | asinh | 0.9076 | 3.80x |
| pow | 0.5053 | 3.44x | acosh | 0.5049 | 3.82x |
| sin | 0.7852 | 2.69x | atanh | 0.7360 | 5.44x |
| cos | 0.7846 | 2.52x | hypot | 0.5000 | 2.39x |
| tan | 0.7376 | 2.32x | erf | 1.0000 | 2.48x |
| sincos | 0.8345 | 3.58x | erfc | 0.7976 | 2.47x |
| sinpi | 1.0000 | 2.70x | tgamma | 1.0000 | 3.17x |
| cospi | 0.8067 | 2.50x | lgamma | 0.8505 | 1.46x |
| tanpi | 1.0000 | 3.84x | lgamma_r | 0.5619 | 1.47x |
| sincospi | 1.0000 | 3.73x |  | |  |

The binary32 entries:

| function | max ulp | speedup | | function | max ulp | speedup |
|---|---|---|---|---|---|---|
| sqrtf | 0.5000 | 7.05x | asinf | 0.8136 | 3.72x |
| rsqrtf | 0.5000 | 2.15x | acosf | 0.8893 | 4.33x |
| cbrtf | 0.5239 | 5.80x | atanf | 0.8238 | 4.62x |
| expf | 0.5257 | 6.26x | atan2f | 0.7150 | 4.35x |
| exp2f | 0.5000 | 3.44x | asinpif | 0.8055 | 4.35x |
| exp10f | 0.5000 | 3.50x | acospif | 0.5719 | 4.57x |
| expm1f | 0.6321 | 4.78x | atanpif | 0.8251 | 4.88x |
| logf | 0.5106 | 2.93x | atan2pif | 0.8029 | 3.32x |
| log2f | 0.5340 | 2.79x | sinhf | 0.6741 | 5.80x |
| log10f | 0.5279 | 5.43x | coshf | 0.5365 | 6.35x |
| log1pf | 0.5796 | 3.39x | tanhf | 0.5683 | 5.92x |
| log2p1f | 0.5010 | 3.95x | asinhf | 0.5238 | 2.84x |
| powf | 0.5569 | 4.89x | acoshf | 0.5055 | 3.80x |
| sinf | 0.6274 | 5.40x | atanhf | 0.5362 | 4.27x |
| cosf | 0.5613 | 4.02x | hypotf | 0.5000 | 4.09x |
| tanf | 0.6406 | 4.01x | erff | 0.5005 | 1.53x |
| sincosf | 0.7285 | 6.99x | erfcf | 0.5312 | 2.26x |
| sinpif | 0.6053 | 6.03x | tgammaf | 0.5002 | 7.29x |
| cospif | 0.6396 | 5.74x | lgammaf | 0.5011 | 5.51x |
| tanpif | 0.6286 | 4.60x | lgamma_rf | 0.5011 | 4.15x |
| sincospif | 0.6040 | 6.84x |  | |  |

Starred entries reach exactly 1.0 ulp, which the contract admits: they
round twice, once at 53-bit precision in a scaled domain and then onto
the coarser subnormal grid (`log2p1`, `sinpi`/`tanpi`/`sincospi`,
`atanpi`/`atan2pi`, `asinpi`, `atan2`, and `hypot` on subnormal-result
lanes), or they are `exp` of a very large `lgamma` (`tgamma`).

### Where the speed comes from

* The binary32 entries are no longer thin widen/narrow wrappers: each has
  a native binary32 kernel (fdlibm float coefficients carried as (hi, lo)
  double-float pairs, or a native f32 fast path) with a per-lane fallback
  to the wide path for lanes outside the fast range, so accuracy and flag
  semantics are inherited while the common case runs at f32 cost.
* The binary64 kernels that used to carry double-double arithmetic
  everywhere now run plain-double fdlibm/SLEEF-style algorithms with a
  double-double tail only where the framework shows it is needed
  (`erf`/`erfc`/`lgamma`/`tgamma`/`hypot`, the hyperbolic and inverse
  hyperbolic families), and the gamma kernels shrank from 25 KB of
  straight-line code to a few hundred instructions.
* The rare-lane scalar fixup epilogue is out of line: its 4.8 KB of stack
  buffers used to be charged to every strip of every kernel, which cost
  30-45% of hot-loop throughput on this core (that single change took
  `log2p1` from 0.99x to 1.38x, and it is what the `exp` family's
  3.5-4.1x depends on).
* The *call* to that epilogue is guarded by a `vfirst` at the call site.
  Passing vector groups by value makes the caller spill its live vector
  registers and build a frame on every strip, even with an empty mask --
  worth 0.3-2.1x per entry once removed (`exp` 1.8 -> 4.1x,
  `exp10` 1.9 -> 3.7x, `sinh` 2.6 -> 4.2x, `log` 1.4 -> 2.2x,
  `log10` 1.5 -> 2.8x, `pow` 2.4 -> 3.4x, `sincos` 2.4 -> 3.7x).
  Both effects together are the single largest source of the 2.76x ->
  3.05x step over the first optimization pass.
* `log2p1` is built as `log2(RN(1+x)) + c/(u ln2)` on the log2 core's dd
  pair, so the 1/ln2 scaling costs no extra rounding; its hot path is a
  leaf with the fixup behind a `vfirst` branch.

* **`atan`/`atanpi` got their quotient from two divisions on the critical
  path.**  One reciprocal now serves both: `r = rnum*rt` and the residual
  `(rnum - r*rden)*rt`, with `rt` formed at `2^-600` so that even
  `rden = |x|` on the `-1/ag` branch (up to the largest finite double)
  inverts without an early underflow.  The Newton residual absorbs the
  extra rounding, so the pair stays as accurate as a true dd divide --
  `atan` goes 73.5 -> 54.5 ns/element (1.32x -> 1.79x) and `atanpi`
  87.4 -> 79.3 (1.29x -> 1.42x), with the same ulp maxima.  The flag is
  compile-time and only the standalone kernels take it: at LMUL 4-8 the
  extra live vector costs more than the division it saves (measured:
  `atanf` 16.3 -> 19.2 ns/element), and inside `atan2`'s larger inlined
  expression the two-division form schedules better.

* **Hardware divides were the most expensive instruction in the library.**
  On the VLEN 256 board a *dependent* `vfdiv` chain costs 50 ns per strip
  and on the VLEN 128 board 40 ns, against 26/21 ns for `vfrec7` plus three
  Newton steps.  `include/vecmath_dd.h` now provides `VM_RCP_FAST` (two
  steps, for callers that add their own fma residual), `VM_RCP_FULL`
  (three steps, half an ulp), `VM_RCP_DD` (corrected pair) and `VM_QDIV`
  (quotient with its own residual correction, the same accuracy as a
  divide).  In use for `atan`/`atanpi` (54.3 -> 48.5 ns/element), `log1p`
  (56.8 -> 45.8, two divides collapsed into one corrected quotient and one
  true divide kept for the subnormal `1+x` case) and `erfc`.  A reciprocal
  estimate of a subnormal yields infinity and raises flags the reference
  does not, so every site is checked for a bounded divisor.

* **The optimal LMUL is per function, and it is not the same on both
  boards.**  RVV arithmetic throughput scales linearly with LMUL on both
  cores, so LMUL only trades loop overhead against register pressure -
  and the VLEN 128 core runs out of registers first.  Measured on both
  boards: `vhyperinv.c` and `vexpm1.c` are faster at LMUL 1 (asinh 1.70x ->
  2.09x, atanh 2.38x -> 3.20x on VLEN 256; asinh 0.66x -> 1.17x on VLEN
  128), `vlog.c` and `verf.c` are faster at LMUL 2, and the inverse-trig
  family splits: `atan`/`atan2`/`atanpi`/`acos` want LMUL 1 (a210: 0.55x ->
  1.23x, 0.30x -> 0.71x) while `atan2pi` wants LMUL 2 (0.78x -> 0.31x at
  LMUL 1), which is why `atan2pi` now lives in its own translation unit.
  The choice is static and per translation unit - no runtime dispatch.

## Engineering surveys

`doc/literature-survey.md` and `doc/rvv-survey.md` are the surveys that
drove the changes above: CORE-MATH, SLEEF, ARM Optimized Routines, RLIBM,
double-word arithmetic papers, plus measured RVV instruction costs and
toolchain pitfalls, each claim tagged measured / measured-here / inferred.
The techniques they recommend and that are *not* yet implemented are listed
there with estimated gains (atan family table + tangent subtraction, erfc
odd-only asymptotics, cancellation-free lgamma around its zeros,
`FastFMA_DW`, native f32 kernels for the remaining widen/narrow entries).

## A second RVV board (a210, VLEN 128)

The same tree was built and measured on a second RVV 1.0 board: a210
(8x RVV 128-bit, glibc 2.41, native GCC 14.2). Accuracy: **82/82 PASS**
at the SMALL tier (2 059 records per function), no class/sweep/flag
failures -- same contract as the VLEN 256 board. Per-run reports are in
`test/results/board-a210/`.

The speedups are structurally smaller there, for two reasons: the vector
unit is half as wide (VLEN 128, so a strip covers half as many lanes)
while its scalar FPU is roughly twice as fast per element, so the scalar
baseline is much stronger. Measured mean is 3.44x with every entry above
scalar when built natively with GCC 14.2
(`test/results/board-a210/bench-a210-gcc14.md`).

The result is also sensitive to the compiler.  That same tree cross-built
with GCC 15.1 instead of the board's native GCC 14.2 measures mean 2.48x
with four entries under parity (`hypot` 0.67x, `lgamma` 0.72x,
`lgamma_r` 0.88x, `acospif` 0.95x), so per-entry RVV code generation still
moves by up to 2x between the two compilers on this core; the numbers
above and in `test/results/board-a210/` are for the board's native
compiler, which is also what ships.  What that sensitivity means for the
source: a call between RVV intrinsics inside the strip loop is the worst
of the patterns (GCC PR 121018 - it made `sinf` 2.16x -> 0.55x on this
board while being a win with GCC 15), explicit rounding-mode conversions
cost an `fsrmi`/`fsrm` pair per strip (the reason `lgamma_r`'s sign
output is derived from its sign bit), and any `vsetvli` type change in
the loop is a real instruction.  The binary32 trig fallbacks therefore
stay inlined and the fast paths stay in one SEW/LMUL shape.

The full 100057-record MPFR gate also passes on this board, not just the
SMALL tier: `test/results/board-a210/acc-a210-full.md` (82/82 PASS, no
class/sweep/flag failures).

## VFABI aliases

`vfabi.c` and `vfabi32.c` export the mangled entry points of the RISC-V
Vector Function ABI (`_ZGV<K>Nxv_<name>` / `_ZGV<K>Nxvv_<name>`, LMUL
1/2/4/8) for the 26 single-output functions at each precision —
`_ZGV<K>Nxv_sin` and `_ZGV<K>Nxv_sinf` and so on — so future compilers
and hand-written SIMD code can call them directly. Each wrapper calls the
same LMUL-generic kernel as the array API with `vl = VLMAX`, so results
are bitwise identical to `vsin()` and friends. `vsincos`, `vsincospi`,
`vlgamma_r` and their binary32 counterparts are not exported — the
mangling cannot express a two-vector result.

Because an `f32m8` operand group would widen into 16 binary64 register
groups, one past the architectural maximum, the LMUL 8 binary32 wrappers
split their operand into two LMUL 4 halves. The ABI defines `vl = VLMAX`,
so the split is exact and the results are unchanged.

## Known limitations

* **Large trigonometric arguments** (`|x| > 2^19·π/2 ≈ 8.2e5`) go through a
  per-lane scalar fallback (`__ieee754_rem_pio2`); results and flags are
  bit-identical to scalar libm and still ≤ 1 ulp, only throughput suffers.
  A vectorised Payne-Hanek reduction is a drop-in replacement for that
  epilogue. (The pi-scaled family never needs it: `rint(2x)` and
  `x - rint(2x)/2` are exact.) For binary32 the threshold is the same but
  the fallback is reached far more often, since `|x| > 8.2e5` is a much
  larger fraction of the type's range.
* **Non-V multilibs** get pure scalar fallbacks from `vfallback.c` and
  `vf_fallback.c` so that every configured `libvecmath.a` exports the
  complete symbol set; accuracy there simply inherits from scalar libm and
  is *not* covered by the ≤ 1 ulp contract.
* All 41 functions are implemented at both precisions. `lgamma_r` is the
  dual-output one: it writes the result and the `int` sign separately, and
  deliberately does not touch the `signgam` global. It is the one entry
  the VFABI mangling cannot express (a second vector in, an `int` vector
  out), so `vlgamma_r` has no `_ZGV` alias.
* **Most binary32 entries are native now, but not all**: exp, log, trig,
  invtrig, hyperbolic, erf and gamma have native binary32 kernels with a
  per-lane fallback; `powf`, `cbrtf`, `rsqrtf` and `sqrtf` still wrap the
  binary64 kernel (they are already 1.5-7x, so there was nothing to win).
  The kernel *interface* is element-generic (`include/f32/`), so
  converting the rest is local.
* **The gamma fast path is 1.27-1.54 ulp on its own band.** `lgamma`,
  `lgamma_r` and `tgamma` dispatch per strip: a strip whose lanes all lie
  in [0.5, 8) runs the compact fast path, anything else runs the full
  double-double kernel.  The FULL MPFR tier passes at <= 1 ulp, but
  `scripts/domain-check.sh` -- which filters the blobs to that band so
  only the fast path runs -- measures 1.27 ulp (tgamma), 1.41 (lgamma)
  and 1.54 (lgamma_r) on ~0.2% of the band's lanes.  The residual is the
  fdlibm term's evaluation plus the one rounding of the final sum: the
  result there is a difference of two moderate terms (about 2x relative
  error amplification), so the term needs double-double *polynomial
  evaluation*, not just a double-double combine.  Doing that was measured
  at +9 to +42 ns/element per entry, which costs the fast path more than
  the accuracy is worth in the shipped contract (the framework gate is
  the FULL tier), so it stays documented rather than paid for.
* **The `erfc` tail used to be the one entry under parity; it is 1.24x
  now.** Its 1.25 <= |x| band is a single serial dependency chain per
  strip (argument pair, exp pair, tail factor, dd multiply, dd divide),
  so three divisions sat on the critical path and each cost ~20 ns/element.
  One reciprocal now supplies all three: `(rt, rt_lo)` is corrected to
  ~2^-105 with one fma, `s = (rt + rt_lo)^2` is rebuilt with an fma
  residual folded into its low part so it still rounds within half an ulp
  (`rho` is sensitive -- six ulps of result per ulp of `s`, which cost a
  lane when the naive form was tried), and the final divide becomes a dd
  multiply by that reciprocal.  That is 181 -> 120 ns/element and the
  worst case improved to 0.8440 ulp.  Its mid band additionally uses a
  mixed-precision Horner (plain double down to coefficient 7, dd below,
  verified to 0.000 ulp against the exact polynomial), which took that
  band from 63 to ~20 ns/element.

## Building and testing

Configure newlib with `--enable-vecmath` (non-RISC-V targets are rejected),
then `make libvecmath.a && make install`. After changing any `Makefile.inc`
or `configure.ac`, re-run `autoreconf -f` in `newlib/` first.

The test framework is plain make, outside the automake build. It covers
both precisions; each function has a binary64 and a binary32 blob.
`scripts/build.sh TAG runner bench` builds a tagged pair of binaries
(parallel, per-tag object dirs so concurrent builds do not collide),
`scripts/acc.sh TAG --func NAME --tier full` runs the accuracy gate,
`scripts/bench-board.sh` runs the throughput benchmark on a real RVV
board (board-side lock, pinned core, best-of runs), and
`scripts/check-all.sh TAG` joins the two into a PASS/FAIL table.
`scripts/filter-blob.py` + `scripts/domain-check.sh` re-run the accuracy
framework on blobs filtered to a homogeneous domain, which is how
fast-path-only accuracy is verified.

```
cd newlib/libm/vecmath/test
make vmgen blobs-small runner-linux
~/qemu-build/qemu-riscv64 -cpu rv64,v=true,vlen=256,vext_spec=v1.0 \
    ./runner-linux --blobs blobs/SMALL [--func NAME]

make blobs-full                 # 100k records per function
make runner-elf-int && qemu-system-riscv64 …   # bare metal
sh scripts/run-board.sh muse-pi --tier full    # real hardware
```

`vmgen` (host, needs libmpfr) writes binary blobs of MPFR-derived test
records; the blob header records the element precision
(`vmb_hdr_t.reserved[0]`), and a binary32 record stores its little-endian
binary32 bit patterns in the low half of the same 48/72/80-byte record
layout. `runner` (target) exercises the vector entry points and checks
accuracy, special-value classes, per-lane exception flags, stripmine-length
and alignment consistency, and canaries. Note that `runner` prints with
`%lu` — newlib's printf has no `%zu`.

Both tiers are run over a VLEN matrix (qemu or spike) and, when hardware is
available, on a real RVV board; the resulting reports are kept under
`test/results/` and are the record of the accuracy figures quoted above.
