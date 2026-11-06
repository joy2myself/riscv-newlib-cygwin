# libvecmath — RISC-V RVV vector math library

A double- and single-precision vector math library for RISC-V RVV 1.0,
built as part of newlib. It is compiled **only** when newlib is
configured with `--enable-vecmath`, and it produces a separate
`libvecmath.a` — nothing is added to `libm.a`.

All 41 functions exist at both precisions, and every one is ≤ 1 ulp
against an MPFR reference (the exact figure per function is in the
accuracy table below).

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

Each binary32 kernel is a thin wrapper around the binary64 kernel of the
same name: the `f32mK` operand group is widened exactly to `f64m2K`,
evaluated by the binary64 kernel that the accuracy table below already
covers, and narrowed back with a single correctly-rounded conversion.
`vsqrt`'s binary32 counterpart is the exception — `vfsqrt.v` on binary32
is correctly rounded on its own, so `vsqrtf` is a native kernel.

This is the same computation newlib's own scalar single-precision
functions perform: the Arm optimized-routines ports behind `expf`,
`exp2f`, `logf`, `log2f`, `powf`, `sinf`, `cosf` and `sincosf` all
evaluate in `double_t` (= `double`) and round once at the end. Because
every binary32 value is a binary64 value, the widening is exact, and a
reference that is accurate to within a few binary64 ulps is accurate to
well under `2^-29` of a binary32 ulp. The binary32 entries are therefore
correctly rounded to within `0.5 ulp + 2^-29 ulp` — strictly inside the
contract — and inherit the binary64 special-case and exception-flag
semantics one for one, including the `FE_OVERFLOW` / `FE_UNDERFLOW` that
the final narrowing raises where the binary32 result's own rounding
requires it.

The cost of that choice is register pressure, not instruction count: an
`f32mK` group is `2K` binary64 register groups, so a binary32 kernel
occupies twice the vector registers of a hand-written binary32 kernel of
the same lane width. Latency and instruction count per lane are the same
as the binary64 kernel's.

Three entries — `vexpm1f`, `vlog1pf` and `vpowf` — add one masked
operation on top of that shape. Their binary64 kernels short-circuit the
tiny region and return the input (or, for `pow`, an exactly subnormal
result) verbatim; for a binary32 argument that is exact even where the
binary32 operation itself would have had to round a correction onto the
much coarser binary32 subnormal grid, so the narrowing conversion stays
silent. Those entries therefore raise `FE_UNDERFLOW` explicitly on the
lanes whose result is a nonzero subnormal, using the same masked
multiply-and-add device the binary64 `log2p1` kernel already carries.

## Accuracy

Measured by `test/runner` against MPFR, FULL tier (100 057 records per
function) at VLEN 256; identical results at VLEN 128/256/512/1024 and on
bare metal. The full per-run reports are checked in under
`test/results/`, including a real-hardware run at FULL tier on a
VLEN 256 RVV 1.0 board (`test/results/board-muse-pi/`), built natively
with the board's own GCC 14 and byte-for-byte identical to the VLEN 256
qemu report.

| function | max ulp | | function | max ulp |
|---|---|---|---|---|
| sqrt | 0.500 | | asin | 0.913 |
| rsqrt | 0.861 | | acos | 0.971 |
| cbrt | 0.665 | | atan | 0.834 |
| exp | 0.503 | | atan2 | 1.000\* |
| exp2 | 0.504 | | asinpi | 1.000\* |
| exp10 | 0.503 | | acospi | 0.811 |
| expm1 | 0.813 | | atanpi | 1.000\* |
| log | 0.502 | | atan2pi | 1.000\* |
| log2 | 0.536 | | sinh | 0.517 |
| log10 | 0.909 | | cosh | 0.501 |
| log1p | 0.617 | | tanh | 0.517 |
| log2p1 | 1.000\* | | asinh | 0.908 |
| pow | 0.505 | | acosh | 0.764 |
| sin | 0.785 | | atanh | 0.895 |
| cos | 0.785 | | hypot | 1.000\* |
| tan | 0.738 | | erf | 1.000\* |
| sincos | 0.835 | | erfc | 0.501 |
| sinpi | 1.000\* | | tgamma | 1.000\* |
| cospi | 0.807 | | lgamma | 0.851 |
| tanpi | 1.000\* | | lgamma_r | 0.562 |
| sincospi | 1.000\* | | | |

\* The maximum is exactly 1.0 ulp, which the ≤ 1 ulp contract admits; no
function has a lane over 1 ulp. On the starred entries the last rounding
lands on a coarser grid than the one the arithmetic was carried out in —
in most cases the subnormal grid — so it can cost a whole grid step:

* `log2p1`, `sinpi`/`tanpi`/`sincospi`/`hypot`/`atanpi`/`atan2pi`, and
  `asinpi`/`atan2` on their subnormal lanes, round twice — once at 53-bit
  precision in the scaled domain, then onto the coarser subnormal grid;
* `tgamma` on the lanes where `lgamma` has already overflowed, where the
  result is `exp` of a very large value and the collapse is the only
  rounding;
* `erf` for `|x| <= 1.97e-308`, where the double-double product's second
  coefficient is worth 0.0345 of a grid step (the grid at 1e-308 is 2^-1074
  and the term is ~1.7e-325). Dropping it costs a step whenever the leading
  product sits within that fraction of a rounding midpoint, so about 3% of
  subnormal x. A `_mu`-masked multiply would recover it.

### binary32

The 41 `v<name>f` entries, measured over the same 100 057 records:

| function | max ulp | | function | max ulp |
|---|---|---|---|---|
| acoshf | 0.4999 | | every other binary32 entry | 0.5000 |

Every binary32 entry is within 0.5 ulp — half the contract — and no lane
in any of them exceeds 0.5000. That is the expected consequence of
evaluating in binary64: the binary64 kernel's own error is at most a few
binary64 ulps, which is under 2^-29 of a binary32 ulp, so the total is
0.5 ulp + 2^-29 and the harness reports 0.5000. Unlike the binary64 table
above there are no entries that round twice onto a coarser grid, so no
binary32 function reaches 1 ulp.

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
* The binary32 kernels are wrappers around the binary64 ones rather than
  hand-written binary32 kernels; rewriting the hot ones (exp, log, sin,
  cos, pow) natively would halve their register pressure. The kernels'
  *interface* is already element-generic (`include/f32/`), so such a
  rewrite is local.
* There is **no installed binary32 counterpart of `<vecmath_rvv.h>`**: the
  binary32 kernel layer lives in `libm/vecmath/include/f32/` and is not
  installed, because an `f32mK` group widens into `2K` binary64 register
  groups and LMUL 8 would need `f64m16`, one past the architectural
  maximum. Hand-written SIMD code that wants a binary32 entry point uses
  the exported VFABI alias `_ZGV<K>Nxv_<name>f` (LMUL 1/2/4/8) instead,
  which is bitwise identical to the array API. The binary64 inline-kernel
  header is unaffected.
* The **binary64** `vexpm1` and `vlog1p` return the input verbatim for
  subnormal inputs and so do not raise `FE_UNDERFLOW`, where scalar
  `expm1`/`log1p` do. The 200-bit MPFR reference cannot resolve the
  subnormal correction, so the framework does not currently detect it;
  the binary32 entries do raise the flag, because at binary32 the
  correction lands on the result grid.
* **`errno` is set after all**, on the rare lanes that go through the
  scalar fixup (`|x| >= 512` for exp, `x <= -1` for log1p, and so on):
  the fixup calls the scalar `exp`/`log`/`pow`, and newlib's Arm ports
  raise `ERANGE` through `math_err*.c` (`WANT_ERRNO` is 1 unless
  `_IEEE_LIBM`, which nothing defines for RISC-V). This contradicts the
  contract above, and applies equally to both precisions — the binary32
  entries inherit it because their guard lanes run the binary64 kernels,
  whose fixups are the ones that call the scalar functions. Verified on
  bare metal (`vexp(1000.0)` and `vexpf(1000.0f)` both leave
  `errno == ERANGE`). Clearing `errno` around the fixup loop in
  `k_fixup.inc` would make the contract hold for both layers.

## Building and testing

Configure newlib with `--enable-vecmath` (non-RISC-V targets are rejected),
then `make libvecmath.a && make install`. After changing any `Makefile.inc`
or `configure.ac`, re-run `autoreconf -f` in `newlib/` first.

The test framework is plain make, outside the automake build. It covers
both precisions; each function has a binary64 and a binary32 blob.

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
