# libvecmath — RISC-V RVV vector math library

A double-precision vector math library for RISC-V RVV 1.0, built as part of
newlib. It is compiled **only** when newlib is configured with
`--enable-vecmath`, and it produces a separate `libvecmath.a` — nothing is
added to `libm.a`.

All functions are ≤ 1 ulp against an MPFR reference (the exact figure per
function is in the accuracy table below).

## API

```c
#include <vecmath.h>
void vsin (double *restrict y, const double *restrict x, size_t n);
```

Array-style, VLA: every entry processes `n` elements and stripmines over the
hardware VLEN internally, so one binary runs on any RVV 1.0 implementation
(VLEN 128 / 256 / 512 / 1024 / …). Link with:

```
-lvecmath -lm -lc
```

### Contract

* `y` (and any second output) must not overlap the inputs.
* Pointers may have any alignment; `n == 0` is legal.
* **`errno` is never set** (libmvec / VFABI semantics). Floating-point
  exception flags in `fcsr` *are* raised per lane exactly as the scalar C99
  functions would — except `FE_INEXACT`, which accumulates legitimately from
  the internal double-double arithmetic and is therefore not meaningful.
* Results are correctly rounded or within 1 ulp of the infinitely precise
  result. Signed zeros, infinities and NaN propagation follow IEEE-754 /
  C99 / SYCL.
* **Bare-metal users must enable the vector unit** (`sstatus.VS != Off`)
  before calling these functions; Linux user space does this automatically.
  The test program does it with `li t0, 0x2200; csrs sstatus, t0`.
* Weak symbols do not pull archive members: linking statically against
  `libvecmath.a` from a bare-metal image needs
  `-Wl,--whole-archive -lvecmath -Wl,--no-whole-archive`.

### Functions

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

## Accuracy

Measured by `test/runner` against MPFR, FULL tier (100 057 records per
function) at VLEN 256; identical results at VLEN 128/256/512/1024 and on
bare metal. The full per-run reports are checked in under
`test/results/`.

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

## VFABI aliases

`vfabi.c` exports the mangled entry points of the RISC-V Vector Function ABI
(`_ZGV<K>Nxv_<name>` / `_ZGV<K>Nxvv_<name>`, LMUL 1/2/4/8) for the 26
single-output functions, so future compilers and hand-written SIMD code can
call them directly. Each wrapper calls the same LMUL-generic kernel as the
array API with `vl = VLMAX`, so results are bitwise identical to `vsin()`
and friends. `vsincos` and `vsincospi` are not exported — the mangling
cannot express a two-vector result.

## Known limitations

* **Large trigonometric arguments** (`|x| > 2^19·π/2 ≈ 8.2e5`) go through a
  per-lane scalar fallback (`__ieee754_rem_pio2`); results and flags are
  bit-identical to scalar libm and still ≤ 1 ulp, only throughput suffers.
  A vectorised Payne-Hanek reduction is a drop-in replacement for that
  epilogue. (The pi-scaled family never needs it: `rint(2x)` and
  `x - rint(2x)/2` are exact.)
* **Non-V multilibs** get pure scalar fallbacks from `vfallback.c` so that
  every configured `libvecmath.a` exports the complete symbol set; accuracy
  there simply inherits from scalar libm and is *not* covered by the ≤ 1 ulp
  contract.
* All 40 functions are implemented.  `lgamma_r` is the dual-output one: it
  writes the `double` result and the `int` sign separately, and deliberately
  does not touch the `signgam` global.  It is the one entry the VFABI
  mangling cannot express (a second vector in, an `int` vector out), so
  `vlgamma_r` has no `_ZGV` alias.

## Building and testing

Configure newlib with `--enable-vecmath` (non-RISC-V targets are rejected),
then `make libvecmath.a && make install`. After changing any `Makefile.inc`
or `configure.ac`, re-run `autoreconf -f` in `newlib/` first.

The test framework is plain make, outside the automake build:

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
records; `runner` (target) exercises the vector entry points and checks
accuracy, special-value classes, per-lane exception flags, stripmine-length
and alignment consistency, and canaries. Note that `runner` prints with
`%lu` — newlib's printf has no `%zu`.

Both tiers are run over a VLEN matrix (qemu or spike) and, when hardware is
available, on a real RVV board; the resulting reports are kept under
`test/results/` and are the record of the accuracy figures quoted above.
