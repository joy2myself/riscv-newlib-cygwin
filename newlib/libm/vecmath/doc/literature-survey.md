# Literature & OSS survey: vectorized floating-point function evaluation

**Target:** `libvecmath` (newlib, RISC-V RVV 1.0, LMUL-templated `.inc` kernels, VLA strip-mining,
MPFR gate at max_ulp ≤ 1.0, must beat scalar libm on muse-pi/VLEN 256 and a210/VLEN 128).
**Date:** 2026-11-11. **Author:** research subagent. **Status:** survey + prioritised recommendations.

## How to read the evidence flags

Every non-trivial claim below carries one of:

* **[M]** — *measured or explicitly claimed by the cited source* (paper, source comment, or the
  source's own benchmark numbers). I quote the source's number, not mine.
* **[R]** — *measured in this repository* (README.md tables, `test/results/`), i.e. our own data.
* **[I]** — *my inference*: an estimate or a projection obtained by reading code and counting
  operations. Not measured anywhere. These are the ones that need a build+bench before we
  believe them.

I did not modify any source file. This document is the only artefact.

---

## 0. Bottom line

Six findings dominate everything else in this survey.

1. **The 2024–2026 state of the art moved from "double-double everywhere" to "cheap table +
   exact residual + short plain-double polynomial + a rounding-interval test (Ziv), with a
   double-double fallback for the rare lanes".** CORE-MATH's binary64 `log` gets **0.5 ulp** out
   of a *plain double* degree-6 polynomial plus one FMA that is *provably exact*, and it fails its
   own rounding test with probability ≈ 2^−11.5 **[M]** (source comment). Same shape in `atan2`
   (3-term correction polynomial), `atan` (4-term), `cospi` (Ziv test), `cbrt` (rounding test with
   failure probability ≈ 1e−7) **[M]**. This is exactly the pattern we can afford on RVV: no
   gather-heavy DD, no long dependency chain, and the fallback machinery we already have
   (`vfirst`-guarded epilogue).
2. **Our thinnest entries (`atan2`, `atan2pi`, `atanpi`, and the atan-derived family) are paying
   for a table-free fdlibm reduction with an 11-term polynomial and 3 divisions.** CORE-MATH's
   `atan2` uses a **128-entry table of atan(j/128) as double-double pairs**, reduces with the
   tangent-subtraction formula `z = (y − t₀x)/(x + t₀y)` (2 FMA + 1 divide) and needs only a
   **3-coefficient** correction polynomial because `|z| ≤ 1/256` **[M]** (source). SLEEF instead
   uses a 20-term polynomial and no table **[M]** (paper §5.4). At VLEN 128 the table route is the
   only one that shortens the chain enough; the table is only read once per lane.
3. **The gamma fast-path cancellation is a solved problem in CORE-MATH's binary64 `lgamma`
   (integrated into glibc 2.43).** It never forms `−log(x) + fdlibm_term(x)`: it expands around
   `1.5`, `2.5`, `3.5` where `|u| ≤ 0.25`, evaluates the polynomial in **double-double with only
   the low-degree coefficients in DD**, and adds the `log`/`log1p` terms as DD **[M]** (source:
   `c0[]` for `(lgamma(x)+ln|x|)/x` on ±0.25, `b[]` for `lgamma(1.5+x)` on ±0.25). Our *accurate*
   kernel already has the cancellation-free `G(z)` form; our *fast* path does not.
4. **Multi-precision cost can be cut ~18–20% by one new primitive**: `FastFMA_DW` (Algorithm 4 of
   Jeannerod–Joldes–Louvet–Muller, ARITH 2026) evaluates a double-word FMA in 6 flops instead of
   8–9, and CORE-MATH measured the end-to-end accurate `exp` path going **68.7 → 55.7 cycles**
   with it on an EPYC 7282 + GCC 15.3 **[M]** (comment in `exp.c`). SLEEF independently reports
   that at 1-ulp accuracy it **skips DD renormalisation** and allows overlap between the two
   words **[M]** (paper §5). Both are drop-in changes to `vecmath_dd.h`.
5. **At VLEN 128 the losers are latency-bound, not op-count-bound.** Our own data shows the
   under-parity entries are exactly the long-dependency-chain kernels and notes "half the lanes
   to hide the chain behind" **[R]**. The literature's answer is ILP: Estrin evaluation (SLEEF:
   "reduce the length of the critical path") **[M]**, 2–4 independent FMA chains, and larger LMUL
   to amortise issue overhead (RISC-V V spec: "larger LMUL … reduce instruction fetch and
   dispatch overheads") **[M]**. A 2-way interleave of the strip loop is VLA-safe, changes no
   arithmetic, and is the cheapest a210 win available.
6. **Do not chase gathers we cannot afford.** SLEEF's measured position is that "a table-lookup
   is expensive … substantially slower than polynomial evaluation without any table lookup"
   **[M]** (paper §5). CORE-MATH uses 363-entry / 128-entry tables because scalar cores have cheap
   L1 loads. On RVV, a 128-entry DD table is 2 loads/lane (`vluxei64`) or a multi-register
   `vrgather`; the right compromise is **16–32 entries in registers via `vrgather`** with a
   6–8-term polynomial, or 128 entries read with `vluxei64` if the core's gather is fast
   (measure first) **[I]**.

---

## 1. Where we currently stand (for reference)

From `README.md` **[R]**:

* mean 3.08× on muse-pi (VLEN 256); thinnest `atan2pi` 1.11×, `erfc` 1.24×, `atan2` 1.36×,
  `log2p1` 1.38×, `lgamma` 1.39×.
* a210 (VLEN 128): 13 entries ≤ 1.0× — `atan2` 0.31×, `atanpi` 0.52×, `atan2f` 0.62×,
  `asinh` 0.66×, `expm1` 0.66×, `erfc` 0.71×, `cbrtf` 0.73×, `erfcf` 0.76×, `atan2pi` 0.78×,
  `cospif` 0.82×, `log2` 0.91×, `log2p1` 1.00×.
* gamma fast path measured at 1.27–1.54 ulp on its own band (`tgamma` 1.27, `lgamma` 1.41,
  `lgamma_r` 1.54); the full gate passes because the band is a small fraction of the FULL tier.
* our `atan`/`atan2` kernel is "a plain-double port of fdlibm `s_atan.c` / `e_atan2.c` … the final
  composition carried as a double-double pair", with **2 divisions** for `atan` (reduction +
  quotient residual) and a third for callers that pass a tail — i.e. `atan2` pays **3 divisions
  and an 11-term polynomial** **[R]** (`k_invtrig.inc` header, README "atan/atanpi got their
  quotient from two divisions on the critical path").
* our `gamma` fast path is a branch-free port of fdlibm `e_lgamma_r.c` with its six coefficient
  tables (`vm_gam_a/t/u/v/s/r`), i.e. the classic `−log(x) + term(x)` reconstruction **[R]**
  (`k_gamma.inc:766–928`).
* our `log` is a vectorisation of ARM ORT `log.c`: N=128 table + polynomial of order 6/12, plus a
  near-1 special polynomial. The header carries **two** forms: a "lean plain-double log (Arm `log.c`
  main path + near-1 path, **no dd**)" and a `VM_LOG_DD` form that is "kept for log10 and for
  [callers needing the dd tap]" **[R]** (`k_log.inc:133–137`). The shipped 0.729 ulp therefore
  comes from the DD form; the lean path alone is an ARM-grade ≈1 ulp.

---

## 2. Techniques

### T1. Table + *exact* FMA residual (the "10-bit reciprocal" trick)

**What.** Choose the table entry so that the reduction residual is **not merely accurate but
exactly representable**, then evaluate a short polynomial in that residual. CORE-MATH `log`
stores `r[i]`, a **10-bit** approximation of `1/x[i]` on `i·2⁻⁹ ≤ x < (i+1)·2⁻⁹`, chosen such that
`r[i]·y − 1` is exactly representable in 53 bits for *every* `y` in the interval, with
`|r[i]·y − 1| ≤ 0.00212097167968735` **[M]** (source comment). Why exact: `r` has 10 significant
bits and `y` has 53, so the product's lowest nonzero bit is at weight `2⁻⁶²`; subtracting 1 leaves
a value of magnitude `< 2⁻⁸·⁹`, hence ≤ 53 significant bits. The FMA `fma(r, y, −1)` therefore
rounds *nothing*.

**Math.**
```
z  = fma(r[i], y, -1)          // EXACT, |z| <= 0.002121
log(y) = log(x[i]) + log1p(z)  // log(x[i]) from a DD table (l1,l2)
       = l1 + [ z + z^2/2 - ... ] + l2
```
The polynomial for `log1p(z)` is degree 6 with absolute error `< 2⁻⁷⁰·²⁷⁸` **[M]**; the DD output
has absolute error `≤ 2⁻⁶⁸·²²` **[M]**; the caller certifies correct rounding with
`err = 0x1.b6p-69` and `h+(l−err) == h+(l+err)`, failing with probability ≈ `2⁻¹¹·⁵` **[M]**.

**Op count.** Reduction: 1 mul (scale by 1 or ½) + 1 FMA + 1 shift/and. Polynomial: 6 FMA (Estrin,
3 levels). Combine: 1 FMA (`e·log2_h + l1`), 1 `fast_two_sum` (3 ops), 1 add, 1 FMA (`e·log2_l`).
≈ **12–13 FP ops + 3 loads** **[I]**. Our shipped `VM_LOG_DD` form pays the same table read plus a
DD combine (≈ 20–22 ops, the price of the 0.729 ulp), and our lean no-DD path is ≈ 12 ops but
ARM-grade (~1 ulp) **[I]** from the header's own description. So T1's win is: **keep lean-path
speed and get the DD path's accuracy**, by making the residual exact instead of adding precision
after the fact.

**Accuracy impact.** 0.5 ulp guaranteed by the interval test, not by the polynomial. The DD table
entry is what carries the 53-bit-accurate `log(x[i])`; the polynomial only has to be right to
`2⁻⁷⁰` over a `±0.0021` interval, which is why it is so short.

**Mapping to our kernel style.**
* Table: `static const struct { uint16_t inv; double l1, l2; } vm_log_tab[512]` indexed by the
  top 9 mantissa bits (after the [√½,√2) normalisation we already do). The `inv` field is a
  `uint16_t` holding the 10-bit reciprocal — *it must be generated by a script that verifies the
  exactness property*, not by `1.0/x`.
* Intrinsics: `VM_VAND/VSRL` for the index, `VM_VLWXEI64`/`vrgather` for the 3 fields,
  `VM_VFMACC` for `z = fma(r,y,−1)`, then `VM_VFMSAC`/`VM_VFMACC` chain, then `VM_VFADD`+`VM_VFSUB`
  for `fast_two_sum` (we already have `vecmath_dd.h` primitives).
* Keep our current path for `log10`/`log2` by **pre-scaling the table's `l1,l2` and the `e·log2`
  constants per function** (two table copies) rather than a final multiply — this removes the
  extra rounding that currently forces the DD form **[I]**.
* The Ziv test is 3 ops (`vfrsub`/`vfadd` twice + `vmseq`), and the failing lanes go through the
  existing scalar/dd fixup behind `vfirst`.

**Citations.** CORE-MATH `log.c` (table comment, `err` bound, failure probability)
<https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/log/log.c>; ancestor of
the technique: de Dinechin, Lauter, Muller, *Fast and correctly rounded logarithms in double
precision*, RAIRO-ITA 41(1) 2007 <https://hal.science/ensl-00000007>; Ziv's test and the
"accurate path" idea: CORE-MATH paper §I-A **[M]**
<https://inria.hal.science/hal-03721525>.

---

### T2. Two-level table for `exp` + `FastFMA_DW`

**What.** CORE-MATH `exp` builds a **12-bit** table out of two **64-entry** DD tables (`t0[i] =
2^(i/64)`, `t1[j] = 2^(j/4096)`) so the reduced argument satisfies `|r| < log2/2¹³` **[M]**. That
tiny `r` is why the polynomial is only **degree 4** (`exp(r)−1` with 4 DD coefficients) **[M]**.
The product `t0[i]·t1[j]` is one `muldd`, and the reduction is a 3-term Cody–Waite with `l2h`
chosen to be exactly representable on 29 bits so that `l2h·t` is exact **[M]**.

**The new primitive.** `FastFMA_DW(ah,al,bh,bl,ch,cl,&l)` implements Algorithm 4 of
Jeannerod–Joldes–Louvet–Muller, *Extended-Precision FMA under Parameterized Double-Word Overlap:
Tight Error Bounds and Examples*, ARITH 2026 **[M]**
<https://inria.hal.science/hal-05517451>:
```c
dh = fma(ah, bh, ch);  t = ch - dh;  e = fma(ah, bh, t);
f  = e + cl;           g = ah*bl + f;      *l = al*bh + g;   return dh;
```
6 flops (7 with the `cl` add). CORE-MATH measured the whole accurate `exp` path at
**68.7 → 55.7 cycles** with it (and 129.6 → 130.6 without hardware FMA) **[M]**.

**Op count.** Our `exp` is 0.503 ulp at 4.0× **[R]**; if it is already table+poly this is a
marginal change: +2 loads, −3 polynomial terms, and −2 flops per DD op in the accurate tail
**[I]**. The bigger win is applying `FastFMA_DW` to *every* DD Horner in the library
(`k_log`, `k_exp`, `k_gamma`, `k_erf`, `k_invtrig`): each `muldd`+`adddd` step drops from ~12 to
~9 flops **[I]**.

**Accuracy.** The paper's bounds are *tight and worst-case*, with matching counterexamples, in the
"dominance regime" `|c| ≥ 2|ab|` that holds in polynomial evaluation **[M]**. No accuracy risk if
we adopt the paper's algorithm verbatim (it is the one CORE-MATH ships).

**Mapping.** Replace `VM_DD_MUL`/`VM_DD_FMA` in `vecmath_dd.h` with the 6-flop form; keep the old
ones for the non-FMA fallback. This is a ~30-line change with a library-wide effect.

**Citations.** `exp.c` comment + code
<https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/exp/exp.c>;
ARITH 2026 paper (above).

---

### T3. Plain-double fast path + rigorous interval ("Ziv") test, DD only on failure

**What.** Compute the result `f` in plain double, compute a *rigorous* absolute error bound `e`
for that evaluation, then form `ub = (f+e)+base`, `lb = (f−e)+base`; if `ub == lb`, the exact
value lies strictly inside one rounding interval and `ub` **is** the correctly rounded result.
Otherwise fall back. CORE-MATH uses this in `atan`, `atan2`, `cospi`, `sinpi`, `log`, `cbrt`,
`erf`/`erfc` **[M]** (source).

**Math (atan, 2⁻⁸ ≤ |x| ≤ 2⁷) [M]:**
```
i  = index from the top bits of x           // 129-entry table, tan(pi*i/256)
h  = (x - ta)/(1 + x*ta)                    // ta = A[i][0], plain double, 1 divide
f  = (ch0 + h2*ch1) + h4*(ch2 + h2*ch3)     // 4 coefficients, degree 7
al = fma(h, f, A[i][1] + pi/256 correction)
e  = h * 0x3.fp-52                          // 3.9375 * 2^-52 * h
if ((al+e)+ah == (al-e)+ah) return (al+e)+ah;
return as_atan_refine2(x, ...);             // DD path, ~10% of the code, rarely taken
```
Note what is *not* there: no DD quotient, no DD polynomial, no second division. The table supplies
`atan(ta)` to 106 bits, so only the small residual needs to be accurate.

**Op count.** atan: 1 divide + ~12 FP ops + 1 load, versus our fdlibm port's 2 divides + 11-term
poly + DD composition **[I]** (~25–30 ops). atan2: 2 divides (one for the index/quotient, one for
`z`) + 3-term poly **[M]**, versus our 3 divides **[R]**.

**Accuracy.** Exactly 0.5 ulp when the test passes; the failure rate is the only question. CORE-MATH
documents 2^−11.5 for `log` **[M]** and ~1e−7 for `cbrt` **[M]**. For `atan` the bound is chosen
with a 2.5% margin over 1.1e10 random pairs (atan2) **[M]**. On a vector strip of 8 lanes, a
per-lane failure probability of 2⁻¹¹ gives a per-strip probability of ~0.4%, so the `vfirst` guard
is taken rarely.

**Mapping.** This is *the* pattern our kernel style was built for: we already compute a mask and
branch to an out-of-line scalar fixup with `vfirst`. Generalise it:
```c
vm_f_t ub = VM_VFADD_VV(VM_VFADD_VV(f, e, vl), base, vl);
vm_f_t lb = VM_VFSUB_VV(VM_VFSUB_VV(f, e, vl), base, vl);
vm_b_t bad = VM_VMNE_VV(ub, lb, vl);       /* 1 compare */
if (VM_VFIRST_M(bad, vl) >= 0) { /* existing dd/scalar path, merged by mask */ }
```
Cost: 4 adds + 1 compare per lane. Saving: the whole DD tail in the common case.

**Citations.** `atan.c` <https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/atan/atan.c>;
`atan2.c` <https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/atan2/atan2.c>;
`log.c` (failure probability) <https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/log/log.c>;
`cospi.c` (same test with `er = |z|*2^-123 + 2^-78`)
<https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/cospi/cospi.c>.

---

### T4. Fine atan table + tangent-subtraction reduction (the atan2 fix)

**What.** `atan2`'s hard part is the quotient. CORE-MATH sidesteps it: take
`x = max(|x|,|y|)`, `y = min(|x|,|y|)` so the ratio is in [0,1]; pick `t₀ = j/128` from the top
7 bits of `y/x`; then
```
z = (y − t₀·x) / (x + t₀·y)        // tan(θ − atan t₀), 2 FMA + 1 divide
atan(y/x) = atan(t₀) + z·(1 + z²·(b₀ + z²·(b₁ + z²·b₂)))   // 3 coefficients
```
with `atan(t₀)` from the 128-entry DD table and the quadrant offset `O[kw]` (0, ±π/2, ±π) added in
DD **[M]** (source). Because `|z| ≤ 1/256`, `z⁷/7 < 2⁻⁵⁹` — only three terms are needed.

**Op count.** 2 divides (index and `z`), 2 FMA (`zn`, `zd`), 1 divide for `z`, 1 mul (`z²`),
3 FMA, 2 mul, 4 add, 1 gather of a DD pair. ≈ **14 ops + 2 divides + 2 loads**, versus our
3 divides + 11-term polynomial + DD composition **[I]** (≈ 30 ops + 3 divides).

**Accuracy.** 0.5 ulp with a measured bound (1.1e10 random pairs, +2.5% margin) **[M]**. The
table's DD `atan(t₀)` carries the absolute accuracy; the polynomial only corrects `|z| ≤ 2⁻⁸`.

**Mapping.**
* For RVV, prefer a **16- or 32-entry** in-register table (`vrgather`) instead of 128 entries:
  step 1/8 gives `|z| ≤ 1/16`, needing ~5 terms in `z²`; step 1/16 gives `|z| ≤ 1/32`, ~4 terms
  **[I]**. Compare against `vluxei64` on the actual core before committing.
* Reuse the *same* kernel for `atan`, `atanpi`, `atan2`, `atan2pi`, `asin`, `acos`, `asinpi`,
  `acospi`, `atanh`, `asinh`, `acosh` (all are `atan`- or `log`-shaped) — this is a family-wide
  rewrite, not a per-function hack.
* Avoid the index division by using `vfrec7` + one Newton step for the index only (the index
  needs 7 bits, not 53) **[I]**.

**Citations.** `atan2.c` (table `f2`, `T2`, `O`, bound comment)
<https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/atan2/atan2.c>; `atan.c`
(129-entry `A`, piecewise-quadratic index, Ziv test)
<https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/atan/atan.c>. For the
no-table alternative: SLEEF's 20-term `atan` polynomial **[M]** (paper §5.4,
<https://arxiv.org/abs/2001.09258>), and ARM ORT's 20-term AdvSIMD `atan2` (1.97 ulp measured)
<https://github.com/ARM-software/optimized-routines/blob/master/math/aarch64/advsimd/atan2.c>.

---

### T5. Odd-only asymptotic polynomial for `erfc` + a specialised `exp`

**What.** For the tail, CORE-MATH evaluates `erfc(x) = exp(−x²)·p(1/x)` where `p` approximates
`erfc(1/x)·exp(1/x²)` and is a **degree-23 polynomial with only odd coefficients** (12 stored
constants, `p[0]+p[1]` a DD pair for degree 1, the rest plain double), fitted separately on six
interval tables `T[6][13]` **[M]**. The reciprocal is a DD pair `(yh, yl)` with
`yl = fma(−x, yh, 1)·yh`, and the exponential is a purpose-built `exp_1` with relative error
`< 2⁻⁷⁴·¹³⁹` — **not** a correctly rounded `exp` **[M]**. The final product is a DD multiply.

**Why this is cheap.** (a) Only odd powers ⇒ half the multiplies of a general polynomial.
(b) The accuracy target is **absolute/relative to a tiny result**, so `exp` needs ~2⁻⁷⁴, not
0.5 ulp — that is a 4–5 term polynomial over a small reduced range, not a full `exp` kernel
**[M]**. (c) No division other than the one reciprocal. CORE-MATH reports **111 cycles average**
for `x ∈ [THRESH1, 5]` on an i7-8700 with GCC 12.2, of which only 20 cycles are `exp_1` **[M]**.

**Op count.** ≈ 1 divide + 1 FMA (residual) + 6 FMA (odd poly) + `exp_1` (≈ 10–12 ops) + 4 ops
(DD multiply) ≈ **25 ops, 1 divide** versus our tail's "argument pair, exp pair, tail factor, DD
multiply, DD divide" with one reciprocal **[R]** — comparable op count but *much shorter critical
path* (no serial DD divide at the end) **[I]**.

**Accuracy.** `p` relative error `< 2⁻⁷¹·⁸` (worst table) and `exp_1` `< 2⁻⁷⁴·¹` combine to
`< 2⁻⁶⁷·²` overall, plus a rounding-error term `< 2⁻⁶⁷·²` **[M]** — comfortably below the 0.5 ulp
target. Our current erfc already measures 0.844 ulp **[R]**, so this is a *speed* technique with
accuracy headroom to spare.

**Mapping.** `k_erf.inc`: replace the `rho` tail-factor branch with the odd polynomial; keep our
existing single-reciprocal trick (it is the same `(yh, yl)` construction CORE-MATH uses). The
polynomial coefficients are per-band (`T[6]`), so the band select must be branch-free
(`vmerge` on 6 coefficient sets, or 3 if we coarsen the bands). `exp_1` can be a stripped copy of
our `k_exp.inc` fast path with the Ziv test deleted and the polynomial truncated to the 2⁻⁷⁴ term.

**Citations.** `erfc.c` (references, `p` comment, `exp_1`, error analysis, cycle count)
<https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/erfc/erfc.c>; the
classical approximation: Shepherd & Laframboise, *Chebyshev approximation of (1+2x)exp(x²)erfc x*,
Math. Comp. 36 (1981) 249–253 <https://doi.org/10.1090/S0025-5718-1981-0595059-5>; error bounds:
Chevillard, *The functions erf and erfc computed with arbitrary precision and explicit error
bounds* <https://inria.hal.science/ensl-00356709>.

---

### T6. Cancellation-free `lgamma` around its zeros

**What.** `lgamma` has zeros at `x = 1` and `x = 2`; any formula that computes it as a difference
of `O(1)` terms loses relative accuracy proportional to the cancellation. CORE-MATH's binary64
`lgamma` (2025, integrated into glibc 2.43 **[M]**) uses four devices **[M]** (source):

1. **Near 0** (`|x| ≤ 0.25`): `lgamma(x) = −log|x| − x·P(x)` where `P` approximates
   `−(lgamma(x)+ln|x|)/x` (the function is *entire*, `P(0) = γ`). No cancellation: the two terms
   are both `O(1)` and the result is `O(1)`.
2. **Around 1.5**: a DD polynomial `b[]` for `lgamma(1.5+u)`, `|u| ≤ 0.25`; then for
   `x ∈ [0.25,0.75]` use `lgamma(x) = b(x−0.5) − log(x)` (because `lgamma(x+1) = lgamma(x)+log x`).
3. **Around 2.5** (`|x−2.5| ≤ 0.25`): `lgamma(x) = b(x−2.5) + log(x−1)`.
4. **Around 3.5**: `lgamma(x) = b(x−3.5) + log(x−1) + log(x−2)`.

Every `log` is a **three-word DD log** (`as_logd_accurate` returns `h,l,ll`) and every combine is
`sumdd`/`twosum`, so the only rounding is the final one. The polynomial coefficients are DD for
the **low degrees only** and plain double above (`polydddfst`), which is what keeps it affordable.

**Why our 2× amplification is unavoidable in the current formulation.** For `x = 1+u`,
`lgamma(x) = −log1p(u) + u·C₁ + u²G(u)`: the first two terms are `≈ −u` and `≈ 0.423u`, the result
is `≈ −0.577u`. Even if each term were *correctly rounded*, the errors propagate as
`1.7 × 0.5 ulp` plus the final rounding ⇒ **≥ 1.2 ulp** **[I]**. No amount of extra precision in
the *polynomial* fixes that; the *sum* must be formed in DD, or the expansion must be centred where
there is no cancellation (the scheme above).

**Concrete plan (surgical, low risk).**
* Keep the existing fast path only for `x ∈ [2,8)` (lgamma ≥ 0.1 there; the fdlibm form is fine).
* Add a third dispatch class, `x ∈ [0.5,2)`, that runs a *zero-aware* kernel: `u = x−1`,
  `lgamma(x) = lgamma(1+u)` computed as `u·Q(u)` with `Q` a mixed-DD polynomial (DD for the low
  4–6 coefficients, plain double above), on `|u| < 1` split at `u = ±0.5` (two coefficient sets)
  **[I]**.
* This is exactly the mixed-precision Horner our own `erfc` mid-band already uses ("plain double
  down to coefficient 7, dd below, verified to 0.000 ulp against the exact polynomial") **[R]** —
  so we know the technique and its cost in this codebase.
* Alternative with zero new coefficients: route the `[0.5,2)` class to the existing accurate
  `G(z)` kernel and pay the measured +9…42 ns/element **[R]**. The mixed-DD variant should cost
  much less because only 4–6 of the ~24 coefficients are DD **[I]**.

**Expected effect.** 1.41 ulp → ≤ 0.6 ulp on the band; cost +6…12 vector ops/lane **[I]**.
`tgamma` uses the same `lgamma` and needs only `2⁻⁵³` *absolute* accuracy in the log (since
`d(exp)/exp = d(lgamma)`), so it can use the same path unchanged.

**Citations.** `lgamma.c` (`c0[]`, `b[]`, `as_logd_accurate`, `polydddfst`)
<https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/lgamma/lgamma.c>;
`tgamma.c` (`1/x − γ` near 0, DD `polydd` with DD coefficients)
<https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/tgamma/tgamma.c>; glibc 2.43
integration: CORE-MATH project page <https://core-math.gitlabpages.inria.fr/> **[M]**. The
fdlibm baseline we currently use: <https://netlib.org/fdlibm/e_lgamma_r.c>. SLEEF's `gammak` also
uses the `(x−0.5)log x − x + ½log2π + z(t)` form with DD and claims 1 ulp, not 0.5 **[M]**
<https://github.com/shibatch/sleef/blob/master/src/libm/sleefsimddp.c> — consistent with that form
being unable to reach 0.5 ulp near the zeros.

---

### T7. Higher-order Newton + rounding test for `cbrt` / `rsqrt`

**What.** CORE-MATH's `cbrt` (used as the worked example in the ARITH 2022 paper) **[M]**:

1. reduce `a` to `[1,2)` exactly (or `[1,8)`);
2. initial **third-order** minimax polynomial, relative error `< 3·10⁻⁴`;
3. one **third-order** Newton step (`x₁ = x₀(1 − h/3 + 2h²/9 − 14h³/81 …)`), relative error
   `< 6·10⁻¹²`;
4. one **second-order** Newton step in double-double, relative error `< 1.32·10⁻²³`;
5. `Fast2Sum` to get `(x_high, x_low)`;
6. **rounding test**: `| |x_low| − 2⁻⁵³ | > 2⁻⁷⁵` ⇒ correctly rounded; failure probability
   ≈ 10⁻⁷; otherwise one more Newton step plus a small hard-case database;
7. exact-case detection for directed modes: `x_high` with 35 trailing zero bits.

The generalised Newton series is
```
x_{i+1} = x_i (1 − h/3 + 2h²/9 − 14h³/81 + 35h⁴/243 − 91h⁵/729 + 728h⁶/6561 − …),  h = (x_i³−a)/a
```
with "each additional term multiply[ing] the accuracy by one order" **[M]**.

**Op count.** 3rd-order poly init (≈ 4 FMA) + 1 third-order Newton (1 reciprocal + ~6 ops) +
1 second-order DD Newton (~10 ops) + Fast2Sum (3) + test (3) ≈ **26 ops**, versus a classic
3-iteration Newton with 3 divisions **[I]**.

**Accuracy.** 0.5 ulp (exact rounding) certified by the test; the hard-case table covers the
residual **[M]**.

**Mapping.** Our `cbrt` is 0.6645 ulp / 3.55× **[R]** — already good. The wins are
(a) `cbrtf` at 0.73× on a210, which today widens to the f64 kernel: give it a native f32 Newton
with a 2⁻²⁶ target (f32 needs only 25 bits, so one third-order + one second-order step in f32
suffices) **[I]**; (b) `rsqrt` 0.8612 ulp / 2.97× **[R]**: the same rounding test would let us
drop the final correction and possibly one Newton iteration. `rsqrt` needs no division at all
(`y ← y(1.5 − 0.5 a y²)` = 2 FMA), so a 2-iteration scheme + test is ~8 ops.

**Citations.** CORE-MATH paper §III (Algorithm 2, Lemma 1, the failure probability and the
`2⁻⁷⁵` threshold) <https://inria.hal.science/hal-03721525> **[M]**; source
<https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/cbrt/cbrt.c>.

---

### T8. Round-to-odd and the "two extra bits" rule (RLIBM)

**What.** If an intermediate is produced with **round-to-odd** and with **two more bits** than the
target format, then *any* subsequent rounding to the target is innocuous (no double-rounding
error). This is Boldo–Melquiond's result **[M]** and the basis of RLIBM-MultiRound's single
implementation for all rounding modes **[M]**.

**Why it matters here.** Our README lists eight entries at exactly 1.0 ulp because they "round
twice, once at 53-bit precision in a scaled domain and then onto the coarser subnormal grid"
(`log2p1`, `sinpi`/`tanpi`/`sincospi`, `atanpi`/`atan2pi`, `asinpi`, `atan2`) **[R]**. Producing
the intermediate with round-to-odd removes the double-rounding term and should take those entries
to ≤ 0.5 ulp *without* extra precision.

**Vector implementation (no FP rounding mode needed).** For a value `v` known to be inexact:
```
ro = bits(RN(v)) | 1        // OR the last mantissa bit: 1 vector OR + 1 compare
```
Setting the low bit of the rounded result moves it to the neighbouring representable value on the
correct side whenever `RN(v)` was not already odd **[I]**; the exactness test
(`v == RN(v)`) must be computed anyway for the flags.

**Op count.** +2 ops per affected entry. **Accuracy.** removes a whole class of 1.0-ulp results.
**Mapping.** Apply at the point where the scaled domain is converted back (e.g. the ×1/π in
`atanpi`), and at the f32 narrowing in the f32 kernels.

**Citations.** Boldo & Melquiond, *Emulation of a FMA and correctly-rounded sums: proved algorithms
using rounding to odd*, IEEE Trans. Computers 57(4) 2008
<https://hal.inria.fr/inria-00080427> **[M]**; RLIBM-MultiRound (2025)
<https://arxiv.org/abs/2504.07409> **[M]**.

---

### T9. RLIBM: correctly rounded from a *single* polynomial (and what it does/doesn't buy us)

**What.** RLIBM frames coefficient generation as a **linear program** over the *evaluated*
polynomial: find coefficients such that, for every input in a sub-interval, the sequence of
double-precision FMA operations produces a value that rounds to the correctly rounded result —
no DD, no Ziv test, no fallback. Sub-intervals are chosen by **bit-pattern splitting** (exponent +
top mantissa bits), each with its own short polynomial **[M]**.

**Measured claims [M].** RLibm-32 produces correctly rounded results for **all** float inputs for
its functions, and is on average **1.1× faster than glibc's float libm**, 1.5× faster than
Intel's float libm, 2× faster than CR-LIBM (PLDI 2021 / arXiv 2104.04043). The 2025 trig paper
extends it to `sinpi`/`cospi`-style range reduction with multi-word π.

**What it buys us.** For **f32** the answer is: *a double-precision evaluation plus one rounding is
enough; the (hi,lo) f32 pair is more precision than the problem needs.* Their measured 1.1× over
glibc (which is not correctly rounded) also shows the ceiling is modest — coefficient search is
not a speed revolution, it is a *correctness* technique that removes the fallback.

**What it does not buy us.** For **f64** you cannot get 53 correct bits from a 53-bit evaluation,
so the DD/table machinery stays. The transferable parts are (a) the *generation method*
(counterexample-guided LP, which we could run offline with Sollya/MPFR for one or two stubborn
entries), and (b) the *bit-pattern splitting* (which we already do with table indices).

**Effort.** High (needs their generator or an equivalent LP/ILP pipeline, plus an exhaustive f32
verifier). **Recommendation: defer**, except possibly for `erfcf`/`cospif` where a single
polynomial already nearly works.

**Citations.** RLIBM-32 <https://arxiv.org/abs/2104.04043>; RLIBM-ALL
<https://arxiv.org/abs/2108.06756>; MultiRound <https://arxiv.org/abs/2504.07409>; trig
<https://arxiv.org/abs/2510.13426>; project page
<https://people.cs.rutgers.edu/~sn349/rlibm/>.

---

### T10. Double-word arithmetic: tight bounds, overlap, and no renormalisation

**What.** Three independent results let us use *less* DD than we do:

* **Joldes–Muller–Popescu (TOMS 2017)** [M]: tight relative error bounds for `DW+FP`, `DW+DW`,
  `DW×FP`, `DW×DW`, `DW/FP`, `DW/DW`; for multiplication and division the bounds are *better* than
  previously published, and the previously published `DW+DW` bound was wrong. Their `muldd` is
  three multiplies, one FMA and two adds (the inner FMA recovers the product's low word)
  <https://hal.science/hal-01351529>.
* **Muller–Rideau (TOMS 2022)** [M]: all of those theorems formalised in Coq, several bounds
  *significantly improved*, and generalised to slightly different rounding modes
  <https://hal.science/hal-02972245>. This is the reference to cite when we want to *reduce* a
  correction term because the old bound was pessimistic.
* **SLEEF's engineering choice** [M]: "we only need 1-ULP overall accuracy for DP functions, we use
  simplified DD operators with less than the full DD accuracy. In SLEEF, we omit re-normalisation
  of DD values by default, allowing overlap between the two numbers. We carry out
  re-normalisation only when necessary." (paper §5) <https://arxiv.org/abs/2001.09258>.

**Op count.** Skipping the `fast_two_sum` after every DD add saves 3 ops per add; a
"no-renormalisation" DD Horner of degree n costs ~4n instead of ~6n **[I]**. Using the tight
`muldd` (`l = fma(ch,xh,−ahhh) + (ch·xl + cl·xh)`) is 3 flops instead of 5–6 **[M]** (source).

**Accuracy.** The 2017/2022 bounds are *tight*, so we can keep a 1-ulp guarantee while allowing
overlap; but for our 0.5-ulp entries the overlap must be tracked in the error budget **[I]**.
Rule of thumb: use no-renormalisation DD only where the final target is ≤ 1 ulp and the DD feeds a
short reconstruction; keep full DD for the `log`/`exp` cores where cancellation is possible.

**Mapping.** `vecmath_dd.h` currently has full two_sum/two_prod pairs. Add a second, "sloppy"
family (`VM_DD_ADD_FAST`, `VM_DD_MUL_FAST`) and use it in `k_erf`/`k_hyper`/`k_gamma` tails.

**Citations.** above; also the standard reference: Muller et al., *Handbook of Floating-Point
Arithmetic*, 2nd ed., Birkhäuser 2018, ch. 5 (function evaluation) and ch. 14 (extended formats)
<https://doi.org/10.1007/978-3-319-76526-6>.

---

### T11. Compensated Horner (and its SIMD-parallel variant)

**What.** The compensated Horner scheme (Graillat–Langlois–Louvet) evaluates a polynomial with
error-free transformations and a final correction so that the result is "as accurate as if computed
in twice the working precision, then rounded" **[M]**. The 2023 paper by Graillat, Ibrahimy,
Jeangoudoux and Lauter gives a **parallel** version for SIMD: the compensation terms are computed
on independent lanes/steps, so the critical path stays short **[M]**
<https://hal.science/hal-04180645>.

**Op count.** ~4 flops per Horner step (2 for the EFT of the multiply-add, 2 for the compensation)
versus ~6 for a full DD Horner and ~2 for plain Horner **[I]**. Crucially it is *more parallel*
than DD: the compensation chain is independent of the main chain, which is exactly what a
latency-bound VLEN-128 kernel needs.

**Where to use it.** The gamma fast-path polynomial (T6), the `erfc` mid-band (already
mixed-precision, could go compensated instead), and `pow`'s log-polynomial.

**Accuracy.** Twice-working-precision accuracy ⇒ ~2⁻¹⁰⁶ for a double polynomial, well inside
0.5 ulp; and unlike plain Horner it comes with a *rigorous* bound we can put into a Ziv test.

**Citations.** <https://hal.science/hal-04180645>; earlier: *Compensated Horner algorithm in K
times the working precision* <https://hal.inria.fr/inria-00267077>; *Faithful polynomial evaluation
with compensated Horner* <https://hal.inria.fr/hal-00107222>.

---

### T12. Trig without Payne–Hanek: the `sinpi`/`cospi`/`tanpi` family

**What.** Our README already states the right fact: for the π-scaled family `rint(2x)` and
`x − rint(2x)/2` are exact, so no Payne–Hanek is needed **[R]**. CORE-MATH's `cospi` confirms the
shape of the fast path: exact reduction to `|z| ≤ 0.25`, a **double-double** polynomial whose
coefficients show π folded in (the small-input branch literally stores `≈ −π²/2`, `≈ +π⁴/24`, …),
and a Ziv test with `er = |z|·2⁻¹²³ + 2⁻⁷⁸` **[M]** (source), plus a one-line "tiny" return
(`1.0 − 0x1p-55`) for inputs so small that `cos(πx)` rounds to the same double **[M]**.

**Is there a cheaper scheme?** Not for the reduction — the exact `rint` reduction is already
optimal. The remaining costs are (a) the polynomial degree and (b) the 4-way quadrant select.
Options:
* **Fold π into the coefficients and work in `z` directly** (CORE-MATH does this): avoids a
  separate multiply by π and its rounding, and lets the Ziv test certify 0.5 ulp **[M]**.
* **Share one polynomial between `sinpi` and `cospi`** (`cos(πz) = sin(π(z+½))` is *not* exact, so
  instead evaluate both from the same `z²` powers with different coefficient sets — saves the
  `z²` chain, not the Horner) **[I]**.
* For **f32**, `cospif` at 0.82× on a210 is the problem: the f32 kernel likely still carries a
  (hi,lo) pair. A 25-bit target with round-to-odd (T8) allows a single f32 Horner **[I]**.

**Op count.** The reduction is 3–4 ops (`vfrint`, `vfrsub`, exponent test); the polynomial is
degree 8–10 in `z²` (4–5 coefficients) if we use CORE-MATH's 0.25 reduction and a Ziv test, or
degree 14–18 (7–9 coefficients) without the test **[I]**. The quadrant select is 2–3 `vmerge`s
plus a sign XOR.

**Citations.** `cospi.c`
<https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/cospi/cospi.c>;
`sinpi.c`, `tanpi.c` in the same tree; ARM ORT's vector `sinpi`/`cospi`/`tanpi` (3.5–4 ulp
variants) show the aggressive end of the trade-off
<https://github.com/ARM-software/optimized-routines/tree/master/math/aarch64/advsimd>.

---

### T13. `expm1` / `log1p` / `log2p1`: reconstruct, don't subtract

**What.** SLEEF's `expm1k` is a good template **[M]** (source):
```
s = x - q*L2U - q*L2L                 // q = rint(x/ln2)
u = POLY10(s)                          // exp(s)-1-s^2/2-s, 10 coefficients, Estrin
u = s + s^2*(0.5 + s*u)                // expm1(s)
u = (q == 0) ? u : ldexp(u+1, q) - 1   // one select, no branch
```
i.e. the `q = 0` case is handled by a select rather than a separate path, and the polynomial
directly approximates `(exp(s)−1−s−s²/2)/s³`.

**Op count.** ~17 ops, no division, no table **[M]** (their code). For `log1p`, the standard
CORE-MATH/ARM shape is `u = 1+x` (exact for `|x| < 2⁻¹`), `log1p(x) = log(u) + (x − (u−1))/u`,
with the second term in DD **[M]** (CORE-MATH `log1p.c`, ARM `log1p.c`).

**Mapping.** Our `expm1` is 2.00× / 0.66× (a210) and `log2p1` 1.38× / 1.00× **[R]`. Both are
"reconstruction" kernels where a single `vmerge` on `q == 0` and a shorter polynomial should help;
`log2p1`'s `c/(u ln2)` term is already the right trick, so the a210 loss is latency, not ops —
see T14.

**Citations.** `sleefsimddp.c` (`expm1k`, `logk`)
<https://github.com/shibatch/sleef/blob/master/src/libm/sleefsimddp.c>; CORE-MATH
<https://gitlab.inria.fr/core-math/core-math/-/tree/master/src/binary64/log1p> **[M]**.

---

### T14. VLA / LMUL: what the literature actually says about one binary for VLEN 128 and 256

**What the spec says [M].** Strip-mining with `vsetvl` is the portable model; "larger LMUL can be
used to avoid stripmining for longer known-size application vectors at the cost of having fewer
available vector register groups"; and "larger LMUL settings can also be used to simply increase
vector length to reduce instruction fetch and dispatch overheads in cases where fewer vector
register groups are needed"
<https://github.com/riscv/riscv-v-spec/blob/master/v-spec.adoc> (§Vector Configuration, §Vector
Length).

**What SLEEF says about the critical path [M].** "The latency of an instruction or a subroutine is
the delay that it generates in a dependency chain… most of the instruction latency in the critical
path of evaluating a vector math function tends to be dominated by FMA operations… we regard each
FMA operation as a job that can be executed in parallel and try to reduce the length of the
critical path." Estrin's scheme is used for exactly this (paper §5).

**What the 2026 RVV measurement paper says [M].** "Default LMUL selection proves near-optimal in
most cases, though GCC 15 shows greater potential for performance gains through larger LMUL
tuning"; "predication overhead and stride load pose performance bottlenecks that current compiler
cost models do not yet fully capture"; GCC 15 introduced "dynamic LMUL selection" (BananaPi-F3,
RVV 1.0) <https://arxiv.org/abs/2605.10860>.

**What our own data says [R].** The a210 losers are "the long-dependency-chain kernels: at VLEN 128
a strip has half the lanes to hide the chain behind" **[R]**.

**Therefore, in priority order:**

1. **2-way strip interleaving (software pipelining).** Process two independent `vsetvl` strips per
   loop iteration, with separate live ranges, then merge stores. VLA-safe (each strip has its own
   `vl`), no LMUL change, no accuracy change. Halves the effective latency per element when the
   core can issue independent vector ops. Expected: the single biggest a210 win for the 13 losers
   **[I]** (1.3–1.8× on those entries; neutral on muse-pi where the chain is already hidden).
2. **Estrin instead of Horner** in every polynomial ≥ 6 terms. Same op count, log-depth chain,
   +2–4 live registers. Accuracy changes (different summation order) ⇒ re-run the gate; expect
   ±0.1 ulp **[I]**. For LMUL 4–8 kernels check register pressure before applying.
3. **`vsetvli` with `AVL = VLMAX` for full strips**, with a separate remainder epilogue: removes
   the `vl`-dependent loop-carried dependency and lets the compiler keep `vtype` constant.
4. **Keep the LMUL we already have** — LMUL 2 for every f64 translation unit, LMUL 4 for the
   f32 kernels and 8 for `vf_invtrig`/`vf_trig` **[R]** (`grep 'define VM_LMUL' *.c`) — so VLMAX
   halves at VLEN 128 exactly as the lane count does. The literature gives no evidence that a
   single LMUL is optimal at both widths, and default LMUL is reported near-optimal **[M]**; the
   fix for VLEN 128 is ILP (items 1–2), not a different LMUL.
5. **If the "no runtime LMUL dispatch" rule is ever relaxed**, the cheapest form is a
   *once-per-process* function-pointer selection on `vlenb` between two compiled kernels (LMUL 2
   and LMUL 4). That is not per-strip dispatch, keeps one binary, and GCC 15 does the same thing
   automatically ("dynamic LMUL selection") **[M]**. I flag it, but do not recommend breaking the
   constraint for this.
6. **Do not** chase masked-load predication: it is a measured bottleneck on RVV 1.0 **[M]**, and
   our fixup path already avoids it by branching on `vfirst`.

**Citations.** RISC-V V spec (above); SLEEF paper §5 <https://arxiv.org/abs/2001.09258>;
*Closer in the Gap* <https://arxiv.org/abs/2605.10860>; *Zoozve: a strip-mining-free RISC-V vector
extension* (evidence that strip-mining overhead is a real research target)
<https://arxiv.org/abs/2504.15678>; NEON→RVV porting experience (128-bit vector algorithms)
<https://arxiv.org/abs/2309.16509>.

---

### T15. Exception databases and hard-to-round search (correctness hardening)

**What.** CORE-MATH ships a tiny sorted table of inputs for which the fast/accurate path is not
correctly rounded in some rounding mode, checked by binary search at the end (`exp`'s `db[]` has a
couple of dozen entries; `atan` a similar list) **[M]**. Separately, the ARITH 2022 paper describes
an algorithm that finds all m-hard-to-round cases for a univariate function in a few cycles per
input using a degree-2 Taylor expansion and 64-bit modular arithmetic **[M]**.

**Why it matters.** Our gate is 100k MPFR records per function. A 1-ulp result that happens to sit
on a hard-to-round input will not be found by sampling. The published HR lists (CORE-MATH "worst
cases" page) are free test vectors.

**Vector mapping.** The database check is `vfirst` on 8 compares against a small sorted list
(binary search only when nonzero, i.e. essentially never) — the same guard shape we already use.
Cheap insurance, ~5 ops/lane if done as a linear scan of a few values, or 0 in the common case if
we check only after a Ziv failure **[I]**.

**Citations.** CORE-MATH worst-cases page (linked from
<https://core-math.gitlabpages.inria.fr/>); ARITH 2022 paper §II-B, Algorithm 1
<https://inria.hal.science/hal-03721525>; Lefèvre's hard-to-round tables are the classical
reference.

---

### T16. What ARM ORT does (and what it tells us about effort/accuracy trade-offs)

ARM's AdvSIMD/SVE math is the closest thing to a "128-bit vector libm" and its per-function ULP
numbers are the bounds its own harness asserts (`TEST_ULP`) **[M]**:

| function | ARM AdvSIMD vector max ULP | technique (from source) |
|---|---|---|
| `exp` | 1.9 | table (N entries) + 5-coefficient poly + bit-built scale, no DD |
| `log` | 1.67 | 128-entry table, per-lane *scalar* loads (`vld1q`), 5-coefficient poly, plain double |
| `erf` | 1.79 | rational + polynomial pieces |
| `erfc` | 1.21 | polynomial + `exp`, no DD |
| `cbrt` | 3.17 | poly + Newton |
| `atan2` | 1.47 (1.97 observed on NEON) | **1 divide + 20-term polynomial, Estrin via z²,z⁴,z⁸,z¹⁶** |
| `lgamma` | — | **scalar `lgamma_r` per lane** — ARM gave up on a vector algorithm |

Two conclusions. (a) ARM accepts 1.2–3.2 ulp; we cannot, so their exact polynomials are not
reusable, but their *structure* (no DD, table + short poly, Estrin) is. (b) Even ARM ships vector
`lgamma` as a scalar loop, which confirms that our own fast-path approach is already ahead of the
state of practice — the accuracy fix (T6) is worth doing carefully rather than by importing
someone else's code.

**Citations.** <https://github.com/ARM-software/optimized-routines/tree/master/math/aarch64/advsimd>
(per-file `TEST_ULP` macros), and the SVE tree
<https://github.com/ARM-software/optimized-routines/tree/master/math/aarch64/sve>.

---

## 3. Per-entry playbook

| entry | current [R] | primary technique | secondary |
|---|---|---|---|
| `atan2`, `atan2pi` | 1.36× / 1.11× (muse), 0.31× / 0.78× (a210) | **T4** fine table + tangent subtraction + 3-term poly | T3 Ziv, T14 interleave |
| `atanpi`, `atan` | 1.42× / 1.79× | T4 (shared kernel) | T3 |
| `atan2f`, `atanf` | 0.62× (a210) | T4 + native f32 path (T8/T9) | T14 |
| `erfc`, `erfcf` | 0.844 ulp, 1.24× / 0.71× | **T5** odd polynomial + `exp_1` | T10 sloppy DD |
| `lgamma`, `lgamma_r`, `tgamma` | 1.27–1.54 ulp band | **T6** zero-aware fast path, mixed-DD Horner | T11 compensated Horner |
| `log2p1`, `log2`, `log10` | 1.38× / 0.91× (a210) | **T1** exact-residual table | T14 |
| `log`, `log1p` | 2.24× / 2.12× | T1 | T14 |
| `expm1`, `expm1f` | 0.66× (a210) | **T13** branchless reconstruction | T14 |
| `cbrtf`, `rsqrtf`, `powf` | 0.73× / 0.73× (a210) | **T7** native f32 Newton + rounding test | T8 |
| `cbrt`, `rsqrt` | 3.55× / 2.97× | T7 (fewer iterations) | — |
| `sinpi`, `cospi`, `tanpi`, `sincospi`, and `f` | 1.0 ulp some, `cospif` 0.82× | **T8** round-to-odd; f32 single-Horner | T12 |
| `asinh`, `acosh`, `atanh` | 0.66× (`asinh`, a210) | T4/T1 composition + T14 | T3 |
| `erf` | 1.0 ulp, 1.40× | T5/T3 | T8 | 
| `hypot` | 0.5 ulp, 2.39× | — | — |
| all DD kernels | — | **T2** `FastFMA_DW`, **T10** tight bounds | T11 |

---

## 4. Prioritised recommendations

Ranked by expected gain ÷ (accuracy risk × effort). "Gain" is an inference unless flagged.

| # | Technique | Entries helped | Expected speed gain (justification) | Accuracy risk | Effort | Verdict |
|---|---|---|---|---|---|---|
| 1 | **2-way strip interleave + Estrin** (T14) | all 13 a210 losers; all latency-bound kernels | **1.3–1.8× on those entries at VLEN 128** [I]: our own README attributes the losses to "half the lanes to hide the chain"; SLEEF's stated method is to shorten the critical path [M]. Neutral on muse-pi. | **none** (interleave) / low (Estrin reorders sums; re-run gate) | low–medium (loop + polynomial rewrite, no algorithm change) | **Do first** |
| 2 | **Fine atan table + tangent subtraction + 3-term poly + Ziv** (T4/T3) | `atan2`, `atan2pi`, `atanpi`, `atan`, `atan2f`, `atanpif`, and the asin/acos family | **1.6–2.0×** on the atan family [I]: 3 divides → 2, 11-term poly → 3-term, no DD composition; CORE-MATH's structure is measured 0.5 ulp [M]. Turns 1.11×→~1.8× and 0.31×→~0.55× (still needs #1 to clear parity). | low (0.5 ulp certified by the interval test; DD fallback kept) | medium (table generation + one shared kernel) | **Do second** |
| 3 | **Gamma zero-aware fast path, mixed-DD Horner** (T6) | `lgamma`, `lgamma_r`, `tgamma`, and `f` | **−5 to −15%** on the fast path [I] (cost, not gain) — buys 1.41 → ≤0.6 ulp on the band. CORE-MATH ships the decomposition [M]; our own erfc shows mixed-DD Horner is affordable [R]. | **very low** (strictly less cancellation) | medium (new coefficient set, reuse `G(z)` form) | **Do third** (contract quality) |
| 4 | **`erfc` odd asymptotic poly + `exp_1`** (T5) | `erfc`, `erfcf` | **1.2–1.35×** [I]: removes the serial DD divide at the end of the tail and uses a 2⁻⁷⁴ `exp` instead of the 0.5-ulp `exp`; CORE-MATH's whole tail is 111 cycles, 20 of them `exp_1` [M]. | low (error budget 2⁻⁶⁷, we have 0.84 ulp of headroom) | medium (6 band tables + stripped exp) | **Do fourth** |
| 5 | **`FastFMA_DW` + tight DD bounds + sloppy DD** (T2/T10) | every DD kernel: log, exp, gamma, erf, hyper, invtrig | **5–15%** on DD-heavy kernels [I]; CORE-MATH measured 68.7→55.7 cycles (−19%) on the accurate exp path with the same primitive [M]. | **none** (bounds tightened, not loosened, if we adopt the paper verbatim) | **low** (one header, ~30 lines) | **Cheap win — do in parallel** |
| 6 | **Log family exact-residual table** (T1) | `log`, `log2`, `log10`, `log1p`, `log2p1` (+f32) | **1.2–1.3×** [I]: drops the DD combine (12–13 ops + 3 loads vs ~22–25); CORE-MATH's fast path fails only 2⁻¹¹·⁵ of the time [M]. Fixes `log2p1` 1.00× at a210 and `log2` 0.91×. | low (0.5 ulp via interval test; table must be *generated and verified*, not computed) | medium–high (new table + generator) | Do after #1/#2 |
| 7 | **Round-to-odd for the double-rounded entries** (T8) | `sinpi`, `tanpi`, `sincospi`, `atanpi`, `atan2pi`, `asinpi`, `log2p1`, `hypot` (subnormal) | **0×** speed, but removes the 1.0-ulp class → accuracy headroom, which *enables* cheaper f32 paths (#9) | **none** (it is the fix, not the risk) | low (+2 ops at the conversion point) | Do with #2 |
| 8 | **Native f32 `cbrtf`/`rsqrtf`/`powf` + f32 Ziv** (T7/T9) | `cbrtf` 0.73×, `rsqrtf`, `powf` | **1.4–1.8×** on those f32 entries at a210 [I]: today they widen to f64, doubling vector ops and halving lanes; RLibm shows a 25-bit target is enough for f32 [M]. | low–medium (needs f32 error analysis; round-to-odd helps) | medium | Do after #1 |
| 9 | **`expm1`/`log2p1` branchless reconstruction** (T13) | `expm1`, `expm1f`, `log2p1`, `log2p1f` | **1.1–1.25×** [I]: fewer ops and one `vmerge` instead of a case split; SLEEF's shape is measured 1 ulp [M]. | low | low | Fill-in work |
| 10 | **`cbrt`/`rsqrt` higher-order Newton + rounding test** (T7) | `cbrt`, `rsqrt`, `cbrtf` | **1.1–1.2×** [I]: one third-order step replaces two second-order ones; the test removes the final correction. Already 3.55×/2.97× so the payoff is small. | low (test is rigorous, failure 1e−7) [M] | low–medium | Defer unless touching cbrt |
| 11 | **Exception database + published HR cases** (T15) | all | 0× | **negative risk** (catches latent 1-ulp failures) | low | Do as test hardening |
| 12 | **RLIBM LP coefficient generation** (T9) | f32 entries, maybe `erfcf` | **1.0–1.1×** at best vs glibc [M]; not a speed play | none | **high** (offline LP pipeline + exhaustive f32 verifier) | Defer |
| 13 | **128-entry gather tables everywhere** | log/atan | **unknown, possibly negative** on RVV: `vluxei64`/`vrgather` cost is the risk; SLEEF measured tables *slower* than polynomials on its targets [M] | none | low | Measure one kernel before committing |

---

## 5. Things I would *not* do

* **Full double-double on the fast paths.** CORE-MATH's own fast paths are plain double + a test;
  the DD is in the table entries and in the rare fallback. Our `VM_LOG_DD` combine is exactly what
  T1 makes unnecessary (a plain-double polynomial over an *exact* residual reaches 0.5 ulp).
* **A 20-term polynomial for `atan` (SLEEF's choice).** It is branch-free and table-free, but at
  VLEN 128 it is a 20-deep chain; a table of 16–32 DD entries plus a 4–5-term polynomial is
  shorter (T4). SLEEF can afford it because its targets have wider vectors or cheap gathers.
* **Runtime per-strip LMUL dispatch.** Not needed; interleave + Estrin gets the ILP without it.
* **Relaxing the 1.0-ulp contract for speed on a210.** The a210 numbers are latency artefacts, not
  precision artefacts; the fixes above keep the ulp maxima.

---

## 6. Reference list

**Correctly-rounded scalar (CORE-MATH / ancestors)**
1. A. Sibidanov, P. Zimmermann, S. Glondu, *The CORE-MATH Project*, ARITH 2022, pp. 26–34 —
   <https://inria.hal.science/hal-03721525> (methodology, hard-to-round search, cbrt Algorithm 2).
2. CORE-MATH project page (function matrix, glibc 2.42/2.43 integration, worst cases) —
   <https://core-math.gitlabpages.inria.fr/>.
3. Sources: `log` <https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/log/log.c>,
   `exp` <https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/exp/exp.c>,
   `atan` <https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/atan/atan.c>,
   `atan2` <https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/atan2/atan2.c>,
   `cbrt` <https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/cbrt/cbrt.c>,
   `erfc` <https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/erfc/erfc.c>,
   `lgamma` <https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/lgamma/lgamma.c>,
   `tgamma` <https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/tgamma/tgamma.c>,
   `cospi` <https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/cospi/cospi.c>.
4. F. de Dinechin, C. Lauter, J.-M. Muller, *Fast and correctly rounded logarithms in double
   precision*, RAIRO-ITA 41(1), 2007 — <https://hal.science/ensl-00000007>.
5. D. Defour, F. de Dinechin, J.-M. Muller, *Correctly rounded exponential function in double
   precision arithmetic*, SIAM J. Sci. Comput. 2001 — <https://hal.inria.fr/inria-00072387>.
6. P. Zimmermann, *Note on FastTwoSum with Directed Roundings*, 2022 —
   <https://hal.inria.fr/hal-03798376> (cited by CORE-MATH `log.c` for the `fast_two_sum` bound we
   use in `vecmath_dd.h`).

**Vectorized libraries**
7. N. Shibata, F. Petrogalli, *SLEEF: A Portable Vectorized Library of C Standard Mathematical
   Functions*, IEEE TPDS 31(6), 2020 — <https://arxiv.org/abs/2001.09258> (u10/u35, no-gather
   preference, DD without renormalisation, Estrin).
8. SLEEF source <https://github.com/shibatch/sleef/blob/master/src/libm/sleefsimddp.c> and RVV
   backend <https://github.com/shibatch/sleef/blob/master/src/arch/helperrvv.h>.
9. ARM Optimized Routines, vector math —
   <https://github.com/ARM-software/optimized-routines/tree/master/math/aarch64/advsimd> and
   <https://github.com/ARM-software/optimized-routines/tree/master/math/aarch64/sve>.
10. J. P. Lim, S. Nagarakatte, *RLIBM-32*, PLDI 2021 — <https://arxiv.org/abs/2104.04043>;
    *RLIBM-ALL* <https://arxiv.org/abs/2108.06756>;
    *RLibm-MultiRound* <https://arxiv.org/abs/2504.07409>;
    *Fast Trigonometric Functions using the RLIBM Approach* <https://arxiv.org/abs/2510.13426>;
    project <https://people.cs.rutgers.edu/~sn349/rlibm/>.

**Multi-precision arithmetic**
11. M. Joldes, J.-M. Muller, V. Popescu, *Tight and rigorous error bounds for basic building blocks
    of double-word arithmetic*, ACM TOMS 44(2), 2017 — <https://hal.science/hal-01351529>.
12. J.-M. Muller, L. Rideau, *Formalization of double-word arithmetic…*, ACM TOMS 2022 —
    <https://hal.science/hal-02972245> (improved bounds).
13. C.-P. Jeannerod, M. Joldes, N. Louvet, J.-M. Muller, *Extended-Precision FMA under Parameterized
    Double-Word Overlap*, ARITH 2026 — <https://inria.hal.science/hal-05517451> (the `FastFMA_DW`
    used by CORE-MATH `exp`).
14. S. Graillat, Y. Ibrahimy, C. Jeangoudoux, C. Lauter, *A parallel compensated Horner scheme for
    SIMD architecture*, 2023 — <https://hal.science/hal-04180645>.
15. S. Boldo, G. Melquiond, *Emulation of a FMA and correctly-rounded sums: proved algorithms using
    rounding to odd*, IEEE Trans. Computers 57(4), 2008 — <https://hal.inria.fr/inria-00080427>.
16. J.-M. Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed., Birkhäuser 2018 —
    <https://doi.org/10.1007/978-3-319-76526-6>.

**Special functions**
17. M. M. Shepherd, J. G. Laframboise, *Chebyshev approximation of (1+2x)exp(x²)erfc x in
    0 ≤ x < ∞*, Math. Comp. 36 (1981) 249–253 —
    <https://doi.org/10.1090/S0025-5718-1981-0595059-5>.
18. S. Chevillard, *The functions erf and erfc computed with arbitrary precision and explicit error
    bounds* — <https://inria.hal.science/ensl-00356709>.
19. fdlibm (our current `atan`/`lgamma` baseline) — <https://netlib.org/fdlibm/>.

**RVV / VLA performance**
20. *The RISC-V Vector Extension Specification v1.0* —
    <https://github.com/riscv/riscv-v-spec/blob/master/v-spec.adoc>.
21. R. Shi et al., *Closer in the Gap: Towards Portable Performance on RISC-V Vector Processors*,
    2026 — <https://arxiv.org/abs/2605.10860> (GCC 15 dynamic LMUL, predication overhead).
22. *Zoozve: A Strip-Mining-Free RISC-V Vector Extension…*, 2025 —
    <https://arxiv.org/abs/2504.15678>.
23. *SIMD Everywhere Optimization from ARM NEON to RISC-V Vector Extensions*, 2023 —
    <https://arxiv.org/abs/2309.16509>.

---

## 7. Caveats

* All "expected speed gain" numbers except those marked **[M]** are my estimates from op counting
  and from the dependency-chain argument; none has been built or benchmarked. The two that I would
  bet on are #1 (interleave) and #5 (`FastFMA_DW`), because both are supported by *measured*
  claims in the sources and neither changes the numerical algorithm.
* The 128-entry-vs-16-entry table decision for `atan2` (T4) is **not** decidable from the
  literature: ARM does per-lane scalar loads on NEON, CORE-MATH relies on scalar L1 hits, SLEEF
  avoids tables. It must be measured on muse-pi and a210 with a micro-benchmark before the kernel
  is rewritten.
* The gamma band numbers (1.27/1.41/1.54 ulp) come from our own `scripts/domain-check.sh`
  measurements quoted in the README **[R]**; I did not re-run them (no board access from this
  session).
* CORE-MATH's glibc integration claim (2.43 for binary64 `erf`/`erfc`/`lgamma`/`tgamma`, 2.42 for
  31 binary32 functions) is from the project page as of 2026-11 **[M]**; I did not fetch the glibc
  tree to confirm the file mapping.
