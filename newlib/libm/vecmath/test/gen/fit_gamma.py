#!/usr/bin/env python3
"""Derive and validate every constant the tgamma/lgamma/lgamma_r kernel
needs, and check the algorithm the kernel is to implement.

Nothing here is compiled; run it to regenerate the tables and to re-check
the design.

    python3 fit_gamma.py            # validate + print the tables

lgamma(x) is built from four regions for x > 0, all of them double-double
so that the final collapse is the only rounding:

    x <= 1.5   u = (x < 0.5) ? x : x-1
               lgamma = -log1p(u) + u*C1 + u^2*G(u) + (x < 0.5 ? -log(x) : 0)
    1.5 < x <= 2.5
               w = x-2 ; lgamma = w*C1 + w^2*G(w)
    2.5 < x < 10
               P = prod(x+k, k=0..7) ; lgamma = stirling(x+8) - log(P)
    x >= 10    lgamma = stirling(x)

with C1 = 1-gamma and G(z) = (logGamma(1+z) + log1p(z) - z*C1)/z^2, fitted
once on [-0.5, 0.5] and used by both of the first two regions.

The x <= 1.5 region carries the zeros at x = 1 and x = 2, where the answer
is exactly 0.  The obvious build -- lgamma(x) = lgamma(x+10) -
log(x(x+1)...(x+9)) -- destroys them: near x = 1 the two sides are both
~15.1, so the result is a cancellation of two large numbers and only a few
digits survive.  Here the cancellation is done analytically instead:
logGamma(1+z) = -log1p(z) + z*C1 + z^2*G(z) IS the series, so a small
result comes out of dd terms of size 1 and keeps its relative accuracy down
to z = 2^-52.  The 1.5 < x <= 2.5 region is the same formula minus its
-log1p term, because logGamma(2+w) = log(1+w) + logGamma(1+w) and the two
log1p's cancel exactly -- so w = 0 gives exactly 0, as it must.

The negative side is the reflection

    lgamma(x) = log(pi) - log|sin(pi x)| - lgamma(1-x)      (x < 0)

and |sin(pi x)| has to be double-double, for two reasons.  A plain double
sinpi is 0.5 ulp, and that relative error turns into an absolute error of
~1e-16 in log|sin| and hence in lgamma -- comparable with the smallest
|lgamma| on x < 0, so it costs a large fraction of an ulp by itself.  And
lgamma has zeros on the negative axis as well (where |Gamma| = 1: x ~ -1.56
and -2.40, and two more between every pair of poles), where the relative
ulp metric is unforgiving and an absolute error of 1e-16 is thousands of
ulp.  dd handles it because the three reflection terms are O(1), so 2^-106
of them is ~1e-31 absolute.

The reduction is the same one k_trig.inc already has: q = rint(2x),
r = x - q/2 (exact for |x| < 2^62), then (y0, y1) = pi*r as a dd from
VM_PI_RED, and sin/cos of that dd from the 11-term series below.  The
quadrant picks which, and |sin| feeds a dd log.  For |x| >= 2^52 the
argument is an integer, so it is a pole and the reflection is not reached.

Stirling: 12 terms of B_2k/(2k(2k-1)) z^(k-1) with z = 1/x^2.  At x = 10
the 12th term is already 1e-23, so the truncation is what limits this
region, not the arithmetic.  z^2 and z are clamped (to 2^100) for the
series only: at x = 1e308, 1/x is subnormal and forming it raises a
spurious UNDERFLOW on a lane whose answer is an overflowing +Inf.

tgamma(x) = sign(Gamma(x)) * exp(lgamma(x)), the exp in dd (VM_EXP_DD) so
that lgamma's ABSOLUTE error is what carries over -- which it does: at
x near 1, lgamma ~ 0 and the absolute error is ~1e-30.  The four special
cases are not symmetric with lgamma's: tgamma(+-0) = +-Inf with
divide-by-zero, tgamma(negative integer) = NaN with invalid (C99 F.10.4.1,
which the scalar fdlibm does not do -- it returns +-Inf through
exp(lgamma), so the generator has to expect the C99 answer, not the
scalar's), and tgamma underflows to +-0/ -0 for large negative x.
lgamma itself returns +Inf at every pole, with divide-by-zero.
"""

import sys
from mpmath import (mp, mpf, cos, fabs, factorial, log, log1p, loggamma,
                    pi, sin)


def dd(v):
    hi = mpf(float(v))
    return hi, v - hi


def emit(name, c):
    print("static const double %s[%d][2] = {" % (name, len(c)))
    for v in c:
        hi = mpf(float(v))
        print("  { %.17e, %.17e }," % (float(hi), float(v - hi)))
    print("};")


def cheb_to_power(cs, scale):
    """sum cs[k] T_k(u) -> power basis in z, with u = z/scale."""
    deg = len(cs) - 1
    T = [[mpf(1)], [mpf(0), 1 / scale]]
    for k in range(2, deg + 1):
        a = [mpf(0)] * (len(T[k - 1]) + 1)
        for j, v in enumerate(T[k - 1]):
            a[j + 1] += 2 * v / scale
        b = T[k - 2] + [mpf(0)] * (len(a) - len(T[k - 2]))
        T.append([a[j] - b[j] for j in range(len(a))])
    out = [mpf(0)] * (deg + 1)
    for k in range(deg + 1):
        for j, v in enumerate(T[k]):
            out[j] += cs[k] * v
    return out


def main():
    mp.dps = 60
    gam = mp.euler
    C1 = 1 - gam

    # ---- G(z) on [-0.5, 0.5], degree 24 --------------------------------
    def G(z):
        return mpf(0) if z == 0 else (loggamma(1 + z) + log1p(z) - z * C1) / (z * z)

    DEG, N = 24, 400
    cs = [2 * mp.fsum(G(mpf('0.5') * cos(pi * (j + mpf('0.5')) / N))
                      * cos(k * pi * (j + mpf('0.5')) / N)
                      for j in range(N)) / N for k in range(DEG + 1)]
    cs[0] /= 2
    Gp = cheb_to_power(cs, mpf('0.5'))
    worstG = max(fabs(mp.fsum(Gp[j] * (mpf(t) / 1000) ** j for j in range(DEG + 1))
                      - G(mpf(t) / 1000)) for t in range(-500, 501, 3))
    print("G fit:       max abs err on [-0.5,0.5] = %.3e" % float(worstG))

    # ---- Stirling -----------------------------------------------------
    B = [mpf(0)] + [mp.bernoulli(2 * k) for k in range(1, 13)]
    S = [B[k] / (2 * k * (2 * k - 1)) for k in range(1, 13)]

    def stirling(x):
        z = 1 / (x * x)
        w = mpf(0)
        for c in reversed(S):
            w = w * z + c
        return (x - mpf('0.5')) * log(x) - x + mpf('0.5') * log(2 * pi) + w / x

    print("Stirling:    rel err at x=10 = %.3e, at x=8 (reached only via"
          " the x+8 shift) = %.3e"
          % (float(fabs(stirling(mpf(10)) - loggamma(mpf(10))) / loggamma(mpf(10))),
             float(fabs(stirling(mpf(8)) - loggamma(mpf(8))) / loggamma(mpf(8)))))

    # ---- sin/cos series on [-pi/4, pi/4], 11 terms each ----------------
    NS = 11
    PS = [mpf((-1) ** (k + 1)) / factorial(2 * k + 3) for k in range(NS)]
    PC = [mpf((-1) ** (k + 1)) / factorial(2 * k + 2) for k in range(NS)]

    def sc(y):
        t = y * y
        ps = pc = mpf(0)
        for c in reversed(PS):
            ps = ps * t + c
        for c in reversed(PC):
            pc = pc * t + c
        return y * (1 + t * ps), 1 + t * pc

    w = mpf(0)
    for i in range(-1000, 1001):
        if i == 0:
            continue                       # sin(0) is 0: no relative error
        y = mpf(i) / 1000 * pi / 4
        s, c = sc(y)
        w = max(w, fabs(s - sin(y)) / fabs(sin(y)),
                fabs(c - cos(y)) / fabs(cos(y)))
    print("sincos fit:  max rel err on [-pi/4,pi/4] = %.3e" % float(w))

    # ---- the algorithm itself, in exact arithmetic ---------------------
    def gp(z):
        v = mpf(0)
        for c in reversed(Gp):
            v = v * z + c
        return v

    def lgam_pos(x):
        if x <= mpf('1.5'):
            u = x if x < mpf('0.5') else x - 1
            r = -log1p(u) + u * C1 + u * u * gp(u)
            return r - log(x) if x < mpf('0.5') else r
        if x <= mpf('2.5'):
            w = x - 2
            return w * C1 + w * w * gp(w)
        if x < 10:
            p = mpf(1)
            for k in range(8):
                p *= (x + k)
            return stirling(x + 8) - log(p)
        return stirling(x)

    def lgam(x):
        if x > 0:
            return lgam_pos(x)
        return log(pi) - log(fabs(sin(pi * x))) - lgam_pos(1 - x)

    worst, wx = mpf(0), None
    tests = [mpf(i) / 1000 for i in range(1, 4000)] \
        + [-mpf(i) / 1000 for i in range(1, 2000)] \
        + [mpf(i) / 100 for i in range(1000, 4000)] \
        + [mpf('0.9999999999999999'), mpf('1.0000000000000002'), mpf('1.5'),
           mpf('2.5'), mpf('2.4999999999999996'), mpf('0.5'), mpf('1e-8'),
           mpf('9.999999999999998'), mpf('10'), mpf('171.7'), mpf('1e10'),
           mpf('-1.56'), mpf('-2.4'), mpf('-3.9554570214751834')]
    for x in tests:
        try:
            ref = loggamma(x).real        # -> log|Gamma| on the negative axis
        except ValueError:
            continue                      # a pole: lgamma is +Inf there
        if ref == 0:
            continue
        e = fabs(lgam(x) - ref) / fabs(ref)
        if e > worst:
            worst, wx = e, x
    print("design:      worst relative error %.3e at x = %s"
          % (float(worst), mp.nstr(wx, 12)))
    ok = worst < mpf('1e-18') and fabs(lgam(mpf(1))) == 0 and fabs(lgam(mpf(2))) == 0
    print("design:      lgamma(1) = %s, lgamma(2) = %s"
          % (mp.nstr(lgam(mpf(1)), 3), mp.nstr(lgam(mpf(2)), 3)))
    print("design:      %s" % ("OK" if ok else "FAILED"))

    print("\n/* --- the tables --- */")
    emit("vm_gam_G", Gp)
    emit("vm_gam_S", S)
    emit("vm_gam_PS", PS)
    emit("vm_gam_PC", PC)
    print("\n/* scalars: hi, lo */")
    for n, v in (("C1", C1), ("LOGPI", log(pi)), ("LOG2PI", log(2 * pi))):
        hi, lo = dd(v)
        print("#define VM_GAM_%-7s %.17e, %.17e" % (n, float(hi), float(lo)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
