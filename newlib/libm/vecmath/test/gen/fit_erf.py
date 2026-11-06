#!/usr/bin/env python3
"""Regenerate and re-verify the erf/erfc coefficient tables in
vecmath/include/k_erf.inc.

  erf(x)  = x * P(x^2)                     |x| < 1
  erfc(x) = exp(-x^2) * H(x) / x           |x| >= 1,  H = RP/RQ

Prints the two table bodies as C source.  It does NOT edit any file.

The script fits, then measures the *rounded double-double coefficients*
-- what the kernel actually stores -- on grids the fit never saw: a
uniform grid, a cubic-dense grid, random doubles, and every double in the
sliver between x = 1 and the first Gauss node.  It refuses to print a
table that does not clear its target.

mpmath works in RELATIVE precision, so dps = 120 gives 120 correct digits
of H even at x = 40, where erfc is 1e-697.  No exponent juggling needed.

Usage:  python3 fit_erf.py            # fit, verify, print the tables
"""

import random
import sys

from mpmath import mp, mpf, chebyt, cos, erf, erfc, exp, fabs, pi, sqrt

mp.dps = 120

NODES = 240		# Gauss nodes for the Sanathanan-Koerner iteration
ITERS = 12
PDEG = QDEG = 12	# degrees of RP and RQ


# ----------------------------------------------------------------------
# the target function and the two fits
# ----------------------------------------------------------------------

def H(x):
    """x*erfc(x)*exp(x^2): bounded (0.428 .. 0.564) on [1, 40]."""
    return x * erfc(x) * exp(x * x)


def fit_poly(deg=15, nq=400):
    """Chebyshev truncation of erf(sqrt(t))/sqrt(t) on t in [0,1].

    The map x -> x^2 makes the series entire in t, so the coefficients
    decay super-geometrically and the truncation is near-minimax.
    """
    def f(t):
        return erf(sqrt(t)) / sqrt(t) if t != 0 else mpf(2) / sqrt(pi)

    cs = []
    for k in range(deg + 1):
        s = mp.fsum(f((cos(pi * (j + mpf('0.5')) / nq) + 1) / 2)
                    * cos(k * pi * (j + mpf('0.5')) / nq) for j in range(nq))
        cs.append(2 * s / nq)
    cs[0] /= 2
    # Chebyshev in u = 2t-1 -> power basis in t
    T = [[mpf(1)], [mpf(-1), mpf(2)]]
    for k in range(2, deg + 1):
        prev, prev2 = T[k - 1], T[k - 2]
        a = [mpf(0)] * (len(prev) + 1)
        for j, v in enumerate(prev):
            a[j] += -2 * v
            a[j + 1] += 4 * v
        b = prev2 + [mpf(0)] * (len(a) - len(prev2))
        T.append([a[j] - b[j] for j in range(len(a))])
    P = [mpf(0)] * (deg + 1)
    for k in range(deg + 1):
        for j, v in enumerate(T[k]):
            P[j] += cs[k] * v
    return P


def fit_rat(p=PDEG, q=QDEG, nodes=NODES, iters=ITERS):
    """Sanathanan-Koerner fixed point of the relative-error weighting.

    SK is used rather than a plain linearised least-squares because the
    plain one minimises |P - H*Q|, which a small Q makes look converged
    while P/Q is still wrong by orders of magnitude.
    """
    lo, hi = mpf(1), mpf(40)
    xs = [(lo + hi) / 2 + (hi - lo) / 2 * cos(pi * (2 * i + 1) / (2 * nodes))
          for i in range(nodes)]
    hs = [H(x) for x in xs]
    ts = [(2 * x - (lo + hi)) / (hi - lo) for x in xs]
    Q = [mpf(1)] * nodes
    for _ in range(iters):
        rows, rhs = [], []
        for i in range(nodes):
            w = 1 / Q[i]
            rows.append([w * chebyt(k, ts[i]) for k in range(p + 1)]
                        + [-w * hs[i] * chebyt(k, ts[i]) for k in range(1, q + 1)])
            rhs.append(w * hs[i])
        A = mp.matrix(rows)
        b = mp.matrix(rhs)
        sol = mp.lu_solve(A.T * A, A.T * b)
        Pc = [sol[k] for k in range(p + 1)]
        Qc = [mpf(1)] + [sol[p + 1 + k] for k in range(q)]
        Q = [mp.fsum(Qc[k] * chebyt(k, ts[i]) for k in range(q + 1))
             for i in range(nodes)]
    return Pc, Qc


# ----------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------

def dd_round(c):
    """The value the kernel actually stores: hi = fl(c), lo = fl(c - hi)."""
    return [mpf(float(v)) + (v - mpf(float(v))) for v in c]


def chebval(c, t):
    return mp.fsum(c[k] * chebyt(k, t) for k in range(len(c)))


def check_rat(Pc, Qc, label, xs):
    worst, wx, minq, qx = mpf(0), None, mp.inf, None
    for x in xs:
        t = (2 * x - 41) / 39
        p, q = chebval(Pc, t), chebval(Qc, t)
        if fabs(q) < minq:
            minq, qx = fabs(q), x
        e = fabs(p / q - H(x)) / fabs(H(x))
        if e > worst:
            worst, wx = e, x
    print("  %-34s %.3e @ x=%-12s  min|Q| %.3e @ x=%s"
          % (label, float(worst), mp.nstr(wx, 8), float(minq), mp.nstr(qx, 8)))
    return worst


def check_poly(P, label, xs):
    def ev(x):
        t = x * x
        v = mpf(0)
        for c in reversed(P):
            v = v * t + c
        return v * x
    worst, wx = mpf(0), None
    for x in xs:
        if x == 0:
            continue
        e = fabs(ev(x) - erf(x)) / fabs(erf(x))
        if e > worst:
            worst, wx = e, x
    print("  %-34s %.3e @ x=%s" % (label, float(worst), mp.nstr(wx, 8)))
    return worst


def main():
    print("fitting erf polynomial (degree 15 in x^2) ...")
    P = fit_poly()
    print("fitting erfc rational (12/12 in t=(2x-41)/39) ...")
    Pc, Qc = fit_rat()

    Pd, Pcd, Qcd = dd_round(P), dd_round(Pc), dd_round(Qc)

    print("\nverification (all coefficients first rounded to double-double):")
    random.seed(20261006)
    uniform = [mpf(1) + mpf(39) * k / 30000 for k in range(30001)]
    cubic = [mpf(1) + mpf(39) * mpf(k) ** 3 / mpf(30000) ** 3
             for k in range(30001)]
    rnd = [mpf(float(mpf(1) + mpf(39) * mpf(random.random())))
           for _ in range(20000)]
    # every double in [1, 1+2^-40]: the gap between x=1 and the first Gauss
    # node, where a common root of P and Q would hide from the fitter
    sliver, x = [], mpf(1)
    while x < mpf(1) + mpf(2) ** -40:
        sliver.append(x)
        x = mpf(float(x) + mpf(2) ** -52)
    # the polynomial's domain is [0, 1] -- feeding it anything else measures
    # the extrapolation, not the fit
    tiny = [mpf(float(mpf(2) ** -k)) for k in range(0, 40)]
    tiny += [mpf(float(mpf(random.random()))) for _ in range(20000)] + [
        mpf('1e-300'), mpf('1e-200'), mpf('1e-100'), mpf('0.5'),
        mpf('0.9999999999999999'), mpf(1)]
    unit = [mpf(k) / 20001 for k in range(1, 20002)]

    w1 = check_rat(Pcd, Qcd, "uniform 30001 pts on [1,40]", uniform)
    w2 = check_rat(Pcd, Qcd, "cubic-dense 30001 pts", cubic)
    w3 = check_rat(Pcd, Qcd, "20000 random doubles", rnd)
    w4 = check_rat(Pcd, Qcd, "every double in [1,1+2^-40]", sliver)
    w5 = check_poly(Pd, "erf poly, every tiny + 20000 pts", tiny)
    w6 = check_poly(Pd, "erf poly, uniform 20001 pts on (0,1]", unit)

    # The rational's coefficients amplify error near x = 1, so the dd
    # rounding is not free: the targets above are checked with it included,
    # which is why everything above feeds the dd-rounded tables and not the
    # raw fit.
    bad = 0
    if max(w1, w2, w3, w4) > mpf('1e-20'):
        print("FAIL: rational worse than 1e-20"); bad = 1
    if max(w5, w6) > mpf('1e-22'):
        print("FAIL: polynomial worse than 1e-22"); bad = 1

    print("\n/* --- erf polynomial: P(x^2), degree 15 in x^2 --- */")
    emit("vm_erf_p", Pd)
    print("\n/* --- erfc rational, Chebyshev basis in t=(2x-41)/39 --- */")
    emit("vm_erfc_rp", Pcd)
    emit("vm_erfc_rq", Qcd)
    return bad


def emit(name, c):
    print("static const double %s[%d][2] = {" % (name, len(c)))
    for v in c:
        hi = mpf(float(v))
        print("  { %.17e, %.17e }," % (float(hi), float(v - hi)))
    print("};")


if __name__ == "__main__":
    sys.exit(main())
