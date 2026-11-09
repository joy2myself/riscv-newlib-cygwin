#!/bin/sh
# domain-check.sh -- accuracy gate restricted to a homogeneous domain.
#
# The FULL blobs mix special values, tiny and huge arguments with the
# ordinary range.  A kernel whose fast path is only accurate on part of its
# range can therefore still show a small aggregate max_ulp, because the
# dangerous lanes are routed to the guarded fallback.  This script filters
# the blobs down to the benchmark domain of each function (see
# bench/bench.c domain_lo/domain_hi) and re-runs the accuracy runner, so the
# fast path is the only path exercised.
#
# Usage: scripts/domain-check.sh TAG [--vlen N] [--report FILE]
#        [--func NAME ...]
#
# Exits 0 only if every checked entry stays within 1 ulp with no
# failure/flag/sweep counts.  Scratch blobs live in .build/TAG-domain/.
set -e

TESTDIR=$(cd "$(dirname "$0")/.." && pwd)
QEMU=${QEMU:-$HOME/qemu-build/qemu-riscv64}
TAG=${1:?usage: domain-check.sh TAG}
shift || true

VLEN=256
REPORT=
FUNCS=
while [ $# -gt 0 ]; do
  case "$1" in
    --vlen) VLEN=$2; shift 2;;
    --report) REPORT=$2; shift 2;;
    --func) FUNCS="$FUNCS $2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done

BUILD=$TESTDIR/.build/$TAG-domain
rm -rf "$BUILD"; mkdir -p "$BUILD"

# Function, then the fast-path domain as "lo hi [field]" (both inputs for
# two-argument entries).  Mirrors bench.c's domain_lo/domain_hi.
domains() {
  cat <<'EOF'
sqrt 0.5 1
rsqrt 0.5 1
cbrt 0.5 1
pow 0.5 1 both
hypot -10 10 both
exp -10 10
exp2 -10 10
exp10 -20 20
expm1 -10 10
log 0.5 1
log2 0.5 1
log10 0.5 1
log1p -0.9 10
log2p1 -0.9 10
sin -10 10
cos -10 10
tan -10 10
sincos -10 10
sinpi -10 10
cospi -10 10
tanpi -10 10
sincospi -10 10
asin -1 1
acos -1 1
atan -10 10
atan2 -10 10 both
asinpi -1 1
acospi -1 1
atanpi -10 10
atan2pi -10 10 both
sinh -10 10
cosh -10 10
tanh -10 10
asinh -10 10
acosh 1 10
atanh -0.9 0.9
erf -10 10
erfc -10 10
tgamma 0.5 5
lgamma 0.5 5
lgamma_r 0.5 5
sqrtf 0.5 1
rsqrtf 0.5 1
cbrtf 0.5 1
powf 0.5 1 both
hypotf -10 10 both
expf -10 10
exp2f -10 10
exp10f -20 20
expm1f -10 10
logf 0.5 1
log2f 0.5 1
log10f 0.5 1
log1pf -0.9 10
log2p1f -0.9 10
sinf -10 10
cosf -10 10
tanf -10 10
sincosf -10 10
sinpif -10 10
cospif -10 10
tanpif -10 10
sincospif -10 10
asinf -1 1
acosf -1 1
atanf -10 10
atan2f -10 10 both
asinpif -1 1
acospif -1 1
atanpif -10 10
atan2pif -10 10 both
sinhf -10 10
coshf -10 10
tanhf -10 10
asinhf -10 10
acoshf 1 10
atanhf -0.9 0.9
erff -10 10
erfcf -10 10
tgammaf 0.5 5
lgammaf 0.5 5
lgamma_rf 0.5 5
EOF
}

domains | while read -r nm lo hi field; do
  [ -n "$nm" ] || continue
  if [ -n "$FUNCS" ]; then
    case " $FUNCS " in *" $nm "*) : ;; *) continue ;; esac
  fi
  src=$TESTDIR/blobs/FULL/$nm.bin
  [ -f "$src" ] || continue
  if [ "$field" = "both" ]; then
    python3 "$TESTDIR/scripts/filter-blob.py" "$src" "$BUILD/$nm.bin" \
      "$lo" "$hi" --field both >/dev/null
  else
    python3 "$TESTDIR/scripts/filter-blob.py" "$src" "$BUILD/$nm.bin" \
      "$lo" "$hi" >/dev/null
  fi
done

n=$(ls "$BUILD"/*.bin 2>/dev/null | wc -l)
echo "domain-check: $n functions, VLEN=$VLEN"
set -- --blobs "$BUILD"
[ -n "$REPORT" ] && set -- "$@" --report "$REPORT"
"$QEMU" -cpu "rv64,v=true,vlen=$VLEN,vext_spec=v1.0" \
      "$TESTDIR/.build/$TAG/runner" "$@" 2>&1 | tee "$BUILD.log"
grep -qE ' [0-9]+ failed' "$BUILD.log" || exit 1
! grep -qE ' [1-9][0-9]* failed' "$BUILD.log"
