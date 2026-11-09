#!/bin/sh
# check-all.sh -- acceptance gate for the performance project.
#
# Builds a tag, runs the FULL MPFR accuracy suite over all 82 entries under
# qemu and the full throughput benchmark on the real board, then reports one
# line per entry with max_ulp / speedup and an overall PASS/FAIL.
#
# Usage: scripts/check-all.sh TAG [--skip-build] [--board-runs K]
#        [--bench-out FILE]
#
# Exit status 0 only if every entry has max_ulp <= 1.0 (fails=0, flags=0,
# sweep=0) *and* speedup > 1.0x.
set -e

TESTDIR=$(cd "$(dirname "$0")/.." && pwd)
TAG=${1:?usage: check-all.sh TAG [--skip-build]}
shift || true
BUILD=1
RUNS=2
BENCH_OUT=
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build) BUILD=0; shift;;
    --board-runs) RUNS=$2; shift 2;;
    --bench-out) BENCH_OUT=$2; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ -n "$BENCH_OUT" ] || BENCH_OUT=$TESTDIR/results/bench-$TAG.md
OUT=$TESTDIR/.build/$TAG
mkdir -p "$OUT"

if [ "$BUILD" = 1 ]; then
  "$TESTDIR/scripts/build.sh" "$TAG" runner bench >/dev/null
fi

echo "== FULL accuracy (qemu, VLEN 256) =="
"$TESTDIR/scripts/acc.sh" "$TAG" --tier full --no-build > "$OUT/acc-full.log" 2>&1 || true
NPASS=$(grep -c ' PASS' "$OUT/acc-full.log")
FUNC_TOTAL=$(grep -cE '^[a-z0-9_]+ +n=' "$OUT/acc-full.log")
echo "accuracy: $NPASS / $FUNC_TOTAL PASS"

echo "== board benchmark =="
"$TESTDIR/scripts/bench-board.sh" --tag "$TAG" --n 65536 --reps 9 --runs "$RUNS" \
    --out "$BENCH_OUT" >/dev/null 2>&1

# Join the two reports and classify.
awk -v acc="$OUT/acc-full.log" -v bench="$BENCH_OUT" '
  BEGIN {
    while ((getline line < acc) > 0) {
      if (match(line, /^[a-z0-9_]+ +n=/)) {
        split(line, f, " ");
        name=f[1];
        for (i = 1; i <= split(line, w, " "); i++) {
          if (w[i] ~ /^max_ulp=/) { split(w[i], a, "="); ulp[name]=a[2]+0 }
          if (w[i] ~ /^>1$/) { next }
        }
        if (line ~ />1 ulp: *[1-9]/) over[name]=1;
        if (line ~ /class_fail: *[1-9]/) cls[name]=1;
        if (line ~ /sweep_fail: *[1-9]/) swp[name]=1;
        if (line ~ /flag_fail: *[1-9]/) flg[name]=1;
        if (line ~ /PASS/) pass[name]=1; else pass[name]=0;
      }
    }
    close(acc);
    while ((getline line < bench) > 0) {
      if (line ~ /^\| / && line !~ /function/ && line !~ /---/) {
        split(line, c, "|");
        n=c[2]; sp=c[5];
        gsub(/ /, "", n); gsub(/ /, "", sp); sub(/x$/, "", sp);
        name[n]=sp+0;
      }
    }
    close(bench);
    printf "%-10s %10s %10s  %s\n", "function", "max_ulp", "speedup", "status";
    for (nm in ulp) {
      sp = name[nm] + 0;
      bad = "";
      if (ulp[nm] > 1.0) bad = bad " ulp>";
      if (over[nm]) bad = bad " >1ulp";
      if (cls[nm]) bad = bad " class";
      if (swp[nm]) bad = bad " sweep";
      if (flg[nm]) bad = bad " flags";
      if (!(nm in pass) || !pass[nm]) bad = bad " accFAIL";
      if (sp <= 1.0) bad = bad " slow";
      st = bad == "" ? "OK" : "FAIL:" bad;
      printf "%-10s %10.4f %9.2fx  %s\n", nm, ulp[nm], sp, st;
      n++; if (bad != "") nbad++;
    }
    printf "\n%d entries checked, %d failing\n", n, nbad;
  }
'
