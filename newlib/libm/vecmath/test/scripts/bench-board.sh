#!/bin/sh
# bench-board.sh -- run the libvecmath throughput benchmark on a real RVV
# board over SSH, with a board-side lock so concurrent runs serialize.
#
# Usage: scripts/bench-board.sh [options]
#   --tag TAG       build tag (default bench); results go to results/bench-TAG.md
#   --host HOST     ssh board (default muse-pi)
#   --core N        core to pin to (default 6)
#   --func NAME     benchmark one function
#   --f32 | --f64   limit to one precision
#   --n N           elements per pass (default 65536)
#   --reps R        best-of repetitions inside one run (default 9)
#   --runs K        whole-benchmark repeats; the best speedup per function is
#                   kept (default 3)
#   --out FILE      markdown output (default test/results/bench-TAG.md)
#   --no-build      reuse .build/TAG/bench
#
# Environment: HOST can also be set instead of --host.
set -e

TESTDIR=$(cd "$(dirname "$0")/.." && pwd)

TAG=bench
HOST=${HOST:-muse-pi}
CORE=6
FUNC=
PREC=
N=65536
REPS=9
RUNS=3
OUT=
BUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG=$2; shift 2;;
    --host) HOST=$2; shift 2;;
    --core) CORE=$2; shift 2;;
    --func) FUNC=$2; shift 2;;
    --f32) PREC=--f32; shift;;
    --f64) PREC=--f64; shift;;
    --n) N=$2; shift 2;;
    --reps) REPS=$2; shift 2;;
    --runs) RUNS=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --no-build) BUILD=0; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ -n "$OUT" ] || OUT=$TESTDIR/results/bench-$TAG.md
mkdir -p "$(dirname "$OUT")" "$TESTDIR/.build/$TAG"

if [ "$BUILD" = 1 ]; then
  "$TESTDIR/scripts/build.sh" "$TAG" bench >/dev/null
fi
[ -x "$TESTDIR/.build/$TAG/bench" ] || { echo "no bench binary; build failed?" >&2; exit 1; }

REMOTE=/tmp/vecmath-bench-$TAG-$$
scp -q "$TESTDIR/.build/$TAG/bench" "$HOST:$REMOTE"
trap 'ssh "$HOST" "rm -f $REMOTE" 2>/dev/null || true' EXIT

ARGS="--n $N --reps $REPS --csv"
[ -n "$FUNC" ] && ARGS="$ARGS --func $FUNC"
[ -n "$PREC" ] && ARGS="$ARGS $PREC"

RAW=$TESTDIR/.build/$TAG/bench-raw-$$.csv
: > "$RAW"
i=0
while [ "$i" -lt "$RUNS" ]; do
  i=$((i + 1))
  echo "run $i/$RUNS on $HOST (core $CORE)..." >&2
  # flock serializes concurrent benchmark runs; taskset pins a core so other
  # board activity and asymmetric cache state do not move the baseline.
  ssh "$HOST" "flock -w 1800 /tmp/vecmath-bench.lock taskset -c $CORE $REMOTE $ARGS" \
      >> "$RAW"
done

# Keep, per function, the row with the best speedup over all runs.
MERGED=$TESTDIR/.build/$TAG/bench-merged-$$.csv
awk -F, '
  /^[A-Za-z_][A-Za-z0-9_]*,/ && $1 != "func" {
    name=$1; gsub(/\r/,"",$4); sp=$4+0;
    if (!(name in best) || sp > bestsp[name]) { best[name]=$0; bestsp[name]=sp; }
    if (!(name in seen)) { seen[name]=1; order[++n]=name; }
  }
  END { for (i = 1; i <= n; i++) print best[order[i]]; }
' "$RAW" > "$MERGED"

awk -F, -v out="$OUT" '
  BEGIN {
    print "| function | scalar ns/e | vector ns/e | speedup |" > out
    print "|---|---|---|---|" > out
  }
  {
    name=$1; gsub(/\r/,"",name); sc=$2; vc=$3; sp=$4; gsub(/\r/,"",sp);
    printf "| %s | %s | %s | %sx |\n", name, sc, vc, sp >> out
    n++; tot += sp + 0;
    if (sp + 0 <= 1.0) { slow = slow " " name "(=" sp "x)"; nslow++ }
  }
  END {
    printf "\n%d entries, mean speedup %.2fx, %d at or below 1.0x\n", \
      n, tot / n, nslow >> out
    if (nslow) printf "below 1.0x:%s\n", slow >> out
    printf "%d entries, mean speedup %.2fx, %d at or below 1.0x\n", \
      n, tot / n, nslow
    if (nslow) printf "below 1.0x:%s\n", slow
  }
' "$MERGED"

rm -f "$RAW" "$MERGED"
echo "report: $OUT" >&2
