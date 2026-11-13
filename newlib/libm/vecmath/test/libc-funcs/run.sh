#!/bin/sh
# Run the correctness harness at VLEN 128/256/512 and then the throughput
# harness, under qemu-riscv64 user mode by default.
#
# Environment:
#   QEMU    qemu-riscv64 binary; set QEMU= (empty) or QEMU=native to run the
#           binaries directly (real hardware), and QEMU_ARGS to add options.
#   VLENS   space separated list of VLEN values (default "128 256 512")
#   BUILD   build directory (default <here>/build)
#   CORR_ARGS  extra arguments for ./corr (function names to restrict to)
#
# Exit status: non-zero if any correctness run reported a failure.
set -e

here=$(cd "$(dirname "$0")" && pwd)
QEMU=${QEMU-$HOME/qemu-build/qemu-riscv64}
QEMU_ARGS=${QEMU_ARGS:-}
VLENS=${VLENS:-128 256 512}
BUILD=${BUILD:-$here/build}
RESULTS=${RESULTS:-$here/results}
CORR_ARGS=${CORR_ARGS:-}

extras=
if [ -n "$CORR_ARGS" ]; then
  extras=$CORR_ARGS
fi

for b in "$BUILD/corr" "$BUILD/bench"; do
  if [ ! -x "$b" ]; then
    echo "error: $b not built; run ./build.sh first" >&2
    exit 2
  fi
done

mkdir -p "$RESULTS"

run_qemu () {			# $1 = vlen, rest = program + args
  v=$1; shift
  if [ -z "$QEMU" ] || [ "$QEMU" = native ]; then
    "$@"
  else
    # shellcheck disable=SC2086
    $QEMU $QEMU_ARGS -cpu "rv64,v=true,vlen=$v" "$@"
  fi
}

rc=0
for v in $VLENS; do
  echo
  echo "================ correctness, VLEN $v ================"
  # shellcheck disable=SC2086
  if run_qemu "$v" "$BUILD/corr" $extras >"$RESULTS/corr-vlen$v.txt" 2>&1; then
    :
  else
    rc=1
  fi
  cat "$RESULTS/corr-vlen$v.txt"
done

for v in $VLENS; do
  echo
  echo "================ throughput, VLEN $v ================"
  # shellcheck disable=SC2086
  run_qemu "$v" "$BUILD/bench" --csv "$RESULTS/bench-vlen$v.csv" \
      >"$RESULTS/bench-vlen$v.txt" 2>&1
  cat "$RESULTS/bench-vlen$v.txt"
done

echo
echo "== results in $RESULTS"
exit $rc
