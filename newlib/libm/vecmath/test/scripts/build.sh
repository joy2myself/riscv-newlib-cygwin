#!/bin/sh
# build.sh -- build the libvecmath accuracy runner and/or throughput bench.
#
# Usage: scripts/build.sh TAG [runner] [bench]
#        (no target given -> bench)
#
#   TAG names a per-invocation object directory under test/.build/, so
#   several people/agents can build concurrently from the same source tree
#   without clobbering each other.
#
# Environment:
#   CROSS_LINUX  cross compiler (default /opt/riscv/bin/riscv64-unknown-linux-gnu-gcc)
#   ARCH         -march/-mabi flags (default "-march=rv64gcv -mabi=lp64d")
#   JOBS         parallel compilation jobs (default 6)
#
# Binaries land in .build/TAG/{runner,bench}.
set -e

TESTDIR=$(cd "$(dirname "$0")/.." && pwd)
VECDIR=$(cd "$TESTDIR/.." && pwd)
SRCROOT=$(cd "$VECDIR/../.." && pwd)

TAG=${1:?usage: build.sh TAG [runner] [bench]}
shift || true
TARGETS=${*:-bench}

CROSS_LINUX=${CROSS_LINUX:-/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc}
ARCH=${ARCH:--march=rv64gcv -mabi=lp64d}
JOBS=${JOBS:-6}
DEFS="-DVECMATH_STANDALONE"
INCS="-I$TESTDIR/include -I$TESTDIR/runner -I$VECDIR/include -I$VECDIR -I$TESTDIR/shim"
OUT=$TESTDIR/.build/$TAG

mkdir -p "$OUT"
if [ ! -e "$TESTDIR/shim/vecmath.h" ]; then
  mkdir -p "$TESTDIR/shim"
  ln -sf "$SRCROOT/libc/include/vecmath.h" "$TESTDIR/shim/vecmath.h"
fi

SRCS="$(ls "$VECDIR"/v*.c) $(ls "$VECDIR"/tables/*.c)"
WANT_BENCH=
WANT_RUNNER=
case " $TARGETS " in *" bench "*)  SRCS="$SRCS $TESTDIR/bench/bench.c"; WANT_BENCH=1;; esac
case " $TARGETS " in *" runner "*) SRCS="$SRCS $TESTDIR/runner/runner.c $TESTDIR/runner/ulp.c"; WANT_RUNNER=1;; esac

echo "build[$TAG]: $(echo $SRCS | wc -w) TUs, jobs=$JOBS -> $OUT"
export CROSS_LINUX ARCH DEFS INCS OUT
printf '%s\n' $SRCS | xargs -P "$JOBS" -n 1 sh -c '
  src="$0"
  obj="$OUT/$(basename "$src" .c).o"
  $CROSS_LINUX -O2 $ARCH $DEFS $INCS -c "$src" -o "$obj" 2>"$obj.log" || {
    echo "FAILED: $src" >&2; cat "$obj.log" >&2; exit 1; }
'

if [ -n "$WANT_BENCH" ]; then
  
  $CROSS_LINUX -static $ARCH -o "$OUT/bench" $OUT/bench.o $OUT/v*.o -lm
  echo "build[$TAG]: bench -> $OUT/bench"
fi
if [ -n "$WANT_RUNNER" ]; then
  
  $CROSS_LINUX -static $ARCH -o "$OUT/runner" $OUT/runner.o $OUT/ulp.o $OUT/v*.o -lm
  echo "build[$TAG]: runner -> $OUT/runner"
fi
