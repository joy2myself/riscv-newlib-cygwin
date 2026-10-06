#!/bin/sh
# run-board.sh -- build and run the libvecmath accuracy suite natively on a
# RISC-V Linux board over SSH (public-key login assumed).
#
# Usage: scripts/run-board.sh [muse-pi|k230|user@host] [--tier small|full]
#
# Steps:
#   0. RVV 1.0 probe on the board (10-line intrinsics hello-world compiled
#      with the board's gcc; SIGILL -> abort with instructions).
#   1. rsync the test tree + vecmath sources + blobs to the board.
#   2. Native build on the board (avoids cross-object/gcc-version mixing).
#   3. Run, fetch results back into results/board-<host>/.
set -e

BOARD="${1:-muse-pi}"
TIER="full"
if [ "$2" = "--tier" ] && [ -n "$3" ]; then TIER="$3"; fi

TESTDIR=$(cd "$(dirname "$0")/.." && pwd)
VECDIR=$(cd "$TESTDIR/.." && pwd)
SRCROOT=$(cd "$VECDIR/../.." && pwd)
REMOTE_DIR="/tmp/vecmath-test"
BLOBTIER=$(echo "$TIER" | tr '[:lower:]' '[:upper:]')

echo "== [0] RVV 1.0 probe on $BOARD =="
ssh "$BOARD" 'mkdir -p /tmp/vecmath-probe && cat > /tmp/vecmath-probe/p.c <<EOF
#include <riscv_vector.h>
#include <stdio.h>
int main(void){
  double x[4]={1,4,9,16},y[4];
  size_t vl=__riscv_vsetvl_e64m1(4);
  __riscv_vse64_v_f64m1(y,__riscv_vfsqrt_v_f64m1(__riscv_vle64_v_f64m1(x,vl),vl),vl);
  printf("probe ok %g\n",y[3]); return y[3]==4.0?0:1;
}
EOF
gcc -O2 -march=rv64gcv -mabi=lp64d -o /tmp/vecmath-probe/p /tmp/vecmath-probe/p.c \
  && /tmp/vecmath-probe/p'
# If the board gcc lacks -march=rv64gcv or the binary SIGILLs, ssh exits
# nonzero and set -e aborts here.

echo "== [1] sync sources to $BOARD:$REMOTE_DIR =="
ssh "$BOARD" "rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR"
# test framework + library sources + public header
(cd "$SRCROOT" && tar cf - \
    libm/vecmath \
    libc/include/vecmath.h) | ssh "$BOARD" "mkdir -p $REMOTE_DIR/newlib && tar xf - -C $REMOTE_DIR/newlib"

echo "== [2] generate blobs on host, sync =="
if [ "$TIER" = "full" ]; then BT=blobs/FULL; else BT=blobs/SMALL; fi
(cd "$TESTDIR" && make vmgen "blobs-$TIER")
(cd "$TESTDIR" && tar cf - blobs) | ssh "$BOARD" "tar xf - -C $REMOTE_DIR/newlib/libm/vecmath/test"

echo "== [3] native build + run on board =="
ssh "$BOARD" "cd $REMOTE_DIR/newlib/libm/vecmath/test \
  && make runner-linux CROSS_LINUX=gcc QEMU=true \
  && ./runner-linux --blobs $BT --report results-board-$BLOBTIER.md"

echo "== [4] fetch results =="
mkdir -p "$TESTDIR/results/board-$BOARD"
scp -q "$BOARD:$REMOTE_DIR/newlib/libm/vecmath/test/results-board-$BLOBTIER.md" \
       "$TESTDIR/results/board-$BOARD/" 2>/dev/null || true
echo "== done: results in $TESTDIR/results/board-$BOARD/ =="
