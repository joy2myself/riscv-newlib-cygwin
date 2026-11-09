#!/bin/sh
# acc.sh -- run the MPFR accuracy runner under qemu for one tag.
#
# Usage: scripts/acc.sh TAG [options]
#   --func NAME     only this function (binary64 name, e.g. "sin"; use
#                   "sinf" for the binary32 entry)
#   --tier small|full   blob tier (default small)
#   --vlen N        VLEN (default 256)
#   --report FILE   write the markdown report here
#   --no-build      use an existing .build/TAG/runner
#
# Environment: QEMU (default $HOME/qemu-build/qemu-riscv64).
set -e

TESTDIR=$(cd "$(dirname "$0")/.." && pwd)
QEMU=${QEMU:-$HOME/qemu-build/qemu-riscv64}

TAG=${1:?usage: acc.sh TAG [--func NAME] [--tier small|full] [--vlen N]}
shift || true

FUNC=
TIER=small
VLEN=256
REPORT=
BUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --func) FUNC=$2; shift 2;;
    --tier) TIER=$2; shift 2;;
    --vlen) VLEN=$2; shift 2;;
    --report) REPORT=$2; shift 2;;
    --no-build) BUILD=0; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done

[ "$BUILD" = 0 ] || "$TESTDIR/scripts/build.sh" "$TAG" runner >/dev/null
BLOBS=$TESTDIR/blobs/$(echo "$TIER" | tr '[:lower:]' '[:upper:]')
[ -d "$BLOBS" ] || { echo "no blobs at $BLOBS (run 'make blobs-$TIER' first)" >&2; exit 2; }

set -- --blobs "$BLOBS"
[ -n "$FUNC" ] && set -- "$@" --func "$FUNC"
[ -n "$REPORT" ] && set -- "$@" --report "$REPORT"

exec "$QEMU" -cpu "rv64,v=true,vlen=$VLEN,vext_spec=v1.0" \
     "$TESTDIR/.build/$TAG/runner" "$@"
