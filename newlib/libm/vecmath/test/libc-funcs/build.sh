#!/bin/sh
# Build the RVV string/memory correctness and throughput harnesses.
#
# Environment:
#   CC        cross compiler        (default /opt/riscv/bin/riscv64-unknown-linux-gnu-gcc)
#   OBJCOPY   objcopy               (default: derived from CC)
#   NM        nm                    (default: derived from OBJCOPY)
#   ARCH      target flags          (default "-march=rv64gcv -mabi=lp64d")
#   ARCH_NOV  non-vector flags for the scalar fallbacks
#                                   (default "-march=rv64gc -mabi=lp64d")
#   OPT       optimisation flags    (default "-O2 -fno-builtin -Wall -Wextra")
#   SRC       newlib/libc/machine/riscv source directory
#   BUILD     output directory      (default <here>/build)
#
# The newlib sources are used completely unmodified: each *-asm.S is
# assembled and its entry symbol renamed with
#     objcopy --redefine-sym strlen=rvv_strlen ...
# so the harness can call both the vector routine and the platform (glibc)
# routine of the same name in one binary.  The same trick works with the
# board toolchains (gcc 14.2 / gcc 15).
set -e

here=$(cd "$(dirname "$0")" && pwd)

srcdir=${SRC:-$(cd "$here/../../../../libc/machine/riscv" && pwd)}
CC=${CC:-/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc}
case "$CC" in
  *gcc)  _objcopy=${CC%gcc}objcopy ;;
  *)     _objcopy=objcopy ;;
esac
OBJCOPY=${OBJCOPY:-$_objcopy}
NM=${NM:-${OBJCOPY%objcopy}nm}
ARCH=${ARCH:--march=rv64gcv -mabi=lp64d}
ARCH_NOV=${ARCH_NOV:--march=rv64gc -mabi=lp64d}
OPT=${OPT:--O2 -fno-builtin -Wall -Wextra}
BUILD=${BUILD:-$here/build}
STUB=$here/scripts/include

if [ ! -d "$srcdir" ]; then
  echo "error: source directory $srcdir not found" >&2
  exit 2
fi
mkdir -p "$BUILD/obj" "$BUILD/scalar"
rm -f "$BUILD"/obj/*.o "$BUILD"/scalar/*.o

# ------------------------------------------------------------------ #
# 1. Assemble the RVV routines and rename their entry points.          #
# ------------------------------------------------------------------ #
echo "== assembling RVV routines from $srcdir"
asm_count=0
for f in "$srcdir"/*-asm.S "$srcdir"/memset.S; do
  [ -f "$f" ] || continue
  b=$(basename "$f" .S)
  $CC -c $ARCH $OPT -I"$srcdir" "$f" -o "$BUILD/obj/$b.o"
  syms=$($NM -g --defined-only "$BUILD/obj/$b.o" | awk '{print $3}')
  if [ -z "$syms" ]; then
    echo "   $b: no global symbol (vector path disabled?)" >&2
    continue
  fi
  main_sym=$(echo "$syms" | head -n 1)
  set --
  for s in $syms; do
    set -- "$@" --redefine-sym "$s=rvv_$s"
  done
  $OBJCOPY "$@" "$BUILD/obj/$b.o" "$BUILD/obj/rvv_$main_sym.o"
  asm_count=$((asm_count + 1))
done
echo "   $asm_count objects -> $BUILD/obj"

# ------------------------------------------------------------------ #
# 2. Best-effort build of the scalar (non-vector) fallbacks.          #
#    Several of them include newlib-private headers, so failures are   #
#    not fatal; the driver only uses the ones that link.               #
# ------------------------------------------------------------------ #
echo "== building scalar fallbacks (best effort)"
scalar_objs=""
scalar_count=0
for f in "$srcdir"/*.c "$srcdir"/strcmp.S; do
  [ -f "$f" ] || continue
  b=$(basename "$f"); b=${b%.*}
  case "$b" in ffs|ieeefp|setjmp|stpcpy) continue ;; esac
  if $CC -c $ARCH_NOV $OPT -I"$srcdir" -I"$STUB" "$f" \
        -o "$BUILD/scalar/$b.o" 2>"$BUILD/scalar/$b.log"; then
    syms=$($NM -g --defined-only "$BUILD/scalar/$b.o" | awk '{print $3}')
    if [ -n "$syms" ]; then
      set --
      for s in $syms; do
        set -- "$@" --redefine-sym "$s=scalar_$s"
      done
      main_sym=$(echo "$syms" | head -n 1)
      $OBJCOPY "$@" "$BUILD/scalar/$b.o" "$BUILD/scalar/scalar_$main_sym.o"
      scalar_objs="$scalar_objs $BUILD/scalar/scalar_$main_sym.o"
      scalar_count=$((scalar_count + 1))
    fi
  fi
done
echo "   $scalar_count scalar objects"

# ------------------------------------------------------------------ #
# 3. Link.                                                            #
# ------------------------------------------------------------------ #
link_corr () {
  # shellcheck disable=SC2086
  $CC $ARCH $OPT -static -o "$BUILD/corr" "$here"/corr/*.c \
      "$BUILD"/obj/rvv_*.o "$@"
}

echo "== linking $BUILD/corr"
if ! link_corr $scalar_objs 2>"$BUILD/link-corr.log"; then
  echo "   note: link with scalar fallbacks failed, retrying without them"
  if ! link_corr 2>"$BUILD/link-corr.log"; then
    echo "error: link failed:" >&2
    cat "$BUILD/link-corr.log" >&2
    exit 1
  fi
fi

echo "== linking $BUILD/bench"
$CC $ARCH $OPT -static -o "$BUILD/bench" "$here/bench/bench.c" \
    "$BUILD/obj/rvv_memcpy.o" "$BUILD/obj/rvv_memset.o" \
    "$BUILD/obj/rvv_memmove.o" "$BUILD/obj/rvv_memcmp.o" \
    "$BUILD/obj/rvv_strlen.o" "$BUILD/obj/rvv_strcmp.o" \
    "$BUILD/obj/rvv_strchr.o" "$BUILD/obj/rvv_strstr.o"

echo "== done: $BUILD/corr $BUILD/bench"
QEMU=${QEMU:-$HOME/qemu-build/qemu-riscv64}
if [ -x "$QEMU" ]; then
  echo "   next: QEMU=$QEMU ./run.sh"
else
  echo "   note: qemu-riscv64 not found at $QEMU; set QEMU or use QEMU=native"
fi
