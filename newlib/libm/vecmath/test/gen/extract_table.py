#!/usr/bin/env python3
"""extract_table.py -- byte-exact extraction of newlib libm data tables.

libvecmath ships SELF-CONTAINED copies of the optimized-routines tables
(__exp_data & friends) under its own symbol names (__vecmath_*), because
the libm symbol layouts differ between newlib and glibc (field order!)
and linking against whichever libm happens to be present would silently
corrupt the kernels.

This script reads the compiled newlib object (from any newlib build dir)
and emits the C initializer for tables/vm_*_tables.c, so the values are
always byte-identical to what scalar newlib libm uses.

Usage:
  extract_table.py <object.o> <symbol> <layout> <out.c> [--cross-prefix P]

Layouts are defined in LAYOUTS below (field name -> count/type pairs,
matching the struct declarations in newlib/libm/common/math_config.h
for rv64 hard-float: HAVE_FAST_FMA=1, EXP_TABLE_BITS=7, etc).
"""
import subprocess
import struct
import sys

LAYOUTS = {
    # struct exp_data (math_config.h): all little-endian IEEE doubles/u64.
    "exp_data": [
        ("invln2N", "d"), ("shift", "d"),
        ("negln2hiN", "d"), ("negln2loN", "d"),
        ("poly", "4d"), ("exp2_shift", "d"), ("exp2_poly", "5d"),
        ("tab", "256u"),
    ],
    "log_data": [
        ("ln2hi", "d"), ("ln2lo", "d"),
        ("poly", "5d"), ("poly1", "11d"),
        ("tab", "128p"),          # pairs {invc, logc}
    ],
    "log2_data": [
        ("invln2hi", "d"), ("invln2lo", "d"),
        ("poly", "6d"), ("poly1", "10d"),
        ("tab", "64p"),
    ],
    "pow_log_data": [
        ("ln2hi", "d"), ("ln2lo", "d"),
        ("poly", "7d"),
        ("tab", "128q"),          # quads {invc, pad, logc, logctail}
    ],
}


def sym_offset_size(obj, symbol, nm):
    out = subprocess.check_output([nm, "-S", obj], text=True)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[3] == symbol:
            return int(parts[0], 16), int(parts[1], 16)
    raise SystemExit(f"symbol {symbol} not found in {obj}")


def section_of(obj, addr, readelf):
    """Find section containing addr; return (file_offset_of_symbol)."""
    out = subprocess.check_output([readelf, "-S", "-W", obj], text=True)
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if "] ." in line or "] .rodata" in line:
            # format: [Nr] Name Type Address Off Size ...
            frag = line.split("] ", 1)[1].split()
            if len(frag) >= 5:
                name, _typ, vaddr, off, size = frag[0], frag[1], int(frag[2], 16), int(frag[3], 16), int(frag[4], 16)
                if vaddr <= addr < vaddr + size:
                    return off + (addr - vaddr)
    raise SystemExit("no section for address")


def dbl(x):
    """Exact C hex-float literal for a double."""
    return float.hex(x)


def emit(layout, blob, out, symbol, struct_name, source):
    fields = LAYOUTS[layout]
    pos = 0
    lines = []
    for name, spec in fields:
        kind = spec[-1]
        count = int(spec[:-1]) if spec[:-1].isdigit() else 1
        if kind == "d":
            vals = struct.unpack_from("<%dd" % count, blob, pos)
            pos += 8 * count
            if count == 1:
                lines.append(f"  .{name} = {dbl(vals[0])},")
            else:
                lines.append(f"  .{name} = {{")
                for j in range(0, count, 2):
                    chunk = ", ".join(dbl(v) for v in vals[j:j + 2])
                    lines.append(f"    {chunk},")
                lines.append("  },")
        elif kind == "u":
            vals = struct.unpack_from("<%dQ" % count, blob, pos)
            pos += 8 * count
            lines.append(f"  .{name} = {{")
            for j in range(0, count, 4):
                chunk = ", ".join("0x%016xULL" % v for v in vals[j:j + 4])
                lines.append(f"    {chunk},")
            lines.append("  },")
        elif kind == "p":       # {double a, b} pairs
            lines.append(f"  .{name} = {{")
            for j in range(count):
                a, b = struct.unpack_from("<2d", blob, pos)
                pos += 16
                lines.append(f"    {{ {dbl(a)}, {dbl(b)} }},")
            lines.append("  },")
        elif kind == "q":       # {double a,b,c,d} quads
            lines.append(f"  .{name} = {{")
            for j in range(count):
                a, b, c, d = struct.unpack_from("<4d", blob, pos)
                pos += 32
                lines.append(f"    {{ {dbl(a)}, {dbl(b)}, {dbl(c)}, {dbl(d)} }},")
            lines.append("  },")
    assert pos == len(blob), f"layout consumed {pos} of {len(blob)} bytes"
    with open(out, "w") as f:
        f.write(f"""/* {out.split('/')[-1]} -- GENERATED FILE, do not edit.

   Byte-exact extraction of `{symbol}` from the compiled newlib object,
   via libm/vecmath/test/gen/extract_table.py (layout '{layout}').
   Source of values: {source}.
   Regenerate whenever the newlib table changes:
     python3 extract_table.py <build>/libm/common/{source.replace('.c','')}.o \\
         {symbol} {layout} <this file>
*/

#include <vm_data.h>

const struct {struct_name} {symbol.replace('__exp_data','__vecmath_exp_data')
        .replace('__log_data','__vecmath_log_data')
        .replace('__log2_data','__vecmath_log2_data')
        .replace('__pow_log_data','__vecmath_pow_log_data')} = {{
""" + "\n".join(lines) + "\n};\n")
    print(f"wrote {out} ({len(blob)} bytes)")


def main():
    obj, symbol, layout, out = sys.argv[1:5]
    prefix = "riscv64-unknown-elf-"
    if "--cross-prefix" in sys.argv:
        prefix = sys.argv[sys.argv.index("--cross-prefix") + 1]
    off, size = sym_offset_size(obj, symbol, prefix + "nm")
    foff = section_of(obj, off, prefix + "readelf")
    with open(obj, "rb") as f:
        f.seek(foff)
        blob = f.read(size)
    struct_names = {"exp_data": "vm_exp_data", "log_data": "vm_log_data",
                    "log2_data": "vm_log2_data", "pow_log_data": "vm_pow_log_data"}
    sources = {"exp_data": "libm/common/exp_data.c",
               "log_data": "libm/common/log_data.c",
               "log2_data": "libm/common/log2_data.c",
               "pow_log_data": "libm/common/pow_log_data.c"}
    emit(layout, blob, out, symbol, struct_names[layout], sources[layout])


if __name__ == "__main__":
    main()
