import argparse
from pathlib import Path
import subprocess
p = argparse.ArgumentParser(description="Assemble the preserved A210 candidate from pre-renamed archives")
p.add_argument("--rvv-archive", required=True, type=Path)
p.add_argument("--scalar-array-archive", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--cc", default="gcc")
p.add_argument("--ar", default="ar")
a = p.parse_args()
root = Path(__file__).resolve().parent
rvv = a.rvv_archive.resolve(strict=True)
scalar = a.scalar_array_archive.resolve(strict=True)
out = a.output.resolve()
if out.exists():
    p.error("output must be a new directory")
if any(any(c.isspace() for c in str(x)) for x in [rvv, scalar, out]):
    p.error("ar MRI paths must not contain whitespace")
out.mkdir(parents=True)
obj = out / "fallback-wrappers.o"
subprocess.run([a.cc, "-O2", "-no-pie", "-fno-builtin", "-fno-tree-vectorize", "-march=rv64gcv", "-I" + str(root), "-c", str(root / "fallback-wrappers.c"), "-o", str(obj)], check=True)
archive = out / "libvecmath.a"
mri = f"CREATE {archive}\nADDLIB {rvv}\nADDLIB {scalar}\nADDMOD {obj}\nSAVE\nEND\n"
subprocess.run([a.ar, "-M"], input=mri, text=True, check=True)
print(archive)
