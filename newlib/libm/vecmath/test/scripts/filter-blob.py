#!/usr/bin/env python3
"""filter-blob.py -- restrict an existing MPFR blob to a homogeneous domain.

The accuracy framework's blobs deliberately mix special values, tiny and
huge arguments with the ordinary domain.  That is the right gate for a
library, but it can hide a fast path that is only accurate on a subset of
its range: if most records take the guarded fallback, the aggregate max_ulp
stays small.  This tool rewrites a blob keeping only the records whose
inputs fall inside a requested interval, so the fast path is the ONLY path
exercised.

Usage:
  filter-blob.py IN.bin OUT.bin LO HI [--prec 32|64] [--field in0|in1]
                 [--abs] [--keep-specials]

  LO / HI      decimal or hex float bounds, or "inf"
  --prec       element precision of the blob (default: read from header)
  --field      which input to test (default: in0; "both" requires both)
  --abs        compare |x| instead of x
  --keep-specials  keep records whose tested input is Inf/NaN regardless

The output keeps the original header (id, kind, flags_policy, name) and
sets nrec to the number of kept records; the runner then sees a blob whose
every lane lies in the requested domain.
"""
import argparse
import struct
import sys

VMB_MAGIC = 0x48544D56
HDR = struct.Struct("<IIII QQ II 16s 8s")  # 64 bytes

REC_SIZES = {48: 1, 72: 2, 80: 3}


def parse_bound(text):
    if text.lower() in ("inf", "+inf"):
        return float("inf")
    if text.lower() == "-inf":
        return float("-inf")
    return float.fromhex(text) if "0x" in text.lower() else float(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("lo")
    ap.add_argument("hi")
    ap.add_argument("--prec", type=int, default=0)
    ap.add_argument("--field", choices=("in0", "in1", "both"), default="in0")
    ap.add_argument("--abs", action="store_true")
    ap.add_argument("--keep-specials", action="store_true")
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        raw = f.read()
    if len(raw) < HDR.size:
        sys.exit("short blob")
    fields = HDR.unpack_from(raw, 0)
    (magic, ver, fid, rec_size, nrec, seed, kind, policy, name, reserved) = fields
    if magic != VMB_MAGIC:
        sys.exit("bad magic in %s" % args.src)
    if rec_size not in REC_SIZES:
        sys.exit("unexpected record size %d" % rec_size)

    prec = args.prec or (32 if reserved[0] == 32 else 64)
    lo, hi = parse_bound(args.lo), parse_bound(args.hi)

    def decode(bits):
        if prec == 32:
            return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]
        return struct.unpack("<d", struct.pack("<Q", bits & 0xFFFFFFFFFFFFFFFF))[0]

    def ok_one(bits):
        # poison marks an unused input
        if bits == 0xFFFFFFFFFFFFFFFF:
            return False
        v = decode(bits)
        if args.abs:
            v = abs(v)
        if v != v:  # NaN
            return args.keep_specials
        if v in (float("inf"), float("-inf")):
            return args.keep_specials
        return lo <= v <= hi

    kept = []
    off = HDR.size
    for _ in range(nrec):
        rec = raw[off:off + rec_size]
        off += rec_size
        in0, in1 = struct.unpack_from("<QQ", rec, 0)
        if args.field == "in0":
            ok = ok_one(in0)
        elif args.field == "in1":
            ok = ok_one(in1)
        else:
            ok = ok_one(in0) and ok_one(in1)
        if ok:
            kept.append(rec)

    out_hdr = HDR.pack(magic, ver, fid, rec_size, len(kept), seed, kind,
                       policy, name, reserved)
    with open(args.dst, "wb") as f:
        f.write(out_hdr)
        for rec in kept:
            f.write(rec)
    nm = name.split(b"\0")[0].decode(errors="replace")
    print("%s: kept %d/%d records in [%g, %g]" % (nm, len(kept), nrec, lo, hi))
    if not kept:
        sys.exit(1)


if __name__ == "__main__":
    main()
