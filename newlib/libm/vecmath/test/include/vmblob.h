/* vmblob.h -- binary test-blob format shared by the MPFR reference
   generator (host) and the accuracy runner (target).

   Part of the libvecmath test framework.  Little-endian, packed layouts;
   both sides are LP64 so the structs below are exchanged verbatim.  */

#ifndef VMBLOB_H
#define VMBLOB_H

#include <stdint.h>

#define VMB_MAGIC   0x48544d56u		/* "VMTH" */
#define VMB_VERSION 1u

/* Output kinds.  */
#define VMB_OUT_SINGLE    1u	/* one double out                       */
#define VMB_OUT_DUAL      2u	/* two doubles out (sincos)             */
#define VMB_OUT_DUAL_INT  3u	/* double + int out (lgamma_r)          */

/* Expected-value classes in rec.expect (bitmask).  */
#define VMB_EXPECT_NORMAL  0x00u	/* use ref_hi/ref_lo ulp metric     */
#define VMB_EXPECT_NAN     0x01u	/* result must be NaN               */
#define VMB_EXPECT_INF     0x02u	/* result must be +-Inf (sign in
					   exact_bits)                      */
#define VMB_EXPECT_EXACT   0x04u	/* result must be bitwise exact_bits
					   (covers +-0 and exactly
					   representable results)         */
#define VMB_NAN_QUIET      0x08u	/* with NAN: must be quiet          */
#define VMB_SIGN_STRICT    0x10u	/* with NAN/EXACT: sign bit must
					   match exact_bits               */
/* Second output expectations occupy bits 8..12 of expect (shifted).  */
#define VMB_EXPECT2_SHIFT  8
#define VMB_EXPECT2(e)     (((e) >> VMB_EXPECT2_SHIFT) & 0x1f)

/* Expected FP flags in rec.fpflags.  Values follow the RISC-V fcsr fflags
   bit order, which is what both newlib (machine/riscv/sys/fenv.h) and
   glibc (bits/fenv.h) use on riscv64:
   NX=0x01 UF=0x02 OF=0x04 DZ=0x08 NV=0x10.
   The runner static-asserts FE_* == VMB_F_* at compile time.
   INEXACT is advisory only (dd internals raise it legitimately).  */
#define VMB_F_INEXACT   0x01u	/* advisory */
#define VMB_F_UNDERFLOW 0x02u
#define VMB_F_OVERFLOW  0x04u
#define VMB_F_DIVBYZERO 0x08u
#define VMB_F_INVALID   0x10u
#define VMB_F_GATE      (VMB_F_INVALID | VMB_F_DIVBYZERO \
			 | VMB_F_OVERFLOW | VMB_F_UNDERFLOW)

/* Flag policy per function (header.flags_policy):
   bit0: gated flags are checked per record (else aggregate only)
   bit1: inexact must not be raised (default: advisory)  */
#define VMB_POLICY_FLAG_PER_REC 0x01u
#define VMB_POLICY_INEXACT_STRICT 0x02u

/* Header: exactly 64 bytes.
   Field sizes: 4+4+4+4+8+8+4+4 = 40; name[16] -> 56; pad to 64.
   The ulp threshold lives in the shared per-function table (vmfuncs.def)
   on both sides, not in the blob.  */
typedef struct {
  uint32_t magic;		/* VMB_MAGIC */
  uint32_t version;		/* VMB_VERSION */
  uint32_t func_id;		/* entry in vmfuncs.def */
  uint32_t rec_size;		/* bytes per record */
  uint64_t nrec;		/* record count */
  uint64_t seed;		/* generator seed (reproducibility) */
  uint32_t kind;		/* VMB_OUT_* */
  uint32_t flags_policy;	/* VMB_POLICY_* */
  char     name[16];		/* function name, zero-padded */
  uint8_t  reserved[8];
} vmb_hdr_t;

/* Single-output record: 48 bytes.  */
typedef struct {
  uint64_t in0;			/* input bits (poison 0xFFFF... if unused) */
  uint64_t in1;			/* second input bits (binary functions) */
  uint64_t ref_hi;		/* dd reference: correctly rounded part */
  uint64_t ref_lo;		/* dd reference: residual (hi+lo ~ exact) */
  uint64_t exact_bits;		/* bitwise result when EXPECT_EXACT /
				   sign source for INF/ZERO */
  uint32_t expect;		/* VMB_EXPECT_* */
  uint32_t fpflags;		/* VMB_F_* */
} vmb_rec1_t;

/* Dual-output record (sincos): 72 bytes.  */
typedef struct {
  uint64_t in0;
  uint64_t in1;
  uint64_t ref_hi, ref_lo;	/* first output */
  uint64_t exact_bits;
  uint64_t ref2_hi, ref2_lo;	/* second output */
  uint64_t exact2_bits;
  uint32_t expect;		/* second-output class in bits 8..12 */
  uint32_t fpflags;
} vmb_rec2_t;

/* Dual-output + int record (lgamma_r): 80 bytes.  */
typedef struct {
  vmb_rec2_t r2;
  int32_t expect_sign;		/* expected sign[] value */
  int32_t pad;
} vmb_rec2i_t;

#define VMB_POISON UINT64_C(0xffffffffffffffff)

#endif /* VMBLOB_H */
