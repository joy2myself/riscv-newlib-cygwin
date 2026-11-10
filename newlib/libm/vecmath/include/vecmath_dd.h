/* vecmath_dd.h -- vector double-double (dd) arithmetic layer for libvecmath.

   Part of the newlib RISC-V vector math library.

   A dd number is a (hi, lo) vector PAIR with value hi+lo and the
   normalization invariant |lo| <= 2^-53*|hi|, giving ~106 bits of
   significand.  Used by erf/erfc/tgamma/lgamma and the *pi scaling paths
   to reach <= 1 ulp.

   API SHAPE -- why macros: RVV vector types are SIZELESS in GCC; they
   cannot appear as struct fields, array elements, or behind pointers
   ("fields cannot have RVV type").  dd values are therefore passed as
   pairs of variables and every dd op is a statement macro:

     vm_f_t rh, rl;
     VM_DD_MUL(rh, rl, ah, al, bh, bl, vl);

   MACRO HYGIENE: each dd macro kind declares its internal temporaries
   under its own fixed prefix (vmdda_/vmddm_/vmddb_/vmddq_/vmdds_), so
   nesting across kinds cannot capture arguments by shadowing.

   Requires vecmath_rvvcompat.h included first (VM_LMUL active);
   re-includable at a different VM_LMUL.

   Error-free transformations rely on fused ops:
     VFNMSAC(d, a, b) = d - a*b with a single rounding, which is exact
     for d = RN(a*b) -- the RVV two-product primitive.

   Flag note: lo-part arithmetic legitimately raises FE_INEXACT; only
   invalid/divbyzero/overflow/underflow are contractually meaningful for
   libvecmath entry points.  */

#ifndef VM_COMPAT_ACTIVE
#error "include vecmath_rvvcompat.h before vecmath_dd.h"
#endif

/* Parameter pasting is not usable here: nested invocations pass output
   arguments as parenthesized macro-call text, which is not a pasteable
   identifier.  Kernel code must not declare locals named vmdd[a-z]_*.  */

/* ---- error-free transformations ---------------------------------- */

/* Knuth two_sum: s+e == a+b exactly, any magnitudes.  */
#define VM_TWO_SUM(s, e, a, b, vl) do {					\
  (s) = VM_VFADD_VV ((a), (b), (vl));					\
  (e) = VM_VFADD_VV (VM_VFSUB_VV ((a),					\
				  VM_VFSUB_VV ((s),			\
					       VM_VFSUB_VV ((s), (a), (vl)), \
					       (vl)), (vl)),		\
		     VM_VFSUB_VV ((b), VM_VFSUB_VV ((s), (a), (vl)), (vl)), \
		     (vl));						\
} while (0)

/* Fast2Sum: requires |a| >= |b|.  */
/* ------------------------------------------------------------------ */
/* Division-free reciprocals.

   RVV 1.0 defines vfrec7 (a ~7-bit reciprocal estimate) and vfrsqrt7,
   both of which are single cheap instructions on the cores we target,
   while vfdiv.vv costs a full iterative pipeline: measured on the two
   boards we test, a dependent vfdiv chain is 50 ns on the VLEN 256 X60
   and 40 ns on the VLEN 128 board, whereas vfrec7 + three Newton steps
   (six dependent fmas) is 26 ns and 21 ns.  Two Newton steps are enough
   when the caller corrects the result with an fma residual, three when
   the reciprocal itself has to stand at ~half an ulp.  */

/* r = 1/x to ~2^-28 relative; only for callers that follow up with a
   residual (Newton) correction.  */
#define VM_RCP_FAST(r, x, vl) do {					\
  vm_f_t vrcp__e;							\
  (r) = VM_VFREC7_V ((x), (vl));					\
  vrcp__e = VM_VFNMSAC_VV (VM_VFMV_V_F (1.0, (vl)), (x), (r), (vl));	\
  (r) = VM_VFMACC_VV ((r), (r), vrcp__e, (vl));			\
  vrcp__e = VM_VFNMSAC_VV (VM_VFMV_V_F (1.0, (vl)), (x), (r), (vl));	\
  (r) = VM_VFMACC_VV ((r), (r), vrcp__e, (vl));			\
} while (0)

/* r = 1/x to ~2^-56 relative (about half an ulp of the reciprocal).  */
#define VM_RCP_FULL(r, x, vl) do {					\
  vm_f_t vrcp__e;							\
  VM_RCP_FAST (r, x, vl);						\
  vrcp__e = VM_VFNMSAC_VV (VM_VFMV_V_F (1.0, (vl)), (x), (r), (vl));	\
  (r) = VM_VFMACC_VV ((r), (r), vrcp__e, (vl));			\
} while (0)

/* q = n/d with the quotient corrected by its own fma residual: ~half an
   ulp, the same accuracy as a hardware divide, but built from a 7-bit
   estimate plus three Newton steps (measured fast on both boards).  */
#define VM_QDIV(q, n, d, vl) do {					\
  vm_f_t qd__rt, qd__q, qd__e;						\
  VM_RCP_FULL (qd__rt, (d), (vl));					\
  qd__q = VM_VFMUL_VV ((n), qd__rt, (vl));				\
  qd__e = VM_VFNMSAC_VV ((n), qd__q, (d), (vl));			\
  (q) = VM_VFMACC_VV (qd__q, qd__e, qd__rt, (vl));			\
} while (0)

/* Corrected double-double reciprocal: rh+rl = 1/x to ~2^-110.  */
#define VM_RCP_DD(rh, rl, x, vl) do {					\
  VM_RCP_FULL (rh, x, vl);						\
  (rl) = VM_VFMUL_VV ((rh),						\
    VM_VFMACC_VV (VM_VFMV_V_F (1.0, (vl)),				\
		  VM_VFNEG_V ((rh), (vl)), (x), (vl)), (vl));		\
} while (0)

#define VM_FAST2SUM(s, e, a, b, vl) do {				\
  (s) = VM_VFADD_VV ((a), (b), (vl));					\
  (e) = VM_VFADD_VV (VM_VFSUB_VV ((a), (s), (vl)), (b), (vl));		\
} while (0)

/* two_prod: p+e == a*b exactly.  */
#define VM_TWO_PROD(p, e, a, b, vl) do {				\
  (p) = VM_VFMUL_VV ((a), (b), (vl));					\
  (e) = VM_VFNEG_V (VM_VFNMSAC_VV ((p), (a), (b), (vl)), (vl));		\
} while (0)

/* ---- dd arithmetic over (hi, lo) variable pairs ------------------- */

#define VM_DD_NEG(rh, rl, ah, al, vl) do {				\
  (rh) = VM_VFNEG_V ((ah), (vl));					\
  (rl) = VM_VFNEG_V ((al), (vl));					\
} while (0)

/* Renormalize an arbitrary pair.  */
#define VM_DD_NORM(rh, rl, ah, al, vl) VM_TWO_SUM (rh, rl, ah, al, vl)

#define VM_DD_ADD(rh, rl, ah, al, bh, bl, vl) do {			\
  vm_f_t vmdda_s, vmdda_e, vmdda_t;					\
  VM_TWO_SUM (vmdda_s, vmdda_e, ah, bh, (vl));				\
  vmdda_t = VM_VFADD_VV (VM_VFADD_VV ((al), (bl), (vl)),		\
			 vmdda_e, (vl));				\
  VM_TWO_SUM ((rh), (rl), vmdda_s, vmdda_t, (vl));			\
} while (0)

#define VM_DD_SUB(rh, rl, ah, al, bh, bl, vl) do {			\
  VM_DD_ADD ((rh), (rl), (ah), (al),					\
	     VM_VFNEG_V ((bh), (vl)), VM_VFNEG_V ((bl), (vl)), (vl));	\
} while (0)

#define VM_DD_MUL(rh, rl, ah, al, bh, bl, vl) do {			\
  vm_f_t vmddm_p, vmddm_e, vmddm_t;					\
  VM_TWO_PROD (vmddm_p, vmddm_e, ah, bh, (vl));				\
  vmddm_t = VM_VFADD_VV (VM_VFMUL_VV ((ah), (bl), (vl)),		\
			 VM_VFMUL_VV ((al), (bh), (vl)), (vl));		\
  vmddm_e = VM_VFADD_VV (vmddm_e, vmddm_t, (vl));			\
  VM_FAST2SUM ((rh), (rl), vmddm_p, vmddm_e, (vl));			\
} while (0)

/* dd * double scalar.  */
#define VM_DD_MUL_D(rh, rl, ah, al, b, vl) do {				\
  vm_f_t vmddb_bv = VM_VFMV_V_F ((b), (vl));				\
  vm_f_t vmddb_p, vmddb_e, vmddb_t;					\
  VM_TWO_PROD (vmddb_p, vmddb_e, ah, vmddb_bv, (vl));			\
  vmddb_t = VM_VFMUL_VV ((al), vmddb_bv, (vl));				\
  vmddb_e = VM_VFADD_VV (vmddb_e, vmddb_t, (vl));			\
  VM_FAST2SUM ((rh), (rl), vmddb_p, vmddb_e, (vl));			\
} while (0)

#define VM_DD_ADD_D(rh, rl, ah, al, b, vl) do {				\
  VM_DD_ADD ((rh), (rl), (ah), (al),					\
	     VM_VFMV_V_F ((b), (vl)), VM_VFMV_V_F (0.0, (vl)), (vl));	\
} while (0)

/* One-Newton-step division; relative error ~2^-106.  */
#define VM_DD_DIV(rh, rl, ah, al, bh, bl, vl) do {			\
  vm_f_t vmddq_q = VM_VFDIV_VV ((ah), (bh), (vl));			\
  vm_f_t vmddq_ph, vmddq_pl;						\
  vm_f_t vmddq_z = VM_VFMV_V_F (0.0, (vl));				\
  VM_DD_MUL (vmddq_ph, vmddq_pl, (bh), (bl), vmddq_q, vmddq_z, (vl));	\
  VM_DD_SUB (vmddq_ph, vmddq_pl, (ah), (al), vmddq_ph, vmddq_pl, (vl));	\
  vm_f_t vmddq_q1 = VM_VFDIV_VV (vmddq_ph, (bh), (vl));			\
  VM_FAST2SUM ((rh), (rl), vmddq_q, vmddq_q1, (vl));			\
} while (0)

/* sqrt of a normalized non-negative dd; relative error ~2^-106.  */
#define VM_DD_SQRT(rh, rl, ah, al, vl) do {				\
  vm_f_t vmdds_s = VM_VFSQRT_V ((ah), (vl));				\
  vm_f_t vmdds_p, vmdds_e, vmdds_t;					\
  VM_TWO_PROD (vmdds_p, vmdds_e, vmdds_s, vmdds_s, (vl));		\
  VM_DD_SUB (vmdds_p, vmdds_e, (ah), (al), vmdds_p, vmdds_e, (vl));	\
  vmdds_t = VM_VFDIV_VV (VM_VFDIV_VF (vmdds_p, 2.0, (vl)),		\
			 vmdds_s, (vl));				\
  VM_FAST2SUM ((rh), (rl), vmdds_s, vmdds_t, (vl));			\
} while (0)

/* Collapse a dd pair to one correctly-rounded double.  */
#define VM_DD_COLLAPSE(r, ah, al, vl) do {				\
  (r) = VM_VFADD_VV ((ah), (al), (vl));					\
} while (0)
