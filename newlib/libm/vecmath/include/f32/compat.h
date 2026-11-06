/* f32/compat.h -- LMUL-generic RVV 1.0 intrinsic wrapper layer for the
   single-precision (binary32) libvecmath kernels.

   Part of the newlib RISC-V vector math library (libvecmath).

   This is the binary32 twin of <vecmath_rvvcompat.h>: the same
   RE-INCLUDABLE design (the includer defines VM32_LMUL before each
   inclusion; every macro is #undef'ed again on re-inclusion, so a kernel
   template can be instantiated once per LMUL from a single source of
   truth), the same naming rules, but every type and intrinsic is the
   SEW=32 one.  Everything is scoped under a VM32_ prefix so a
   translation unit may include this header AND the binary64 layer with
   no macro collisions (vfabi32.c does exactly that).

   Usage:
     #define VM32_LMUL 2
     #include <f32/compat.h>
     ... use vm32_f_t, VM32_VFADD_VV(a, b, vl), VM32_KFN(sin)(x, vl) ...

   If VM32_COMPAT_UNDEF_ONLY is defined before inclusion, only the #undef
   pass runs (used to clean up after the last instantiation).

   Mask types: for SEW=32 the mask element count of a vector group with
   LMUL K gives vbool(32/K)_t -- b32/b16/b8/b4 for K = 1/2/4/8.

   Kernel names are __vecmath_<name>_f32mK, the binary32 counterparts of
   the binary64 __vecmath_<name>_f64mK.  */

#ifndef VM32_LMUL
# if !defined(VM32_COMPAT_UNDEF_ONLY)
#  error "define VM32_LMUL (1,2,4,8) before including f32/compat.h"
# endif
# define VM32_LMUL 1
#endif

/* ------------------------------------------------------------------ */
/* Undef pass (runs on every inclusion).                              */
/* ------------------------------------------------------------------ */
#ifdef VM32_COMPAT_ACTIVE

#undef vm32_f_t
#undef vm32_b_t
#undef vm32_u_t
#undef vm32_i_t
#undef VM32_BSUF
#undef VM32_VSETVL
#undef VM32_VSETVLMAX
#undef VM32_KFN

#undef VM32_VLE
#undef VM32_VSE
#undef VM32_VLE_U
#undef VM32_VLE_I
#undef VM32_VSE_U
#undef VM32_VSE_I
#undef VM32_VLUXEI32
#undef VM32_VLUXEI32_U

#undef VM32_VFADD_VV
#undef VM32_VFADD_VF
#undef VM32_VFSUB_VV
#undef VM32_VFSUB_VF
#undef VM32_VFRSUB_VF
#undef VM32_VFMUL_VV
#undef VM32_VFMUL_VF
#undef VM32_VFDIV_VV
#undef VM32_VFDIV_VF
#undef VM32_VFRDIV_VF
#undef VM32_VFMACC_VV
#undef VM32_VFMACC_VF
#undef VM32_VFNMACC_VV
#undef VM32_VFNMACC_VF
#undef VM32_VFMSAC_VV
#undef VM32_VFMSAC_VF
#undef VM32_VFNMSAC_VV
#undef VM32_VFNMSAC_VF
#undef VM32_VFSQRT_V
#undef VM32_VFABS_V
#undef VM32_VFNEG_V
#undef VM32_VFSGNJ_VV
#undef VM32_VFSGNJX_VV
#undef VM32_VFSGNJN_VV
#undef VM32_VFMIN_VV
#undef VM32_VFMIN_VF
#undef VM32_VFMAX_VV
#undef VM32_VFMAX_VF
#undef VM32_VFCLASS_V
#undef VM32_VFMV_V_F
#undef VM32_VFMV_F_S
#undef VM32_VFMERGE_VFM
#undef VM32_VFMERGE_VVM

#undef VM32_VFMUL_VV_M
#undef VM32_VFMUL_VF_M
#undef VM32_VFADD_VV_M
#undef VM32_VFADD_VF_M
#undef VM32_VFSUB_VV_M
#undef VM32_VFSUB_VF_M
#undef VM32_VFDIV_VF_M
#undef VM32_VFRDIV_VF_M
#undef VM32_VFSQRT_V_M
#undef VM32_VFMACC_VF_M
#undef VM32_VFMACC_VV_M
#undef VM32_VFNMSAC_VV_M
#undef VM32_VFMAX_VV_M
#undef VM32_VFMERGE_VVM_TU
#undef VM32_FN_MU
#undef VM32_VFMUL_VV_MU
#undef VM32_VFMUL_VF_MU
#undef VM32_VFADD_VV_MU
#undef VM32_VFADD_VF_MU
#undef VM32_VFSUB_VV_MU
#undef VM32_VFSUB_VF_MU
#undef VM32_VFDIV_VV_MU
#undef VM32_VFDIV_VF_MU
#undef VM32_VFRDIV_VF_MU
#undef VM32_VFSQRT_V_MU
#undef VM32_VFMAX_VV_MU
#undef VM32_VFMIN_VV_MU

#undef VM32_VMFEQ_VV
#undef VM32_VMFEQ_VF
#undef VM32_VMFNE_VV
#undef VM32_VMFNE_VF
#undef VM32_VMFLT_VV
#undef VM32_VMFLT_VF
#undef VM32_VMFLE_VV
#undef VM32_VMFLE_VF
#undef VM32_VMFGT_VF
#undef VM32_VMFGE_VF

#undef VM32_VFCVT_X_F_V
#undef VM32_VFCVT_X_F_V_RM
#undef VM32_VFCVT_RTZ_X_F_V
#undef VM32_VFCVT_F_X_V
#undef VM32_VFCVT_F_X_V_RM
#undef VM32_VFCVT_XU_F_V_RM
#undef VM32_VFCVT_F_XU_V

#undef VM32_VADD_VX
#undef VM32_VADD_VV
#undef VM32_VSUB_VX
#undef VM32_VSUB_VV
#undef VM32_VMUL_VX
#undef VM32_VSLL_VX
#undef VM32_VSRL_VX
#undef VM32_VSRL_VV
#undef VM32_VSRA_VX
#undef VM32_VAND_VX
#undef VM32_VAND_VV
#undef VM32_VOR_VX
#undef VM32_VOR_VV
#undef VM32_VXOR_VX
#undef VM32_VXOR_VV
#undef VM32_VMSEQ_VX
#undef VM32_VMSNE_VX
#undef VM32_VMSLT_VX
#undef VM32_VMSLTU_VX
#undef VM32_VMSLT_VV
#undef VM32_VMSLTU_VV
#undef VM32_VMSLE_VX
#undef VM32_VMSLEU_VX
#undef VM32_VMSGT_VX
#undef VM32_VMSGTU_VX
#undef VM32_VMSGE_VX
#undef VM32_VMSGEU_VX
#undef VM32_VIOTA_M
#undef VM32_VMV_V_X
#undef VM32_VMV_V_V
#undef VM32_VMV_X_S
#undef VM32_VMERGE_VXM
#undef VM32_VMERGE_VVM
#undef VM32_VSLIDEDOWN_VX
#undef VM32_VSLIDEUP_VX

#undef VM32_VFIRST_M
#undef VM32_VMSBF_M
#undef VM32_VMSIF_M
#undef VM32_VMSOF_M
#undef VM32_VCMP_POPC_M
#undef VM32_VCOMPRESS
#undef VM32_VMAND_MM
#undef VM32_VMOR_MM
#undef VM32_VMXOR_MM
#undef VM32_VMNOT_M
#undef VM32_VMNAND_MM
#undef VM32_VMNOR_MM
#undef VM32_BITS
#undef VM32_FROM_BITS
#undef VM32_REINT_U_I
#undef VM32_REINT_I_U

#undef VM32_PASTE2
#undef VM32_PASTE3
#undef VM32_XP2
#undef VM32_XP3
#undef VM32_FN
#undef VM32_FN_M
#undef VM32_FN_B
#undef VM32_U32FN
#undef VM32_U32FN_B
#undef VM32_I32FN
#undef VM32_I32FN_B
#undef VM32_BFN

#undef VM32_COMPAT_ACTIVE
#endif /* VM32_COMPAT_ACTIVE */

#ifdef VM32_COMPAT_UNDEF_ONLY
# undef VM32_COMPAT_UNDEF_ONLY
# undef VM32_LMUL
#else

/* ------------------------------------------------------------------ */
/* Definition pass.                                                   */
/* ------------------------------------------------------------------ */

/* Token pasting helpers (two levels so arguments get expanded).  */
#define VM32_PASTE2(a, b) a##b
#define VM32_PASTE3(a, b, c) a##b##c
#define VM32_XP2(a, b) VM32_PASTE2(a, b)
#define VM32_XP3(a, b, c) VM32_PASTE3(a, b, c)

/* Type-name macros (macros, not typedefs, so they can be #undef'ed).  */
#if VM32_LMUL == 1
# define vm32_f_t  vfloat32m1_t
# define vm32_b_t  vbool32_t
# define vm32_u_t  vuint32m1_t
# define vm32_i_t  vint32m1_t
# define VM32_BSUF _b32
#elif VM32_LMUL == 2
# define vm32_f_t  vfloat32m2_t
# define vm32_b_t  vbool16_t
# define vm32_u_t  vuint32m2_t
# define vm32_i_t  vint32m2_t
# define VM32_BSUF _b16
#elif VM32_LMUL == 4
# define vm32_f_t  vfloat32m4_t
# define vm32_b_t  vbool8_t
# define vm32_u_t  vuint32m4_t
# define vm32_i_t  vint32m4_t
# define VM32_BSUF _b8
#elif VM32_LMUL == 8
# define vm32_f_t  vfloat32m8_t
# define vm32_b_t  vbool4_t
# define vm32_u_t  vuint32m8_t
# define vm32_i_t  vint32m8_t
# define VM32_BSUF _b4
#else
# error "VM32_LMUL must be 1, 2, 4 or 8"
#endif

/* Intrinsic-name builders.  */
#define VM32_FN(op)    VM32_XP3(__riscv_, op, VM32_XP2(f32m, VM32_LMUL))
#define VM32_FN_M(op)  VM32_XP2(VM32_FN(op), _m)
#define VM32_FN_B(op)  VM32_XP2(VM32_FN(op), VM32_BSUF)
#define VM32_U32FN(op)   VM32_XP3(__riscv_, op, VM32_XP2(u32m, VM32_LMUL))
#define VM32_U32FN_B(op) VM32_XP2(VM32_U32FN(op), VM32_BSUF)
#define VM32_I32FN(op)   VM32_XP3(__riscv_, op, VM32_XP2(i32m, VM32_LMUL))
#define VM32_I32FN_B(op) VM32_XP2(VM32_I32FN(op), VM32_BSUF)
#define VM32_BFN(op)     VM32_XP3(__riscv_, op, VM32_BSUF)

/* Kernel-name builder: VM32_KFN(sin) -> __vecmath_sin_f32m2 */
#define VM32_KFN(n) VM32_XP3(__vecmath_, n, VM32_XP2(f32m, VM32_LMUL))

#define VM32_VSETVL(n)    VM32_XP2(__riscv_vsetvl_e32m, VM32_LMUL)(n)
#define VM32_VSETVLMAX()  VM32_XP2(__riscv_vsetvlmax_e32m, VM32_LMUL)()

/* ---- loads / stores ---- */
#define VM32_VLE(ptr, vl)      VM32_FN(vle32_v_)(ptr, vl)
#define VM32_VSE(ptr, v, vl)   VM32_FN(vse32_v_)(ptr, v, vl)
#define VM32_VLE_U(ptr, vl)    VM32_U32FN(vle32_v_)(ptr, vl)
#define VM32_VLE_I(ptr, vl)    VM32_I32FN(vle32_v_)(ptr, vl)
#define VM32_VSE_U(ptr, v, vl) VM32_U32FN(vse32_v_)(ptr, v, vl)
#define VM32_VSE_I(ptr, v, vl) VM32_I32FN(vse32_v_)(ptr, v, vl)
#define VM32_VLUXEI32(base, idx, vl)   VM32_FN(vluxei32_v_)(base, idx, vl)
#define VM32_VLUXEI32_U(base, idx, vl) VM32_U32FN(vluxei32_v_)(base, idx, vl)

/* ---- float arithmetic ---- */
#define VM32_VFADD_VV(a, b, vl)  VM32_FN(vfadd_vv_)(a, b, vl)
#define VM32_VFADD_VF(a, s, vl)  VM32_FN(vfadd_vf_)(a, s, vl)
#define VM32_VFSUB_VV(a, b, vl)  VM32_FN(vfsub_vv_)(a, b, vl)
#define VM32_VFSUB_VF(a, s, vl)  VM32_FN(vfsub_vf_)(a, s, vl)
#define VM32_VFRSUB_VF(a, s, vl) VM32_FN(vfrsub_vf_)(a, s, vl)
#define VM32_VFMUL_VV(a, b, vl)  VM32_FN(vfmul_vv_)(a, b, vl)
#define VM32_VFMUL_VF(a, s, vl)  VM32_FN(vfmul_vf_)(a, s, vl)
#define VM32_VFDIV_VV(a, b, vl)  VM32_FN(vfdiv_vv_)(a, b, vl)
#define VM32_VFDIV_VF(a, s, vl)  VM32_FN(vfdiv_vf_)(a, s, vl)
#define VM32_VFRDIV_VF(a, s, vl) VM32_FN(vfrdiv_vf_)(a, s, vl)
#define VM32_VFMACC_VV(d, a, b, vl)  VM32_FN(vfmacc_vv_)(d, a, b, vl)
#define VM32_VFMACC_VF(d, s, b, vl)  VM32_FN(vfmacc_vf_)(d, s, b, vl)
#define VM32_VFNMACC_VV(d, a, b, vl) VM32_FN(vfnmacc_vv_)(d, a, b, vl)
#define VM32_VFNMACC_VF(d, s, b, vl) VM32_FN(vfnmacc_vf_)(d, s, b, vl)
#define VM32_VFMSAC_VV(d, a, b, vl)  VM32_FN(vfmsac_vv_)(d, a, b, vl)
#define VM32_VFMSAC_VF(d, s, b, vl)  VM32_FN(vfmsac_vf_)(d, s, b, vl)
#define VM32_VFNMSAC_VV(d, a, b, vl) VM32_FN(vfnmsac_vv_)(d, a, b, vl)
#define VM32_VFNMSAC_VF(d, s, b, vl) VM32_FN(vfnmsac_vf_)(d, s, b, vl)
#define VM32_VFSQRT_V(a, vl)     VM32_FN(vfsqrt_v_)(a, vl)
#define VM32_VFABS_V(a, vl)      VM32_FN(vfabs_v_)(a, vl)
#define VM32_VFNEG_V(a, vl)      VM32_FN(vfneg_v_)(a, vl)
#define VM32_VFSGNJ_VV(a, b, vl)  VM32_FN(vfsgnj_vv_)(a, b, vl)
#define VM32_VFSGNJX_VV(a, b, vl) VM32_FN(vfsgnjx_vv_)(a, b, vl)
#define VM32_VFSGNJN_VV(a, b, vl) VM32_FN(vfsgnjn_vv_)(a, b, vl)
#define VM32_VFMIN_VV(a, b, vl)  VM32_FN(vfmin_vv_)(a, b, vl)
#define VM32_VFMIN_VF(a, s, vl)  VM32_FN(vfmin_vf_)(a, s, vl)
#define VM32_VFMAX_VV(a, b, vl)  VM32_FN(vfmax_vv_)(a, b, vl)
#define VM32_VFMAX_VF(a, s, vl)  VM32_FN(vfmax_vf_)(a, s, vl)
#define VM32_VFCLASS_V(a, vl)    VM32_U32FN(vfclass_v_)(a, vl)
#define VM32_VFMV_V_F(s, vl)     VM32_FN(vfmv_v_f_)(s, vl)
#define VM32_VFMV_F_S(a)         VM32_XP2(VM32_FN(vfmv_f_s_), _f32)(a)
#define VM32_VFMERGE_VFM(a, s, m, vl) VM32_FN(vfmerge_vfm_)(a, s, m, vl)
/* No explicit __riscv_vfmerge_vvm_* exists in GCC: select through the
   integer vmerge over bit reinterprets (all register-local, zero cost).  */
#define VM32_VFMERGE_VVM(a, b, m, vl) \
  VM32_FROM_BITS(VM32_VMERGE_VVM(VM32_BITS(a), VM32_BITS(b), m, vl))

/* ---- masked float arithmetic (see vecmath_rvvcompat.h for the _M/_MU
   contract: _M leaves inactive lanes unspecified for two-operand
   results, _MU merges an explicit operand).  ---- */
#define VM32_VFMUL_VV_M(m, a, b, vl) VM32_FN_M(vfmul_vv_)(m, a, b, vl)
#define VM32_VFMUL_VF_M(m, a, s, vl) VM32_FN_M(vfmul_vf_)(m, a, s, vl)
#define VM32_VFADD_VV_M(m, a, b, vl) VM32_FN_M(vfadd_vv_)(m, a, b, vl)
#define VM32_VFADD_VF_M(m, a, s, vl) VM32_FN_M(vfadd_vf_)(m, a, s, vl)
#define VM32_VFSUB_VV_M(m, a, b, vl) VM32_FN_M(vfsub_vv_)(m, a, b, vl)
#define VM32_VFSUB_VF_M(m, a, s, vl) VM32_FN_M(vfsub_vf_)(m, a, s, vl)
#define VM32_VFDIV_VF_M(m, a, s, vl) VM32_FN_M(vfdiv_vf_)(m, a, s, vl)
#define VM32_VFRDIV_VF_M(m, a, s, vl) VM32_FN_M(vfrdiv_vf_)(m, a, s, vl)
#define VM32_VFSQRT_V_M(m, a, vl)    VM32_FN_M(vfsqrt_v_)(m, a, vl)
#define VM32_VFMACC_VF_M(m, d, s, b, vl) VM32_FN_M(vfmacc_vf_)(m, d, s, b, vl)
#define VM32_VFMACC_VV_M(m, d, a, b, vl) VM32_FN_M(vfmacc_vv_)(m, d, a, b, vl)
#define VM32_VFNMSAC_VV_M(m, d, a, b, vl) VM32_FN_M(vfnmsac_vv_)(m, d, a, b, vl)
#define VM32_VFMAX_VV_M(m, a, b, vl) VM32_FN_M(vfmax_vv_)(m, a, b, vl)

#define VM32_FN_MU(op) VM32_XP2(VM32_FN(op), _mu)
#define VM32_VFMUL_VV_MU(m, mg, a, b, vl) VM32_FN_MU(vfmul_vv_)(m, mg, a, b, vl)
#define VM32_VFMUL_VF_MU(m, mg, a, s, vl) VM32_FN_MU(vfmul_vf_)(m, mg, a, s, vl)
#define VM32_VFADD_VV_MU(m, mg, a, b, vl) VM32_FN_MU(vfadd_vv_)(m, mg, a, b, vl)
#define VM32_VFADD_VF_MU(m, mg, a, s, vl) VM32_FN_MU(vfadd_vf_)(m, mg, a, s, vl)
#define VM32_VFSUB_VV_MU(m, mg, a, b, vl) VM32_FN_MU(vfsub_vv_)(m, mg, a, b, vl)
#define VM32_VFSUB_VF_MU(m, mg, a, s, vl) VM32_FN_MU(vfsub_vf_)(m, mg, a, s, vl)
#define VM32_VFDIV_VV_MU(m, mg, a, b, vl) VM32_FN_MU(vfdiv_vv_)(m, mg, a, b, vl)
#define VM32_VFDIV_VF_MU(m, mg, a, s, vl) VM32_FN_MU(vfdiv_vf_)(m, mg, a, s, vl)
#define VM32_VFRDIV_VF_MU(m, mg, a, s, vl) VM32_FN_MU(vfrdiv_vf_)(m, mg, a, s, vl)
#define VM32_VFSQRT_V_MU(m, mg, a, vl)    VM32_FN_MU(vfsqrt_v_)(m, mg, a, vl)
#define VM32_VFMAX_VV_MU(m, mg, a, b, vl) VM32_FN_MU(vfmax_vv_)(m, mg, a, b, vl)
#define VM32_VFMIN_VV_MU(m, mg, a, b, vl) VM32_FN_MU(vfmin_vv_)(m, mg, a, b, vl)

/* ---- float compares (result: vm32_b_t) ---- */
#define VM32_VMFEQ_VV(a, b, vl)  VM32_FN_B(vmfeq_vv_)(a, b, vl)
#define VM32_VMFEQ_VF(a, s, vl)  VM32_FN_B(vmfeq_vf_)(a, s, vl)
#define VM32_VMFNE_VV(a, b, vl)  VM32_FN_B(vmfne_vv_)(a, b, vl)
#define VM32_VMFNE_VF(a, s, vl)  VM32_FN_B(vmfne_vf_)(a, s, vl)
#define VM32_VMFLT_VV(a, b, vl)  VM32_FN_B(vmflt_vv_)(a, b, vl)
#define VM32_VMFLT_VF(a, s, vl)  VM32_FN_B(vmflt_vf_)(a, s, vl)
#define VM32_VMFLE_VV(a, b, vl)  VM32_FN_B(vmfle_vv_)(a, b, vl)
#define VM32_VMFLE_VF(a, s, vl)  VM32_FN_B(vmfle_vf_)(a, s, vl)
#define VM32_VMFGT_VF(a, s, vl)  VM32_FN_B(vmfgt_vf_)(a, s, vl)
#define VM32_VMFGE_VF(a, s, vl)  VM32_FN_B(vmfge_vf_)(a, s, vl)

/* ---- float/int converts (int width == SEW here) ---- */
#define VM32_VFCVT_X_F_V(a, vl)      VM32_I32FN(vfcvt_x_f_v_)(a, vl)
#define VM32_VFCVT_X_F_V_RM(a, rm, vl) \
  VM32_XP2(VM32_I32FN(vfcvt_x_f_v_), _rm)(a, rm, vl)
#define VM32_VFCVT_RTZ_X_F_V(a, vl)  VM32_I32FN(vfcvt_rtz_x_f_v_)(a, vl)
#define VM32_VFCVT_F_X_V(a, vl)      VM32_FN(vfcvt_f_x_v_)(a, vl)
#define VM32_VFCVT_F_X_V_RM(a, rm, vl) \
  VM32_XP2(VM32_FN(vfcvt_f_x_v_), _rm)(a, rm, vl)
#define VM32_VFCVT_XU_F_V_RM(a, rm, vl) \
  VM32_XP2(VM32_U32FN(vfcvt_xu_f_v_), _rm)(a, rm, vl)
#define VM32_VFCVT_F_XU_V(a, vl)     VM32_FN(vfcvt_f_xu_v_)(a, vl)

/* ---- integer arithmetic (u32/i32 element width) ---- */
#define VM32_VADD_VX(a, s, vl)   VM32_U32FN(vadd_vx_)(a, s, vl)
#define VM32_VADD_VV(a, b, vl)   VM32_U32FN(vadd_vv_)(a, b, vl)
#define VM32_VSUB_VX(a, s, vl)   VM32_U32FN(vsub_vx_)(a, s, vl)
#define VM32_VSUB_VV(a, b, vl)   VM32_U32FN(vsub_vv_)(a, b, vl)
#define VM32_VMUL_VX(a, s, vl)   VM32_U32FN(vmul_vx_)(a, s, vl)
#define VM32_VSLL_VX(a, s, vl)   VM32_U32FN(vsll_vx_)(a, s, vl)
#define VM32_VSRL_VX(a, s, vl)   VM32_U32FN(vsrl_vx_)(a, s, vl)
#define VM32_VSRL_VV(a, b, vl)   VM32_U32FN(vsrl_vv_)(a, b, vl)
#define VM32_VSRA_VX(a, s, vl)   VM32_I32FN(vsra_vx_)(a, s, vl)
#define VM32_VAND_VX(a, s, vl)   VM32_U32FN(vand_vx_)(a, s, vl)
#define VM32_VAND_VV(a, b, vl)   VM32_U32FN(vand_vv_)(a, b, vl)
#define VM32_VOR_VX(a, s, vl)    VM32_U32FN(vor_vx_)(a, s, vl)
#define VM32_VOR_VV(a, b, vl)    VM32_U32FN(vor_vv_)(a, b, vl)
#define VM32_VXOR_VX(a, s, vl)   VM32_U32FN(vxor_vx_)(a, s, vl)
#define VM32_VXOR_VV(a, b, vl)   VM32_U32FN(vxor_vv_)(a, b, vl)
#define VM32_VMSEQ_VX(a, s, vl)  VM32_U32FN_B(vmseq_vx_)(a, s, vl)
#define VM32_VMSNE_VX(a, s, vl)  VM32_U32FN_B(vmsne_vx_)(a, s, vl)
#define VM32_VMSLT_VX(a, s, vl)  VM32_I32FN_B(vmslt_vx_)(a, s, vl)
#define VM32_VMSLTU_VX(a, s, vl) VM32_U32FN_B(vmsltu_vx_)(a, s, vl)
#define VM32_VMSLE_VX(a, s, vl)  VM32_I32FN_B(vmsle_vx_)(a, s, vl)
#define VM32_VMSLEU_VX(a, s, vl) VM32_U32FN_B(vmsleu_vx_)(a, s, vl)
#define VM32_VMSGT_VX(a, s, vl)  VM32_I32FN_B(vmsgt_vx_)(a, s, vl)
#define VM32_VMSGTU_VX(a, s, vl) VM32_U32FN_B(vmsgtu_vx_)(a, s, vl)
#define VM32_VMSGE_VX(a, s, vl)  VM32_I32FN_B(vmsge_vx_)(a, s, vl)
#define VM32_VMSGEU_VX(a, s, vl) VM32_U32FN_B(vmsgeu_vx_)(a, s, vl)
#define VM32_VMSLT_VV(a, b, vl)  VM32_I32FN_B(vmslt_vv_)(a, b, vl)
#define VM32_VMSLTU_VV(a, b, vl) VM32_U32FN_B(vmsltu_vv_)(a, b, vl)
#define VM32_VIOTA_M(m, vl)      VM32_U32FN(viota_m_)(m, vl)
#define VM32_VMV_V_X(s, vl)      VM32_U32FN(vmv_v_x_)(s, vl)
#define VM32_VMV_V_V(a, vl)      VM32_U32FN(vmv_v_v_)(a, vl)
#define VM32_VMV_X_S(a)          VM32_XP2(VM32_U32FN(vmv_x_s_), _u32)(a)
#define VM32_VMERGE_VXM(a, s, m, vl) VM32_U32FN(vmerge_vxm_)(a, s, m, vl)
#define VM32_VMERGE_VVM(a, b, m, vl) VM32_U32FN(vmerge_vvm_)(a, b, m, vl)
#define VM32_VSLIDEDOWN_VX(a, s, vl) VM32_U32FN(vslidedown_vx_)(a, s, vl)
#define VM32_VSLIDEUP_VX(a, s, vl)   VM32_U32FN(vslideup_vx_)(a, s, vl)

/* ---- mask ops ---- */
#define VM32_VFIRST_M(m, vl)     VM32_BFN(vfirst_m)(m, vl)
#define VM32_VMSBF_M(m, vl)      VM32_BFN(vmsbf_m)(m, vl)
#define VM32_VMSIF_M(m, vl)      VM32_BFN(vmsif_m)(m, vl)
#define VM32_VMSOF_M(m, vl)      VM32_BFN(vmsof_m)(m, vl)
#define VM32_VCMP_POPC_M(m, vl)  VM32_BFN(vcpop_m)(m, vl)
#define VM32_VCOMPRESS(src, m, vl) VM32_FN(vcompress_vm_)(src, m, vl)
#define VM32_VMAND_MM(a, b, vl)  VM32_BFN(vmand_mm)(a, b, vl)
#define VM32_VMOR_MM(a, b, vl)   VM32_BFN(vmor_mm)(a, b, vl)
#define VM32_VMXOR_MM(a, b, vl)  VM32_BFN(vmxor_mm)(a, b, vl)
#define VM32_VMNOT_M(a, vl)      VM32_BFN(vmnot_m)(a, vl)
#define VM32_VMNAND_MM(a, b, vl) VM32_BFN(vmnand_mm)(a, b, vl)
#define VM32_VMNOR_MM(a, b, vl)  VM32_BFN(vmnor_mm)(a, b, vl)

/* ---- bit reinterpretation ---- */
#define VM32_BITS(v) \
  VM32_XP2(VM32_XP3(__riscv_vreinterpret_v_f32m, VM32_LMUL, _u32m), VM32_LMUL)(v)
#define VM32_FROM_BITS(u) \
  VM32_XP2(VM32_XP3(__riscv_vreinterpret_v_u32m, VM32_LMUL, _f32m), VM32_LMUL)(u)
#define VM32_REINT_U_I(u) \
  VM32_XP2(VM32_XP3(__riscv_vreinterpret_v_u32m, VM32_LMUL, _i32m), VM32_LMUL)(u)
#define VM32_REINT_I_U(i) \
  VM32_XP2(VM32_XP3(__riscv_vreinterpret_v_i32m, VM32_LMUL, _u32m), VM32_LMUL)(i)

#define VM32_COMPAT_ACTIVE 1
#endif /* VM32_COMPAT_UNDEF_ONLY */
