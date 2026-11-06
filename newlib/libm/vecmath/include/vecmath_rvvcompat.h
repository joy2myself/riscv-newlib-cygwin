/* vecmath_rvvcompat.h -- LMUL-generic RVV 1.0 intrinsic wrapper layer.

   Part of the newlib RISC-V vector math library (libvecmath).

   This header is RE-INCLUDABLE: the includer defines VM_LMUL (1, 2, 4 or 8)
   before each inclusion and every macro defined here is #undef'ed again on
   re-inclusion, so kernel templates (k_*.inc) can be instantiated once per
   LMUL from a single source of truth.

   Usage:
     #define VM_LMUL 2
     #include <vecmath_rvvcompat.h>
     ... use vm_f_t, VM_VFADD_VV(a, b, vl), VM_KFN(sin)(x, vl) ...

   If VM_COMPAT_UNDEF_ONLY is defined before inclusion, only the #undef
   pass runs (used to clean up after the last instantiation).

   Naming rules implemented here (verified against GCC 15.1 and matching
   GCC 13/14 rvv-intrinsic-doc conventions):
     arithmetic/load   __riscv_<op>_<shape>_f64mK          (vfadd_vv_f64m4)
     masked policy     __riscv_<op>_<shape>_f64mK_m        (vfadd_vv_f64m4_m)
     comparisons       __riscv_<op>_<shape>_f64mK_bR       (vmfeq_vf_f64m4_b16)
     mask-mask ops     __riscv_<op>_mm_bR / _m_bR          (vmand_mm_b16)
     int merge         __riscv_vmerge_vvm_u64mK (no bool suffix)
     float vv merge    does NOT exist as an explicit name -- implemented
                       here as an integer vmerge over bit reinterprets.
     vcompress         __riscv_vcompress_vm_f64mK(src, mask, vl), 3 args.
     scalar moves      __riscv_vmv_x_s_u64mK_u64 / __riscv_vfmv_f_s_f64mK_f64.  */

#ifndef VM_LMUL
# if !defined(VM_COMPAT_UNDEF_ONLY)
#  error "define VM_LMUL (1,2,4,8) before including vecmath_rvvcompat.h"
# endif
# define VM_LMUL 1
#endif

/* ------------------------------------------------------------------ */
/* Undef pass (runs on every inclusion).                              */
/* ------------------------------------------------------------------ */
#ifdef VM_COMPAT_ACTIVE

#undef vm_f_t
#undef vm_b_t
#undef vm_i64_t
#undef vm_u64_t
#undef vm_i32_t
#undef VM_F64SUF
#undef VM_BSUF
#undef VM_VSETVL
#undef VM_VSETVLMAX
#undef VM_KFN

#undef VM_VLE64
#undef VM_VSE64
#undef VM_VSE32
#undef VM_VNCVT_X_X_W
#undef VM_VLUXEI64
#undef VM_VLUXEI64_U

#undef VM_VFADD_VV
#undef VM_VFADD_VF
#undef VM_VFSUB_VV
#undef VM_VFSUB_VF
#undef VM_VFRSUB_VF
#undef VM_VFMUL_VV
#undef VM_VFMUL_VF
#undef VM_VFDIV_VV
#undef VM_VFDIV_VF
#undef VM_VFRDIV_VF
#undef VM_VFMACC_VV
#undef VM_VFMACC_VF
#undef VM_VFNMACC_VV
#undef VM_VFNMACC_VF
#undef VM_VFMSAC_VV
#undef VM_VFMSAC_VF
#undef VM_VFNMSAC_VV
#undef VM_VFNMSAC_VF
#undef VM_VFSQRT_V
#undef VM_VFABS_V
#undef VM_VFNEG_V
#undef VM_VFSGNJ_VV
#undef VM_VFSGNJX_VV
#undef VM_VFSGNJN_VV
#undef VM_VFMIN_VV
#undef VM_VFMIN_VF
#undef VM_VFMAX_VV
#undef VM_VFMAX_VF
#undef VM_VFCLASS_V
#undef VM_VFMV_V_F
#undef VM_VFMV_F_S
#undef VM_VFMERGE_VFM
#undef VM_VFMERGE_VVM

#undef VM_VFMUL_VV_M
#undef VM_VFMUL_VF_M
#undef VM_VFADD_VV_M
#undef VM_VFADD_VF_M
#undef VM_VFSUB_VF_M
#undef VM_VFSUB_VV_M
#undef VM_VFDIV_VF_M
#undef VM_VFRDIV_VF_M
#undef VM_VFSQRT_V_M
#undef VM_VFMACC_VF_M
#undef VM_VFMACC_VF_MU
#undef VM_VFMACC_VV_M
#undef VM_VFNMSAC_VV_M
#undef VM_VFMAX_VV_M
#undef VM_VFMERGE_VVM_TU
#undef VM_F64FN_MU
#undef VM_VFMUL_VV_MU
#undef VM_VFMUL_VF_MU
#undef VM_VFADD_VV_MU
#undef VM_VFADD_VF_MU
#undef VM_VFSUB_VV_MU
#undef VM_VFSUB_VF_MU
#undef VM_VFDIV_VV_MU
#undef VM_VFDIV_VF_MU
#undef VM_VFRDIV_VF_MU
#undef VM_VFSQRT_V_MU
#undef VM_VFMAX_VV_MU
#undef VM_VFMIN_VV_MU

#undef VM_VMFEQ_VV
#undef VM_VMFEQ_VF
#undef VM_VMFNE_VV
#undef VM_VMFNE_VF
#undef VM_VMFLT_VV
#undef VM_VMFLT_VF
#undef VM_VMFLE_VV
#undef VM_VMFLE_VF
#undef VM_VMFGT_VF
#undef VM_VMFGE_VF

#undef VM_VFCVT_X_F_V
#undef VM_VFCVT_X_F_V_RM
#undef VM_VFCVT_RTZ_X_F_V
#undef VM_VFCVT_F_X_V
#undef VM_VFCVT_F_X_V_RM
#undef VM_VFCVT_XU_F_V_RM
#undef VM_VFCVT_F_XU_V

#undef VM_VADD_VX
#undef VM_VADD_VV
#undef VM_VSUB_VX
#undef VM_VSUB_VV
#undef VM_VMUL_VX
#undef VM_VMUL_VV
#undef VM_VSLL_VX
#undef VM_VSRL_VX
#undef VM_VSRL_VV
#undef VM_VSRA_VX
#undef VM_VAND_VX
#undef VM_VAND_VV
#undef VM_VOR_VX
#undef VM_VOR_VV
#undef VM_VXOR_VX
#undef VM_VXOR_VV
#undef VM_VMSEQ_VX
#undef VM_VMSNE_VX
#undef VM_VMSLT_VX
#undef VM_VMSLTU_VX
#undef VM_VMSLTU_VV
#undef VM_VMSLE_VX
#undef VM_VMSLEU_VX
#undef VM_VMSGT_VX
#undef VM_VMSGTU_VX
#undef VM_VMSGE_VX
#undef VM_VMSGEU_VX
#undef VM_VIOTA_M
#undef VM_VMSLT_VV
#undef VM_VMV_V_X
#undef VM_VMV_V_V
#undef VM_VMV_X_S
#undef VM_VMERGE_VXM
#undef VM_VMERGE_VVM
#undef VM_VSLIDEDOWN_VX
#undef VM_VSLIDEUP_VX

#undef VM_VFIRST_M
#undef VM_VMSBF_M
#undef VM_VMSIF_M
#undef VM_VMSOF_M
#undef VM_VCMP_POPC_M
#undef VM_VCOMPRESS
#undef VM_VMAND_MM
#undef VM_VMOR_MM
#undef VM_VMXOR_MM
#undef VM_VMNOT_M
#undef VM_VMNAND_MM
#undef VM_VMNOR_MM
#undef VM_BITS
#undef VM_FROM_BITS
#undef VM_REINT_U_I
#undef VM_REINT_I_U

#undef VM_PASTE2
#undef VM_PASTE3
#undef VM_XP2
#undef VM_XP3
#undef VM_F64FN
#undef VM_F64FN_M
#undef VM_F64FN_B
#undef VM_U64FN
#undef VM_U64FN_B
#undef VM_I64FN_B
#undef VM_I64FN
#undef VM_I32FN
#undef VM_BFN

#undef VM_COMPAT_ACTIVE
#endif /* VM_COMPAT_ACTIVE */

#ifdef VM_COMPAT_UNDEF_ONLY
# undef VM_COMPAT_UNDEF_ONLY
# undef VM_LMUL
#else

/* ------------------------------------------------------------------ */
/* Definition pass.                                                   */
/* ------------------------------------------------------------------ */

/* Token pasting helpers (two levels so arguments get expanded).  */
#define VM_PASTE2(a, b) a##b
#define VM_PASTE3(a, b, c) a##b##c
#define VM_XP2(a, b) VM_PASTE2(a, b)
#define VM_XP3(a, b, c) VM_PASTE3(a, b, c)

/* Type-name macros (macros, not typedefs, so they can be #undef'ed).  */
#if VM_LMUL == 1
# define vm_f_t   vfloat64m1_t
# define vm_b_t   vbool64_t
# define vm_i64_t vint64m1_t
# define vm_u64_t vuint64m1_t
# define vm_i32_t vint32mf2_t
# define VM_BSUF  _b64
# define VM_I32FN(op) VM_XP3(__riscv_, op, i32mf2)
#elif VM_LMUL == 2
# define vm_f_t   vfloat64m2_t
# define vm_b_t   vbool32_t
# define vm_i64_t vint64m2_t
# define vm_u64_t vuint64m2_t
# define vm_i32_t vint32m1_t
# define VM_BSUF  _b32
# define VM_I32FN(op) VM_XP3(__riscv_, op, i32m1)
#elif VM_LMUL == 4
# define vm_f_t   vfloat64m4_t
# define vm_b_t   vbool16_t
# define vm_i64_t vint64m4_t
# define vm_u64_t vuint64m4_t
# define vm_i32_t vint32m2_t
# define VM_BSUF  _b16
# define VM_I32FN(op) VM_XP3(__riscv_, op, i32m2)
#elif VM_LMUL == 8
# define vm_f_t   vfloat64m8_t
# define vm_b_t   vbool8_t
# define vm_i64_t vint64m8_t
# define vm_u64_t vuint64m8_t
# define vm_i32_t vint32m4_t
# define VM_BSUF  _b8
# define VM_I32FN(op) VM_XP3(__riscv_, op, i32m4)
#else
# error "VM_LMUL must be 1, 2, 4 or 8"
#endif

/* Intrinsic-name builders.  */
#define VM_F64FN(op)   VM_XP3(__riscv_, op, VM_XP2(f64m, VM_LMUL))
#define VM_F64FN_M(op) VM_XP2(VM_F64FN(op), _m)
#define VM_F64FN_B(op) VM_XP2(VM_F64FN(op), VM_BSUF)
#define VM_U64FN(op)   VM_XP3(__riscv_, op, VM_XP2(u64m, VM_LMUL))
#define VM_U64FN_B(op) VM_XP2(VM_U64FN(op), VM_BSUF)
#define VM_I64FN_B(op) VM_XP2(VM_I64FN(op), VM_BSUF)
#define VM_I64FN(op)   VM_XP3(__riscv_, op, VM_XP2(i64m, VM_LMUL))
#define VM_BFN(op)     VM_XP3(__riscv_, op, VM_BSUF)

/* Kernel-name builder: VM_KFN(sin) -> __vecmath_sin_f64m2 */
#define VM_KFN(n) VM_XP3(__vecmath_, n, VM_XP2(f64m, VM_LMUL))

#define VM_VSETVL(n)    VM_XP2(__riscv_vsetvl_e64m, VM_LMUL)(n)
#define VM_VSETVLMAX()  VM_XP2(__riscv_vsetvlmax_e64m, VM_LMUL)()

/* ---- loads / stores ---- */
#define VM_VLE64(ptr, vl)      VM_F64FN(vle64_v_)(ptr, vl)
#define VM_VSE64(ptr, v, vl)   VM_F64FN(vse64_v_)(ptr, v, vl)
#define VM_VSE32(ptr, v, vl)   VM_I32FN(vse32_v_)(ptr, v, vl)
/* i64 -> i32 narrowing, for entry points that write an int array out of a
   value taken from a f64 vector.  */
#define VM_VNCVT_X_X_W(a, vl)  VM_I32FN(vncvt_x_x_w_)(a, vl)
#define VM_VLUXEI64(base, idx, vl) VM_F64FN(vluxei64_v_)(base, idx, vl)
#define VM_VLUXEI64_U(base, idx, vl) VM_U64FN(vluxei64_v_)(base, idx, vl)

/* ---- float arithmetic ---- */
#define VM_VFADD_VV(a, b, vl)  VM_F64FN(vfadd_vv_)(a, b, vl)
#define VM_VFADD_VF(a, s, vl)  VM_F64FN(vfadd_vf_)(a, s, vl)
#define VM_VFSUB_VV(a, b, vl)  VM_F64FN(vfsub_vv_)(a, b, vl)
#define VM_VFSUB_VF(a, s, vl)  VM_F64FN(vfsub_vf_)(a, s, vl)
#define VM_VFRSUB_VF(a, s, vl) VM_F64FN(vfrsub_vf_)(a, s, vl)
#define VM_VFMUL_VV(a, b, vl)  VM_F64FN(vfmul_vv_)(a, b, vl)
#define VM_VFMUL_VF(a, s, vl)  VM_F64FN(vfmul_vf_)(a, s, vl)
#define VM_VFDIV_VV(a, b, vl)  VM_F64FN(vfdiv_vv_)(a, b, vl)
#define VM_VFDIV_VF(a, s, vl)  VM_F64FN(vfdiv_vf_)(a, s, vl)
#define VM_VFRDIV_VF(a, s, vl) VM_F64FN(vfrdiv_vf_)(a, s, vl)
#define VM_VFMACC_VV(d, a, b, vl)  VM_F64FN(vfmacc_vv_)(d, a, b, vl)
#define VM_VFMACC_VF(d, s, b, vl)  VM_F64FN(vfmacc_vf_)(d, s, b, vl)
#define VM_VFNMACC_VV(d, a, b, vl) VM_F64FN(vfnmacc_vv_)(d, a, b, vl)
#define VM_VFNMACC_VF(d, s, b, vl) VM_F64FN(vfnmacc_vf_)(d, s, b, vl)
#define VM_VFMSAC_VV(d, a, b, vl)  VM_F64FN(vfmsac_vv_)(d, a, b, vl)
#define VM_VFMSAC_VF(d, s, b, vl)  VM_F64FN(vfmsac_vf_)(d, s, b, vl)
#define VM_VFNMSAC_VV(d, a, b, vl) VM_F64FN(vfnmsac_vv_)(d, a, b, vl)
#define VM_VFNMSAC_VF(d, s, b, vl) VM_F64FN(vfnmsac_vf_)(d, s, b, vl)
#define VM_VFSQRT_V(a, vl)     VM_F64FN(vfsqrt_v_)(a, vl)
#define VM_VFABS_V(a, vl)      VM_F64FN(vfabs_v_)(a, vl)
#define VM_VFNEG_V(a, vl)      VM_F64FN(vfneg_v_)(a, vl)
#define VM_VFSGNJ_VV(a, b, vl)  VM_F64FN(vfsgnj_vv_)(a, b, vl)
#define VM_VFSGNJX_VV(a, b, vl) VM_F64FN(vfsgnjx_vv_)(a, b, vl)
#define VM_VFSGNJN_VV(a, b, vl) VM_F64FN(vfsgnjn_vv_)(a, b, vl)
#define VM_VFMIN_VV(a, b, vl)  VM_F64FN(vfmin_vv_)(a, b, vl)
#define VM_VFMIN_VF(a, s, vl)  VM_F64FN(vfmin_vf_)(a, s, vl)
#define VM_VFMAX_VV(a, b, vl)  VM_F64FN(vfmax_vv_)(a, b, vl)
#define VM_VFMAX_VF(a, s, vl)  VM_F64FN(vfmax_vf_)(a, s, vl)
#define VM_VFCLASS_V(a, vl)    VM_U64FN(vfclass_v_)(a, vl)
#define VM_VFMV_V_F(s, vl)     VM_F64FN(vfmv_v_f_)(s, vl)
#define VM_VFMV_F_S(a)         VM_XP2(VM_F64FN(vfmv_f_s_), _f64)(a)
#define VM_VFMERGE_VFM(a, s, m, vl) VM_F64FN(vfmerge_vfm_)(a, s, m, vl)
/* No explicit __riscv_vfmerge_vvm_* exists in GCC: select through the
   integer vmerge over bit reinterprets (all register-local, zero cost).  */
#define VM_VFMERGE_VVM(a, b, m, vl) \
  VM_FROM_BITS(VM_VMERGE_VVM(VM_BITS(a), VM_BITS(b), m, vl))

/* ---- masked float arithmetic.

   _M forms (mask-undisturbed, no merge argument): inactive lanes are
   unspecified for two-operand results (vfmul/vfadd/vfsub/vfdiv/vfsqrt);
   they only raise no flags.  Use _M only when the result's inactive
   lanes are consumed exclusively under the same mask (or never read).
   Exception: accumulating ops (vfmacc/vfnmsac/...) take vd as an
   argument, so their _m form deterministically keeps vd in inactive
   lanes.

   _MU forms (explicit merge operand): inactive lanes == merge.  Use
   whenever the result may be read on inactive lanes.  ---- */
#define VM_VFMUL_VV_M(m, a, b, vl) VM_F64FN_M(vfmul_vv_)(m, a, b, vl)
#define VM_VFMUL_VF_M(m, a, s, vl) VM_F64FN_M(vfmul_vf_)(m, a, s, vl)
#define VM_VFADD_VV_M(m, a, b, vl) VM_F64FN_M(vfadd_vv_)(m, a, b, vl)
#define VM_VFADD_VF_M(m, a, s, vl) VM_F64FN_M(vfadd_vf_)(m, a, s, vl)
#define VM_VFSUB_VF_M(m, a, s, vl) VM_F64FN_M(vfsub_vf_)(m, a, s, vl)
#define VM_VFSUB_VV_M(m, a, b, vl) VM_F64FN_M(vfsub_vv_)(m, a, b, vl)
#define VM_VFDIV_VF_M(m, a, s, vl) VM_F64FN_M(vfdiv_vf_)(m, a, s, vl)
#define VM_VFRDIV_VF_M(m, a, s, vl) VM_F64FN_M(vfrdiv_vf_)(m, a, s, vl)
#define VM_VFSQRT_V_M(m, a, vl)    VM_F64FN_M(vfsqrt_v_)(m, a, vl)
#define VM_VFMACC_VF_M(m, d, s, b, vl) VM_F64FN_M(vfmacc_vf_)(m, d, s, b, vl)
#define VM_VFMACC_VV_M(m, d, a, b, vl) VM_F64FN_M(vfmacc_vv_)(m, d, a, b, vl)
#define VM_VFNMSAC_VV_M(m, d, a, b, vl) VM_F64FN_M(vfnmsac_vv_)(m, d, a, b, vl)
#define VM_VFMAX_VV_M(m, a, b, vl) VM_F64FN_M(vfmax_vv_)(m, a, b, vl)

#define VM_F64FN_MU(op) VM_XP2(VM_F64FN(op), _mu)
#define VM_VFMUL_VV_MU(m, mg, a, b, vl) VM_F64FN_MU(vfmul_vv_)(m, mg, a, b, vl)
#define VM_VFMUL_VF_MU(m, mg, a, s, vl) VM_F64FN_MU(vfmul_vf_)(m, mg, a, s, vl)
#define VM_VFADD_VV_MU(m, mg, a, b, vl) VM_F64FN_MU(vfadd_vv_)(m, mg, a, b, vl)
#define VM_VFADD_VF_MU(m, mg, a, s, vl) VM_F64FN_MU(vfadd_vf_)(m, mg, a, s, vl)
#define VM_VFSUB_VV_MU(m, mg, a, b, vl) VM_F64FN_MU(vfsub_vv_)(m, mg, a, b, vl)
#define VM_VFSUB_VF_MU(m, mg, a, s, vl) VM_F64FN_MU(vfsub_vf_)(m, mg, a, s, vl)
#define VM_VFDIV_VV_MU(m, mg, a, b, vl) VM_F64FN_MU(vfdiv_vv_)(m, mg, a, b, vl)
#define VM_VFDIV_VF_MU(m, mg, a, s, vl) VM_F64FN_MU(vfdiv_vf_)(m, mg, a, s, vl)
#define VM_VFRDIV_VF_MU(m, mg, a, s, vl) VM_F64FN_MU(vfrdiv_vf_)(m, mg, a, s, vl)
#define VM_VFSQRT_V_MU(m, mg, a, vl)    VM_F64FN_MU(vfsqrt_v_)(m, mg, a, vl)
#define VM_VFMACC_VF_MU(m, mg, s, b, vl) VM_F64FN_MU(vfmacc_vf_)(m, mg, s, b, vl)
#define VM_VFMAX_VV_MU(m, mg, a, b, vl) VM_F64FN_MU(vfmax_vv_)(m, mg, a, b, vl)
#define VM_VFMIN_VV_MU(m, mg, a, b, vl) VM_F64FN_MU(vfmin_vv_)(m, mg, a, b, vl)

/* ---- float compares (result: vm_b_t) ---- */
#define VM_VMFEQ_VV(a, b, vl)  VM_F64FN_B(vmfeq_vv_)(a, b, vl)
#define VM_VMFEQ_VF(a, s, vl)  VM_F64FN_B(vmfeq_vf_)(a, s, vl)
#define VM_VMFNE_VV(a, b, vl)  VM_F64FN_B(vmfne_vv_)(a, b, vl)
#define VM_VMFNE_VF(a, s, vl)  VM_F64FN_B(vmfne_vf_)(a, s, vl)
#define VM_VMFLT_VV(a, b, vl)  VM_F64FN_B(vmflt_vv_)(a, b, vl)
#define VM_VMFLT_VF(a, s, vl)  VM_F64FN_B(vmflt_vf_)(a, s, vl)
#define VM_VMFLE_VV(a, b, vl)  VM_F64FN_B(vmfle_vv_)(a, b, vl)
#define VM_VMFLE_VF(a, s, vl)  VM_F64FN_B(vmfle_vf_)(a, s, vl)
#define VM_VMFGT_VF(a, s, vl)  VM_F64FN_B(vmfgt_vf_)(a, s, vl)
#define VM_VMFGE_VF(a, s, vl)  VM_F64FN_B(vmfge_vf_)(a, s, vl)

/* ---- float/int converts ---- */
#define VM_VFCVT_X_F_V(a, vl)      VM_I64FN(vfcvt_x_f_v_)(a, vl)
#define VM_VFCVT_X_F_V_RM(a, rm, vl) \
  VM_XP2(VM_I64FN(vfcvt_x_f_v_), _rm)(a, rm, vl)
#define VM_VFCVT_RTZ_X_F_V(a, vl)  VM_I64FN(vfcvt_rtz_x_f_v_)(a, vl)
#define VM_VFCVT_F_X_V(a, vl)      VM_F64FN(vfcvt_f_x_v_)(a, vl)
#define VM_VFCVT_F_X_V_RM(a, rm, vl) \
  VM_XP2(VM_F64FN(vfcvt_f_x_v_), _rm)(a, rm, vl)
#define VM_VFCVT_XU_F_V_RM(a, rm, vl) \
  VM_XP2(VM_U64FN(vfcvt_xu_f_v_), _rm)(a, rm, vl)
#define VM_VFCVT_F_XU_V(a, vl)     VM_F64FN(vfcvt_f_xu_v_)(a, vl)

/* ---- integer arithmetic (u64/i64 element width) ---- */
#define VM_VADD_VX(a, s, vl)   VM_U64FN(vadd_vx_)(a, s, vl)
#define VM_VADD_VV(a, b, vl)   VM_U64FN(vadd_vv_)(a, b, vl)
#define VM_VSUB_VX(a, s, vl)   VM_U64FN(vsub_vx_)(a, s, vl)
#define VM_VSUB_VV(a, b, vl)   VM_U64FN(vsub_vv_)(a, b, vl)
#define VM_VMUL_VX(a, s, vl)   VM_U64FN(vmul_vx_)(a, s, vl)
#define VM_VMUL_VV(a, b, vl)   VM_U64FN(vmul_vv_)(a, b, vl)
#define VM_VSLL_VX(a, s, vl)   VM_U64FN(vsll_vx_)(a, s, vl)
#define VM_VSRL_VX(a, s, vl)   VM_U64FN(vsrl_vx_)(a, s, vl)
#define VM_VSRL_VV(a, b, vl)   VM_U64FN(vsrl_vv_)(a, b, vl)
#define VM_VSRA_VX(a, s, vl)   VM_I64FN(vsra_vx_)(a, s, vl)
#define VM_VAND_VX(a, s, vl)   VM_U64FN(vand_vx_)(a, s, vl)
#define VM_VAND_VV(a, b, vl)   VM_U64FN(vand_vv_)(a, b, vl)
#define VM_VOR_VX(a, s, vl)    VM_U64FN(vor_vx_)(a, s, vl)
#define VM_VOR_VV(a, b, vl)    VM_U64FN(vor_vv_)(a, b, vl)
#define VM_VXOR_VX(a, s, vl)   VM_U64FN(vxor_vx_)(a, s, vl)
#define VM_VXOR_VV(a, b, vl)   VM_U64FN(vxor_vv_)(a, b, vl)
#define VM_VMSEQ_VX(a, s, vl)  VM_U64FN_B(vmseq_vx_)(a, s, vl)
#define VM_VMSNE_VX(a, s, vl)  VM_U64FN_B(vmsne_vx_)(a, s, vl)
#define VM_VMSLT_VX(a, s, vl)  VM_I64FN_B(vmslt_vx_)(a, s, vl)
#define VM_VMSLTU_VX(a, s, vl) VM_U64FN_B(vmsltu_vx_)(a, s, vl)
#define VM_VMSLE_VX(a, s, vl)  VM_I64FN_B(vmsle_vx_)(a, s, vl)
#define VM_VMSLEU_VX(a, s, vl) VM_U64FN_B(vmsleu_vx_)(a, s, vl)
#define VM_VMSGT_VX(a, s, vl)  VM_I64FN_B(vmsgt_vx_)(a, s, vl)
#define VM_VMSGTU_VX(a, s, vl) VM_U64FN_B(vmsgtu_vx_)(a, s, vl)
#define VM_VMSGE_VX(a, s, vl)  VM_I64FN_B(vmsge_vx_)(a, s, vl)
#define VM_VMSGEU_VX(a, s, vl) VM_U64FN_B(vmsgeu_vx_)(a, s, vl)
#define VM_VIOTA_M(m, vl)      VM_U64FN(viota_m_)(m, vl)
#define VM_VMSLT_VV(a, b, vl)  VM_I64FN_B(vmslt_vv_)(a, b, vl)
#define VM_VMSLTU_VV(a, b, vl) VM_U64FN_B(vmsltu_vv_)(a, b, vl)
#define VM_VMV_V_X(s, vl)      VM_U64FN(vmv_v_x_)(s, vl)
#define VM_VMV_V_V(a, vl)      VM_U64FN(vmv_v_v_)(a, vl)
#define VM_VMV_X_S(a)          VM_XP2(VM_U64FN(vmv_x_s_), _u64)(a)
#define VM_VMERGE_VXM(a, s, m, vl) VM_U64FN(vmerge_vxm_)(a, s, m, vl)
#define VM_VMERGE_VVM(a, b, m, vl) VM_U64FN(vmerge_vvm_)(a, b, m, vl)
#define VM_VSLIDEDOWN_VX(a, s, vl) VM_U64FN(vslidedown_vx_)(a, s, vl)
#define VM_VSLIDEUP_VX(a, s, vl)   VM_U64FN(vslideup_vx_)(a, s, vl)

/* ---- mask ops ---- */
#define VM_VFIRST_M(m, vl)     VM_BFN(vfirst_m)(m, vl)
#define VM_VMSBF_M(m, vl)      VM_BFN(vmsbf_m)(m, vl)
#define VM_VMSIF_M(m, vl)      VM_BFN(vmsif_m)(m, vl)
#define VM_VMSOF_M(m, vl)      VM_BFN(vmsof_m)(m, vl)
#define VM_VCMP_POPC_M(m, vl)  VM_BFN(vcpop_m)(m, vl)
#define VM_VCOMPRESS(src, m, vl) VM_F64FN(vcompress_vm_)(src, m, vl)
#define VM_VMAND_MM(a, b, vl)  VM_BFN(vmand_mm)(a, b, vl)
#define VM_VMOR_MM(a, b, vl)   VM_BFN(vmor_mm)(a, b, vl)
#define VM_VMXOR_MM(a, b, vl)  VM_BFN(vmxor_mm)(a, b, vl)
#define VM_VMNOT_M(a, vl)      VM_BFN(vmnot_m)(a, vl)
#define VM_VMNAND_MM(a, b, vl) VM_BFN(vmnand_mm)(a, b, vl)
#define VM_VMNOR_MM(a, b, vl)  VM_BFN(vmnor_mm)(a, b, vl)

/* ---- bit reinterpretation ---- */
#define VM_BITS(v) \
  VM_XP2(VM_XP3(__riscv_vreinterpret_v_f64m, VM_LMUL, _u64m), VM_LMUL)(v)
#define VM_FROM_BITS(u) \
  VM_XP2(VM_XP3(__riscv_vreinterpret_v_u64m, VM_LMUL, _f64m), VM_LMUL)(u)
#define VM_REINT_U_I(u) \
  VM_XP2(VM_XP3(__riscv_vreinterpret_v_u64m, VM_LMUL, _i64m), VM_LMUL)(u)
#define VM_REINT_I_U(i) \
  VM_XP2(VM_XP3(__riscv_vreinterpret_v_i64m, VM_LMUL, _u64m), VM_LMUL)(i)

#define VM_COMPAT_ACTIVE 1
#endif /* VM_COMPAT_UNDEF_ONLY */
