/* f32/k_gamma.inc -- binary32 tgamma / lgamma / lgamma_r kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_gamma.inc,
   which pulls k_exp.inc, k_log.inc and k_trig.inc) active at
   VM_LMUL == 2*VM32_LMUL.  */

VM32_W1 (tgamma, tgamma)
VM32_W1 (lgamma, lgamma)

/* lgamma_r: the binary64 kernel yields the double-double value and the
   +/-1 sign; collapse and narrow the value, and derive the int sign from
   the narrowed +/-1 (exact, so the round-trip through binary32 changes
   nothing).  */
#define VM32_GAM_LGAMMA_R(yo, sgno, x, vl) do {				\
  VM32_WD_T vmg32__xd = VM32_WIDEN ((x), (vl));				\
  vm_f_t vmg32__h, vmg32__l, vmg32__s, vmg32__r;			\
  VM_GAM_LGAMMA_R (vmg32__h, vmg32__l, vmg32__s, vmg32__xd, (vl));	\
  VM_DD_COLLAPSE (vmg32__r, vmg32__h, vmg32__l, (vl));			\
  (yo) = VM32_NARROW (vmg32__r, (vl));					\
  (sgno) = VM32_VFCVT_RTZ_X_F_V (VM32_NARROW (vmg32__s, (vl)), (vl));	\
} while (0)
