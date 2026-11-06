/* f32/k_erf.inc -- binary32 erf / erfc kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_erf.inc,
   which pulls k_exp.inc and vecmath_dd.h) active at
   VM_LMUL == 2*VM32_LMUL.  */

VM32_W1 (erf, erf)
VM32_W1 (erfc, erfc)
