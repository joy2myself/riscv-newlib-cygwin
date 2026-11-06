/* f32/k_hyperinv.inc -- binary32 inverse hyperbolic kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels
   (k_hyperinv.inc) active at VM_LMUL == 2*VM32_LMUL.  */

VM32_W1 (asinh, asinh)
VM32_W1 (acosh, acosh)
VM32_W1 (atanh, atanh)
