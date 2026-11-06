/* f32/k_invtrig.inc -- binary32 inverse trigonometric kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_invtrig.inc)
   active at VM_LMUL == 2*VM32_LMUL.  */

VM32_W1 (asin, asin)
VM32_W1 (acos, acos)
VM32_W1 (atan, atan)
VM32_W2 (atan2, atan2)
VM32_W1 (asinpi, asinpi)
VM32_W1 (acospi, acospi)
VM32_W1 (atanpi, atanpi)
VM32_W2 (atan2pi, atan2pi)
