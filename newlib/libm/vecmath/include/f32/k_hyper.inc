/* f32/k_hyper.inc -- binary32 hyperbolic kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_hyper.inc,
   which pulls k_exp.inc) active at VM_LMUL == 2*VM32_LMUL.  */

VM32_W1 (sinh, sinh)
VM32_W1F (cosh, cosh)
VM32_W1 (tanh, tanh)
