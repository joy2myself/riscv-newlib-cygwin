/* f32/k_trig.inc -- binary32 trigonometric kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_trig.inc)
   active at VM_LMUL == 2*VM32_LMUL.  */

VM32_W1 (sin, sin)
VM32_W1 (cos, cos)
VM32_W1 (tan, tan)
VM32_W1 (sinpi, sinpi)
VM32_W1 (cospi, cospi)
VM32_W1 (tanpi, tanpi)

/* Two-output kernels: the binary64 pair macro writes two f64 vectors,
   each narrowed separately.  */
#define VM32_SINCOS(sy, cy, x, vl) do {					\
  vm_f_t vmsc32__s, vmsc32__c;						\
  VM_SINCOS (vmsc32__s, vmsc32__c, VM32_WIDEN ((x), (vl)), (vl));	\
  (sy) = VM32_NARROW (vmsc32__s, (vl));					\
  (cy) = VM32_NARROW (vmsc32__c, (vl));					\
} while (0)

#define VM32_SINCOSPI(sy, cy, x, vl) do {				\
  vm_f_t vmscp32__s, vmscp32__c;					\
  VM_SINCOSPI (vmscp32__s, vmscp32__c, VM32_WIDEN ((x), (vl)),		\
	       VM_PI_ALL ((vl)), (vl));					\
  (sy) = VM32_NARROW (vmscp32__s, (vl));				\
  (cy) = VM32_NARROW (vmscp32__c, (vl));				\
} while (0)
