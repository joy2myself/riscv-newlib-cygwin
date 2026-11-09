/* k_fixup.inc -- masked scalar-fixup epilogue (LMUL-generic template).

   Lanes selected by the mask are compressed out, passed one by one
   through a scalar libm function, and scattered back via viota plus
   indexed gather; results and FP exception flags are identical to the
   scalar path by construction.  The common case (no selected lanes)
   costs one vfirst.

   Buffer bound: vl <= VLMAX(e64, LMUL <= 4) = VLEN/16 lanes; 300 doubles
   cover VLEN up to 4096 at m4.  Trailing inactive lanes get viota index
   == popcount, hence the slack over 256.

   The epilogue is deliberately out of line: its 4.8 KB of stack buffers
   would otherwise be charged (frame setup, stack probing) to every strip
   of every kernel that carries a fixup, even when no lane takes it --
   worth 30-45% of the hot-loop throughput on the X60.  */

__attribute__ ((noinline)) static vm_f_t
VM_KFN (scalar_fixup) (vm_f_t x, vm_f_t y, vm_b_t m, size_t vl,
		       double (*fn) (double))
{
  if (VM_VFIRST_M (m, vl) < 0)
    return y;
  double xbuf[300], ybuf[300];
  size_t n = (size_t) VM_VCMP_POPC_M (m, vl);
  vm_f_t xc = VM_VCOMPRESS (x, m, vl);
  VM_VSE64 (xbuf, xc, n);
  for (size_t i = 0; i < n; i++)
    ybuf[i] = fn (xbuf[i]);
  vm_u64_t idx = VM_VIOTA_M (m, vl);
  vm_f_t picked = VM_VLUXEI64 (ybuf, VM_VSLL_VX (idx, 3, vl), vl);
  return VM_VFMERGE_VVM (y, picked, m, vl);
}

/* Two-input variant (pow & co): fn(x, y).  */
__attribute__ ((noinline)) static vm_f_t
VM_KFN (scalar_fixup2) (vm_f_t x, vm_f_t y, vm_f_t r, vm_b_t m, size_t vl,
			double (*fn) (double, double))
{
  if (VM_VFIRST_M (m, vl) < 0)
    return r;
  double xbuf[300], ybuf[300], rbuf[300];
  size_t n = (size_t) VM_VCMP_POPC_M (m, vl);
  VM_VSE64 (xbuf, VM_VCOMPRESS (x, m, vl), n);
  VM_VSE64 (ybuf, VM_VCOMPRESS (y, m, vl), n);
  for (size_t i = 0; i < n; i++)
    rbuf[i] = fn (xbuf[i], ybuf[i]);
  vm_u64_t idx = VM_VIOTA_M (m, vl);
  vm_f_t picked = VM_VLUXEI64 (rbuf, VM_VSLL_VX (idx, 3, vl), vl);
  return VM_VFMERGE_VVM (r, picked, m, vl);
}

/* Guarded entry points for the two epilogues above.  The call itself is
   what costs: passing lm-group vectors by value makes the call site
   spill its live vector registers and build a frame, roughly 40 ns per
   element on the X60 even when the mask is empty.  A vfirst at the call
   site keeps that off the hot path; the fallback value is the vector the
   unguarded call would have returned unchanged.  */
#define VM_FIXUP1(x_, y_, m_, vl_, fn_)					\
  (VM_VFIRST_M ((m_), (vl_)) < 0					\
   ? (y_)								\
   : VM_KFN (scalar_fixup) ((x_), (y_), (m_), (vl_), (fn_)))

#define VM_FIXUP2(x_, y_, r_, m_, vl_, fn_)				\
  (VM_VFIRST_M ((m_), (vl_)) < 0					\
   ? (r_)								\
   : VM_KFN (scalar_fixup2) ((x_), (y_), (r_), (m_), (vl_), (fn_)))
