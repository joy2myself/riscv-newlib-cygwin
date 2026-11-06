/* vm_data.h -- self-contained copies of the libm tables used by the
   vecmath kernels.

   libvecmath does not reference libm's __exp_data/__log_data/... symbols:
   their struct layouts differ between C libraries (glibc orders the
   exp_data fields differently than newlib) and both may be hidden, so
   linking against whatever libm is present would silently corrupt the
   kernels.  The tables instead live in libm/vecmath/tables/*.c under
   __vecmath_* names, extracted byte-exactly from the newlib objects by
   test/gen/extract_table.py.

   Struct layouts mirror newlib libm/common/math_config.h for rv64
   hard-float (HAVE_FAST_FMA=1, EXP_TABLE_BITS=7, EXP_POLY_ORDER=5,
   EXP2_POLY_ORDER=5, LOG_TABLE_BITS=7, LOG_POLY_ORDER=6,
   LOG_POLY1_ORDER=12, LOG2_TABLE_BITS=6, LOG2_POLY_ORDER=7,
   LOG2_POLY1_ORDER=11, POW_LOG_TABLE_BITS=7, POW_LOG_POLY_ORDER=8).
   Keep in sync; the accuracy suite catches any drift.  */

#ifndef VM_DATA_H
#define VM_DATA_H

#include <stdint.h>

struct vm_exp_data
{
  double invln2N;
  double shift;
  double negln2hiN;
  double negln2loN;
  double poly[4];		/* exp poly C2..C5 */
  double exp2_shift;
  double exp2_poly[5];		/* exp2 poly C1..C5 */
  uint64_t tab[256];		/* (tail bits, scale bits) pairs */
};

struct vm_log_data
{
  double ln2hi;
  double ln2lo;
  double poly[5];		/* LOG_POLY_ORDER-1, first coeff is 1 */
  double poly1[11];		/* LOG_POLY1_ORDER-1, near-1 poly */
  struct { double invc, logc; } tab[128];
};

struct vm_log2_data
{
  double invln2hi;
  double invln2lo;
  double poly[6];		/* LOG2_POLY_ORDER-1 */
  double poly1[10];		/* LOG2_POLY1_ORDER-1 */
  struct { double invc, logc; } tab[64];
};

struct vm_pow_log_data
{
  double ln2hi;
  double ln2lo;
  double poly[7];		/* POW_LOG_POLY_ORDER-1 */
  struct { double invc, pad, logc, logctail; } tab[128];
};

extern const struct vm_exp_data __vecmath_exp_data;
extern const struct vm_log_data __vecmath_log_data;
extern const struct vm_log2_data __vecmath_log2_data;
extern const struct vm_pow_log_data __vecmath_pow_log_data;

#endif /* VM_DATA_H */
