/* ulp.h -- ULP error evaluation for the libvecmath runner.  */

#ifndef ULP_H
#define ULP_H

#include <stdint.h>
#include "vmblob.h"

typedef struct {
  double ulp;		/* ULP error (>=0); meaningful iff class_ok */
  int class_ok;		/* class/bitwise expectations satisfied */
  int flags_ok;		/* gated flags match (set by caller) */
} ulp_verdict_t;

/* Evaluate one single-output result y against its record.
   Pure and allocation-free; uses double + fma only (no long double),
   so the runner is portable across targets.  */
ulp_verdict_t ulp_eval (double y, const vmb_rec1_t *r);

/* Explicit-field core (shared with the dual-output evaluator).  */
ulp_verdict_t ulp_eval_core (double y, uint32_t expect, uint64_t ref_hi,
			     uint64_t ref_lo, uint64_t exact_bits);

/* Second output of a dual-output record (sincos/sincospi).  */
ulp_verdict_t ulp_eval2 (double y2, const vmb_rec2_t *r2, uint32_t expect);

/* ULP distance between two finite doubles (for cross-check reporting);
   returns a huge value when either is not finite or classes differ.  */
double ulp_distance (double got, double want);

#endif /* ULP_H */
