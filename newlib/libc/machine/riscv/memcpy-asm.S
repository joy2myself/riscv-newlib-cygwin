/* Copyright (c) 2019  SiFive Inc. All rights reserved.

   This copyrighted material is made available to anyone wishing to use,
   modify, copy, or redistribute it subject to the terms and conditions
   of the FreeBSD License.  This program is distributed in the hope that
   it will be useful, but WITHOUT ANY WARRANTY expressed or implied,
   including the implied warranties of MERCHANTABILITY or FITNESS FOR
   A PARTICULAR PURPOSE.  A copy of this license is available at
   http://www.opensource.org/licenses.
*/

#include <sys/asm.h>

#if defined(PREFER_SIZE_OVER_SPEED) || defined(__OPTIMIZE_SIZE__)
ENTRY(memcpy)

  mv a3, a0
  beqz a2, 2f

1:
  lbu a4, 0(a1)
  sb a4, 0(a3)
  addi   a2, a2, -1
  addi   a3, a3, 1
  addi   a1, a1, 1
  bnez a2, 1b

2:
  ret
END(memcpy)
#elif defined(__riscv_vector) && !defined(__riscv_e)
ENTRY(memcpy)
	mv	t0, a0			/* running dst */
	mv	t1, a1			/* running src */
	beqz	a2, .Lret
	li	t3, 16
	bltu	a2, t3, .Lsmall
	vsetvli	t2, a2, e8, m8, ta, ma	/* t2 = min(n, VLMAX) */
	bltu	a2, t2, .Lone		/* n < VLMAX: one exact strip */
	slli	t3, t2, 1		/* 2 strips per iteration */
	add	t4, t1, t2
	add	t5, t0, t2
	bltu	a2, t3, .Llast
.Lloop:
	vle8.v	v0, (t1)
	vle8.v	v8, (t4)
	vse8.v	v0, (t0)
	vse8.v	v8, (t5)
	add	t1, t1, t3
	add	t0, t0, t3
	add	t4, t4, t3
	add	t5, t5, t3
	sub	a2, a2, t3
	bgeu	a2, t3, .Lloop
.Llast:
	beqz	a2, .Lret
1:	bltu	a2, t2, .Ltail
	vle8.v	v0, (t1)
	vse8.v	v0, (t0)
	add	t1, t1, t2
	add	t0, t0, t2
	sub	a2, a2, t2
	beqz	a2, .Lret
	j	1b
.Ltail:
	add	t1, t1, a2		/* last full strip, overlapping */
	add	t0, t0, a2
	sub	t1, t1, t2
	sub	t0, t0, t2
	vle8.v	v0, (t1)
	vse8.v	v0, (t0)
	ret
.Lone:
	vle8.v	v0, (t1)
	vse8.v	v0, (t0)
.Lret:
	ret
.Lsmall:
	li	t3, SZREG
	bltu	a2, t3, .Ltiny
	REG_L	t3, 0(t1)
	add	t4, t1, a2
	REG_L	t5, -SZREG(t4)
	REG_S	t3, 0(t0)
	add	t4, t0, a2
	REG_S	t5, -SZREG(t4)
	ret
.Ltiny:
	lbu	t3, 0(t1)
	sb	t3, 0(t0)
	addi	t1, t1, 1
	addi	t0, t0, 1
	addi	a2, a2, -1
	bnez	a2, .Ltiny
	ret
END(memcpy)
#endif
