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
ENTRY(memmove)

  beqz a2, .Ldone		/* in case there are 0 bytes to be copied, return immediately */

  mv a4, a0			/* copy the destination address over to a4, since memmove should return that address in a0 at the end */
  li a3, 1
  bgtu  a1, a0, .Lcopy		/* in case of source address > destination address, copy from start to end of the specified memory area */

  li a3, -1			/* otherwhise, start copying from the end of the specified memory area in order to prevent data loss in case of overlapping memory areas.*/
  add   a4, a4, a2		/* add the number of bytes to be copied to both addresses. this gives us the address one byte past the end of the memory area we want to copy, */
  add   a1, a1, a2		/* therefore we need to subtract 1 from both addresses in the next step before starting the copying process. */

.Lincrement:
  add   a4, a4, a3 		/* in case of source address < destination address, increment both addresses by -1 before copying any data to obtain the correct start addresses */
  add   a1, a1, a3
.Lcopy:
  lbu a5, 0(a1)
  addi  a2, a2, -1		/* copy bytes as long as a2 (= the number of bytes to be copied) > 0. the increment is done here to relax the RAW dependency between load and store */
  sb a5, 0(a4)
  bnez a2, .Lincrement

.Ldone:
  ret

END(memmove)
#elif defined(__riscv_vector) && !defined(__riscv_e)
ENTRY(memmove)
	beqz	a2, .Lret
	li	t3, 2*SZREG
	bltu	a2, t3, .Lsmall		/* n < 16: direction-agnostic */
	beq	a0, a1, .Lret
	bgeu	a1, a0, .Lfwd	/* src >= dst: forward */
	sub	t2, a0, a1
	bgeu	t2, a2, .Lfwd	/* disjoint: forward */
	add	t0, a0, a2		/* dst end */
	add	t1, a1, a2		/* src end */
	vsetvli	t2, a2, e8, m8, ta, ma
	bltu	a2, t2, .Lbone
	slli	t3, t2, 1
	bltu	a2, t3, .Lblast
.Lbloop:
	sub	t1, t1, t3
	sub	t0, t0, t3
	vle8.v	v0, (t1)
	add	t4, t1, t2
	vle8.v	v8, (t4)
	vse8.v	v0, (t0)
	add	t5, t0, t2
	vse8.v	v8, (t5)
	sub	a2, a2, t3
	bgeu	a2, t3, .Lbloop
.Lblast:
	beqz	a2, .Lret
1:	bltu	a2, t2, .Lbtail
	sub	t1, t1, t2
	sub	t0, t0, t2
	vle8.v	v0, (t1)
	vse8.v	v0, (t0)
	sub	a2, a2, t2
	beqz	a2, .Lret
	j	1b
.Lbtail:
	vsetvli	t3, a2, e8, m8, ta, ma
	sub	t1, t1, t3
	sub	t0, t0, t3
	vle8.v	v0, (t1)
	vse8.v	v0, (t0)
	ret
.Lbone:
	sub	t1, t1, t2
	sub	t0, t0, t2
	vle8.v	v0, (t1)
	vse8.v	v0, (t0)
	ret
.Lfwd:
	mv	t0, a0
	mv	t1, a1
	vsetvli	t2, a2, e8, m8, ta, ma
	bltu	a2, t2, .Lfone
	slli	t3, t2, 1
	add	t4, t1, t2
	add	t5, t0, t2
	bltu	a2, t3, .Lflast
.Lfloop:
	vle8.v	v0, (t1)
	vle8.v	v8, (t4)
	vse8.v	v0, (t0)
	vse8.v	v8, (t5)
	add	t1, t1, t3
	add	t0, t0, t3
	add	t4, t4, t3
	add	t5, t5, t3
	sub	a2, a2, t3
	bgeu	a2, t3, .Lfloop
.Lflast:
	beqz	a2, .Lret
1:	bltu	a2, t2, .Lftail
	vle8.v	v0, (t1)
	vse8.v	v0, (t0)
	add	t1, t1, t2
	add	t0, t0, t2
	sub	a2, a2, t2
	beqz	a2, .Lret
	j	1b
.Lftail:
	vsetvli	t3, a2, e8, m8, ta, ma
	vle8.v	v0, (t1)
	vse8.v	v0, (t0)
	ret
.Lfone:
	vle8.v	v0, (t1)
	vse8.v	v0, (t0)
.Lret:
	ret
.Lsmall:
	li	t3, SZREG
	bltu	a2, t3, .Ltiny
	REG_L	t3, 0(a1)		/* both loads before any store */
	add	t4, a1, a2
	REG_L	t5, -SZREG(t4)
	add	t4, a0, a2
	REG_S	t5, -SZREG(t4)	/* high word first */
	REG_S	t3, 0(a0)
	ret
.Ltiny:
	beq	a0, a1, .Lret
	bltu	a0, a1, .Ltinyf
	add	t0, a0, a2
	add	t1, a1, a2
1:	addi	t0, t0, -1
	addi	t1, t1, -1
	lbu	t3, 0(t1)
	sb	t3, 0(t0)
	addi	a2, a2, -1
	bnez	a2, 1b
	ret
.Ltinyf:
	mv	t0, a0
	mv	t1, a1
1:	lbu	t3, 0(t1)
	sb	t3, 0(t0)
	addi	t1, t1, 1
	addi	t0, t0, 1
	addi	a2, a2, -1
	bnez	a2, 1b
	ret
END(memmove)
#endif
