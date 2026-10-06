/* vm_crt0.S -- bare-metal entry glue for the libvecmath test runner.

   qemu-system (-bios none -kernel) enters the ELF with a0=hartid,
   a1=dtb and an UNDEFINED sp; libgloss' crt0 however expects a
   spike/pk-style stack whose first word is argc.  This stub sets up a
   private stack (linker-provided __stack_top) with a synthetic
   argc/argv, then falls through to the real crt0 (_start).

   Linked first; the linker script sets ENTRY(_vm_start).  */

	.section .boot,"ax"
	.globl _boot
	.type  _boot, @function
_boot:
	j	_vm_start		/* first bytes at the load base */
	.size _boot, . - _boot

	.section .text
	.globl _vm_start
	.type  _vm_start, @function
_vm_start:
	/* Enable FPU and vector unit: sstatus.FS = Initial (bits 14:13)
	   and sstatus.VS = Initial (bits 10:9).  Bare-metal reset state
	   has both Off, so every scalar FP instruction -- including the
	   fsd/fs0 saves in compiler-generated code -- and every RVV
	   instruction traps.  Linux enables them lazily in the kernel
	   and spike/pk in firmware; a -bios none boot gets neither, so
	   the startup code must.  0x2000 | 0x200 = 0x2200.  */
	li	t0, 0x2200
	csrs	sstatus, t0
	la	sp, __stack_top
	/* Synthetic process environment at sp: argc, argv[], envp[] */
	li	t0, 1
	sw	t0, 0(sp)		/* argc = 1 */
	la	t1, vm_argv
	sd	t1, 8(sp)		/* argv[0] */
	sd	zero, 16(sp)		/* argv[1] = NULL */
	sd	zero, 24(sp)		/* envp[0] = NULL */
	/* Fall into libgloss crt0 (bss clear, init array, main).  */
	tail	_start
	.size _vm_start, . - _vm_start

	.section .data
	.align 3
vm_argv:
	.dword vm_progname
	.dword 0
vm_progname:
	.asciz "runner-elf-int"
