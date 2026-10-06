## libvecmath -- RISC-V RVV vector math library.
## Built only when configured with --enable-vecmath (NEWLIB_VECMATH).
## Objects go into libvecmath.a (declared in newlib/Makefile.am), NOT libm.a.

%C%_src = \
	%D%/vmisc.c \
	%D%/vpow.c \
	%D%/vfallback.c \
	%D%/tables/vm_pow_log_data.c

libvecmath_a_SOURCES += $(%C%_src)
