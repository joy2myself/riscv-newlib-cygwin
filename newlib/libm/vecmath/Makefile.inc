## libvecmath -- RISC-V RVV vector math library.
## Built only when configured with --enable-vecmath (NEWLIB_VECMATH).
## Objects go into libvecmath.a (declared in newlib/Makefile.am), NOT libm.a.

%C%_src = \
	%D%/vfallback.c \

libvecmath_a_SOURCES += $(%C%_src)
