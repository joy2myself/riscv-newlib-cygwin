## libvecmath -- RISC-V RVV vector math library.
## Built only when configured with --enable-vecmath (NEWLIB_VECMATH).
## Objects go into libvecmath.a (declared in newlib/Makefile.am), NOT libm.a.

%C%_src = \
	%D%/vmisc.c \
	%D%/vexp.c \
	%D%/vexpm1.c \
	%D%/vlog.c \
	%D%/vpow.c \
	%D%/vsin.c \
	%D%/vsinpi.c \
	%D%/vatan.c \
	%D%/vhyper.c \
	%D%/vhyperinv.c \
	%D%/vhypot.c \
	%D%/vfallback.c \
	%D%/tables/vm_exp_data.c \
	%D%/tables/vm_log_data.c \
	%D%/tables/vm_log2_data.c \
	%D%/tables/vm_pow_log_data.c

libvecmath_a_SOURCES += $(%C%_src)
