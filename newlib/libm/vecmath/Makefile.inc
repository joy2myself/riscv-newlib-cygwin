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
	%D%/vatan2pi.c \
	%D%/vhyper.c \
	%D%/vhyperinv.c \
	%D%/vhypot.c \
	%D%/verf.c \
	%D%/vgamma.c \
	%D%/vfallback.c \
	%D%/vfabi.c \
	%D%/vf_misc.c \
	%D%/vf_exp.c \
	%D%/vf_log.c \
	%D%/vf_pow.c \
	%D%/vf_trig.c \
	%D%/vf_invtrig.c \
	%D%/vf_hyper.c \
	%D%/vf_hyperinv.c \
	%D%/vf_erf.c \
	%D%/vf_gamma.c \
	%D%/vf_fallback.c \
	%D%/vfabi32.c \
	%D%/tables/vm_exp_data.c \
	%D%/tables/vm_log_data.c \
	%D%/tables/vm_log2_data.c \
	%D%/tables/vm_pow_log_data.c

libvecmath_a_SOURCES += $(%C%_src)
