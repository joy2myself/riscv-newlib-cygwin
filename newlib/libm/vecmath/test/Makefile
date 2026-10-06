# libvecmath test framework -- plain GNU make (not part of the automake build).
#
# Targets:
#   vmgen          host reference generator (needs libmpfr)
#   blobs-small    SMALL tier blobs (2k records/fn; specials + sampled randoms)
#   blobs-full     FULL tier blobs (100k records/fn)
#   runner-linux   static linux runner (cross gcc, rv64gcv) for qemu/board
#   runner-elf     bare-metal runner (semihosting) for spike
#   run-qemu       qemu-riscv64 VLEN matrix over blobs-full
#   run-spike      spike VLEN matrix over blobs-small
#   clean

TESTDIR  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
VECDIR   := $(abspath $(TESTDIR)/..)
# SRCROOT is the newlib/ source directory.
SRCROOT  := $(abspath $(VECDIR)/../..)

HOSTCC    ?= gcc
HOSTCFLAGS ?= -O2 -Wall -Wextra
MPFR_LIBS ?= -lmpfr -lm

CROSS_LINUX ?= /opt/riscv/bin/riscv64-unknown-linux-gnu-gcc
CROSS_ELF   ?= riscv64-unknown-elf-gcc
QEMU        ?= $(HOME)/qemu-build/qemu-riscv64
SPIKE       ?= spike

ARCH   ?= -march=rv64gcv -mabi=lp64d
DEFS   := -DVECMATH_STANDALONE
# Not -I$(SRCROOT)/libc/include: that would shadow the toolchain's own
# <math.h>/<fenv.h> with newlib's.  Only vecmath.h is needed from the source
# tree, exposed through a shim directory symlink.
INCS   := -I$(TESTDIR)/include -I$(TESTDIR)/runner \
	  -I$(VECDIR)/include -I$(VECDIR) -I$(TESTDIR)/shim

$(TESTDIR)/shim/vecmath.h: $(SRCROOT)/libc/include/vecmath.h
	@mkdir -p $(TESTDIR)/shim
	@ln -sf $< $@

# Library sources under test.
VECSRCS := $(wildcard $(VECDIR)/v*.c) $(wildcard $(VECDIR)/tables/*.c)
VECHDRS := $(wildcard $(VECDIR)/include/*.h) $(wildcard $(VECDIR)/include/*.inc) \
	   $(wildcard $(VECDIR)/*.h) $(SRCROOT)/libc/include/vecmath.h
RUNSRCS := $(TESTDIR)/runner/runner.c $(TESTDIR)/runner/ulp.c
RUNHDRS := $(wildcard $(TESTDIR)/include/*.h) $(wildcard $(TESTDIR)/include/*.def) \
	   $(TESTDIR)/runner/ulp.h

QEMU_VLENS  ?= 128 256 512 1024
SPIKE_VLENS ?= 128 256 1024

.PHONY: all blobs-small blobs-full run-qemu run-qemu-small run-qemu-system run-spike clean results
all: vmgen runner-linux

vmgen: $(TESTDIR)/gen/vmgen.c $(TESTDIR)/include/vmblob.h $(TESTDIR)/include/vmfuncs.def
	$(HOSTCC) $(HOSTCFLAGS) -I$(TESTDIR)/include -o $@ $< $(MPFR_LIBS)

blobs-small: vmgen
	@mkdir -p blobs/SMALL
	./vmgen --outdir blobs/SMALL --tier small

blobs-full: vmgen
	@mkdir -p blobs/FULL
	./vmgen --outdir blobs/FULL --tier full

runner-linux: $(RUNSRCS) $(VECSRCS) $(RUNHDRS) $(VECHDRS) $(TESTDIR)/shim/vecmath.h
	$(CROSS_LINUX) -O2 -static $(ARCH) $(DEFS) $(INCS) -o $@ $(RUNSRCS) $(VECSRCS) -lm

runner-elf: $(RUNSRCS) $(VECSRCS) $(RUNHDRS) $(VECHDRS) $(TESTDIR)/shim/vecmath.h
	$(CROSS_ELF) -O2 $(ARCH) $(DEFS) $(INCS) -o $@ $(RUNSRCS) $(VECSRCS) \
		--specs=semihost.specs -lm

# Integrated bare-metal runner: links against the INSTALLED libvecmath.a in
# a medany newlib+libgloss sysroot (validates configure --enable-vecmath ->
# make install -> -lvecmath toolchain end-to-end).  Requires:
#   SYSROOT_ELFMEDANY/riscv64-unknown-elf/{include,lib} populated, e.g.
#   newlib:  ../newlib/configure --host=riscv64-unknown-elf \
#              --target=riscv64-unknown-elf --enable-vecmath --disable-multilib \
#              --prefix=$$SYSROOT CFLAGS="-O2 -march=rv64gcv -mabi=lp64d -mcmodel=medany"
#   libgloss: same prefix, CC with -isystem $$SYSROOT/riscv64-unknown-elf/include
SYSROOT_ELFMEDANY ?= $(HOME)/vecmath-build/inst-elf-medany
QEMU_SYSTEM ?= $(HOME)/qemu-build/qemu-system-riscv64
ELFSYSROOT := $(SYSROOT_ELFMEDANY)/riscv64-unknown-elf

vm_crt0.o: $(TESTDIR)/runner/vm_crt0.S
	$(CROSS_ELF) -c -mcmodel=medany $(ARCH) -o $@ $<

runner-elf-int: $(RUNSRCS) $(RUNHDRS) $(VECHDRS) vm_crt0.o
	$(CROSS_ELF) -O2 $(ARCH) -mcmodel=medany \
		-I$(ELFSYSROOT)/include -I$(TESTDIR)/include -I$(TESTDIR)/runner \
		-o $@ vm_crt0.o $(RUNSRCS) \
		-B$(ELFSYSROOT)/lib --specs=semihost.specs \
		-specs=$(TESTDIR)/scripts/baremetal.specs \
		-Wl,-T,$(TESTDIR)/scripts/baremetal.ld \
		-L$(ELFSYSROOT)/lib \
		-Wl,--whole-archive -lvecmath -Wl,--no-whole-archive -lm

run-qemu-system: runner-elf-int blobs-small results
	@rc=0; for V in $(SPIKE_VLENS); do \
	  echo "=== qemu-system vlen=$$V ==="; \
	  timeout 600 $(QEMU_SYSTEM) -M virt -cpu rv64,v=true,vlen=$$V \
	    -bios none -kernel ./runner-elf-int -nographic -m 1G \
	    -semihosting-config enable=on,target=native \
	    || rc=1; \
	done; exit $$rc

results:
	@mkdir -p results

run-qemu: runner-linux blobs-full results
	@rc=0; for V in $(QEMU_VLENS); do \
	  echo "=== qemu VLEN=$$V ==="; \
	  $(QEMU) -cpu rv64,v=true,v=true,vlen=$$V,vext_spec=v1.0 \
	    ./runner-linux --blobs blobs/FULL --report results/qemu-$$V.md \
	    || rc=1; \
	done; exit $$rc

run-qemu-small: runner-linux blobs-small results
	@rc=0; for V in $(QEMU_VLENS); do \
	  echo "=== qemu VLEN=$$V (small) ==="; \
	  $(QEMU) -cpu rv64,v=true,vlen=$$V,vext_spec=v1.0 \
	    ./runner-linux --blobs blobs/SMALL --report results/qemu-small-$$V.md \
	    || rc=1; \
	done; exit $$rc

run-spike: runner-elf blobs-small results
	@rc=0; for V in $(SPIKE_VLENS); do \
	  echo "=== spike vlen=$$V ==="; \
	  $(SPIKE) --isa=RV64GCV --varch=vlen:$$V ./runner-elf \
	    --blobs blobs/SMALL --report results/spike-$$V.md || rc=1; \
	done; exit $$rc

clean:
	rm -rf vmgen runner-linux runner-elf blobs results shim
