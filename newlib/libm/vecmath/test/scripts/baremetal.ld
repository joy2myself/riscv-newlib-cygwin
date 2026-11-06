/* baremetal.ld -- minimal linker script for the libvecmath test runner
   on qemu-system-riscv64 virt (RAM base 0x80000000).

   Some qemu builds with `-bios none -kernel X.elf` start the first hart
   at the ELF's LOAD BASE rather than e_entry.  To work under both
   conventions, the first bytes at 0x80000000 are a .boot jump stub
   (file-offset separation is automatic: ld places .boot after the ELF
   headers in the file while its vaddr is the segment base), and ENTRY
   also points at the stub.

   Everything is compiled -mcmodel=medany, so the 0x80000000 base is
   fine for the prebuilt (medlow) newlib/libgloss archives too: all
   symbols stay within the same 2 GiB window.  */

OUTPUT_ARCH(riscv)
ENTRY(_boot)

SECTIONS
{
  . = 0x80000000;

  .boot : { *(.boot) }
  . = ALIGN(8);

  .text : {
    *(.text.unlikely .text.*_unlikely)
    *(.text.startup .text.startup.*)
    *(.text .text.*)
    *(.gnu.linkonce.t.*)
  }

  . = ALIGN(8);
  PROVIDE(__etext = .);

  .rodata : { *(.rodata .rodata.*) *(.gnu.linkonce.r.*) }
  .srodata : { *(.srodata*) }
  .eh_frame : { *(.eh_frame*) }
  .init_array : {
    PROVIDE_HIDDEN(__init_array_start = .);
    *(.init_array*)
    PROVIDE_HIDDEN(__init_array_end = .);
  }
  .fini_array : {
    PROVIDE_HIDDEN(__fini_array_start = .);
    *(.fini_array*)
    PROVIDE_HIDDEN(__fini_array_end = .);
  }

  . = ALIGN(8);
  .data : { *(.data .data.*) *(.gnu.linkonce.d.*) }
  .sdata : { *(.sdata .sdata.*) *(.gnu.linkonce.s.*) }

  . = ALIGN(8);
  PROVIDE(__global_pointer$ = . + 0x800);
  __bss_start = .;
  .sbss : { *(.sbss*) *(.gnu.linkonce.sb.*) }
  .bss : { *(.bss .bss.*) *(.gnu.linkonce.b.*) *(COMMON) }

  . = ALIGN(16);
  _end = .;
  PROVIDE(end = .);
  PROVIDE(__stack_top = _end + 0x800000);	/* 8 MiB stack above bss */

  /* HTIF symbols (spike compatibility; unused by qemu semihosting).  */
  .tohost : { *(.tohost) }
}
