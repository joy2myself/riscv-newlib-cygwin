/* Minimal stand-in for the newlib-generated "newlib.h", used only to build
   the scalar (non-vector) fallbacks of the RISC-V string/memory routines
   with a Linux/glibc cross compiler.  */
#ifndef _NEWLIB_H_STUB
#define _NEWLIB_H_STUB 1
/* qemu and the target boards both support misaligned scalar loads/stores.  */
#define _HAVE_HW_MISALIGNED_ACCESS 1
#endif
