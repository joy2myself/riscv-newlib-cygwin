# libc-funcs — RVV vs glibc correctness and throughput harness

Self-contained test infrastructure for the hand-written RVV 1.0 versions of
the C string/memory functions in `newlib/libc/machine/riscv/`:

| harness | what it does |
|---|---|
| `corr/`  | differential correctness: every vector routine is compared against the platform (glibc) routine of the same name, on identical input, value for value and byte for byte |
| `bench/` | throughput in GB/s and ns/call, `rvv_*` vs glibc, printed as a table and as CSV |

Nothing in the newlib tree is modified and no git command is used.  Each
`*-asm.S` is assembled exactly as shipped and its entry symbol is renamed:

```
objcopy --redefine-sym strlen=rvv_strlen strlen-asm.o rvv_strlen.o
```

so one binary can call both `rvv_strlen` and glibc's `strlen`.  The scalar
(non-vector) fallbacks `str*.c` / `mem*.c` are built the same way with
`-march=rv64gc` and renamed to `scalar_*`; those that pull in newlib-private
headers are skipped (the driver references them weakly and only uses the ones
that link).

## Quick start

```sh
./build.sh          # cross-compile both harnesses (static, -march=rv64gcv)
./run.sh            # correctness at VLEN 128/256/512, then throughput
```

`run.sh` writes `results/corr-vlen<N>.txt` and `results/bench-vlen<N>.{txt,csv}`.
Correctness prints one `[PASS]`/`[FAIL]` line per function plus a final
`## VLEN <n> bits: <checks> checks, <fails> failures` line and exits non-zero
if anything failed.

Environment (all optional):

| variable | default | meaning |
|---|---|---|
| `CC` | `/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc` | compiler |
| `OBJCOPY`, `NM` | derived from `CC` | binutils |
| `ARCH` | `-march=rv64gcv -mabi=lp64d` | target flags |
| `ARCH_NOV` | `-march=rv64gc -mabi=lp64d` | flags for the scalar fallbacks |
| `OPT` | `-O2 -fno-builtin -Wall -Wextra` | compile flags |
| `SRC` | `../../../../libc/machine/riscv` | newlib source dir |
| `BUILD` | `./build` | output dir |
| `QEMU` | `$HOME/qemu-build/qemu-riscv64` | emulator; `QEMU=native` runs directly |
| `VLENS` | `128 256 512` | VLEN matrix |
| `CORR_ARGS` | – | restrict correctness to listed functions |

On a RISC-V Linux board (gcc 14.2 / gcc 15, glibc):

```sh
CC=gcc ARCH="-march=rv64gcv -mabi=lp64d" ./build.sh
QEMU=native VLENS=256 ./run.sh
# or a single harness:
QEMU=native VLENS=256 CORR_ARGS="strlen memcpy" ./run.sh
./build/bench --csv bench.csv
```

The bench binary is plain POSIX C (`clock_gettime`, `mmap` only in `corr/`) and
uses the same objcopy trick, so it builds unchanged with the board toolchains.

## What the correctness harness checks

For every one of the 26 routines:

```
strlen strnlen strchr strchrnul strrchr strcmp strncmp strcpy strncpy
strcat strncat strspn strcspn strpbrk strstr strcasecmp strncasecmp
memcpy memmove memccpy mempcpy memchr memrchr memcmp rawmemchr memset
```

* **lengths** `0..200`, `255/256/257/511/512/513`, `997`, `4096`, `100003`,
  with **all 16 source and all 16 destination alignments** (256 combinations
  for the main case of each function; the secondary cases use a reduced but
  still boundary-covering alignment set).  Because the length sweeps 0..200 at
  every alignment, the NUL (or the end of the range) lands at every offset
  0..15 relative to a 16/32/64/128-byte vector boundary — i.e. at the
  `e8`/`m1`..`m8` strip boundaries for VLEN 128/256/512.
* **content**: eight pseudo-random 4096-byte patterns, including embedded
  `0x00`, `0x80`, `0xff`, all of `0x80..0xff`, mixed case (`AaBb…`), the
  non-letter neighbours of the case fold (`@`, `[`, `` ` ``, `{`, `~`), and an
  all-`'a'` / `"abab…"` pattern for `strstr`/`strspn`.  Patterns are copied in
  4096-byte chunks so the effective period is 4096: an implementation that
  shifts data by a whole vector chunk cannot hide behind a short repeat.
* **string semantic edges**: NUL at every position 0..200; `strchr`/`strrchr`/
  `strchrnul` with `c = 0`, absent, first/last byte, `0x80`, `0xff`, and
  out-of-range (`0x141`, must be truncated to `char`); `strcmp`/`strncmp` with
  equal / differ-at-0 / differ-at-mid / differ-at-last / prefix pairs,
  `0x80` vs `0x7f` (unsigned ordering), case-only differences, `n = 0`,
  `n = len/2`, `n = len`, `n = len+1`; `strncpy` with `n` shorter / equal /
  longer than the source (must zero-fill) and `n` that cuts a vector chunk;
  `strncat` truncation, `strncat(…, 0)`, `strncat` where the source is exactly
  `n` bytes; `strspn`/`strcspn`/`strpbrk` with an empty set, a one-byte set, a
  set larger than the haystack, a 255-byte set, and a set of high bytes;
  `strstr` with an empty needle, an absent needle, the whole haystack, a
  needle at position 0 / middle / last, a needle longer than the haystack;
  `strcasecmp`/`strncasecmp` on case-only differences and on the fold
  boundaries; `memcmp` with `n = 0`, equal, and differences at 0 / mid / last;
  `memccpy` with `c` present at 0 / mid / last and absent.
* **overwrite detection**: each destination is a canary-filled window
  (128 bytes before and after the payload); the complete window of the rvv and
  glibc runs is compared afterwards, so one byte written past `n` (or before
  the payload) is reported as `trailing guard byte at payload offset <n>` with
  both byte values.
* **overlapping `memmove`**: `dst < src`, `dst > src`, `dst == src`,
  distances 1..40, plus adjacent (`d = n`) and far (`d = 1024`), for every
  length and alignment, comparing the whole region afterwards.
* **page boundary (4 KiB)**: `mmap` two pages, `mprotect` the second
  `PROT_NONE`, and place the NUL / end of range so that the last byte the
  routine may touch is the last byte of the mapped page (and a mirrored case
  with the guard page *before* the buffer).  A routine that reads or writes
  one byte too far takes `SIGSEGV`; a handler `siglongjmp`s back and the case
  is reported as a failure instead of killing the run.  Covered for every
  function, with the NUL/end at 0/1/3/7/8/15 bytes before the boundary and
  lengths up to 2048.  This is the direct test for "a vector load that is not
  fault-only-first", for "read one element past the NUL", and for "write one
  byte too many".
* **exhaustive small-alphabet enumeration**: every haystack over `{a,b}` up to
  length 5 against every needle up to length 5 (`strstr`), every string over
  `{a,A,0x80}` up to length 3 against every other (`strcmp` family, all `n`),
  every string over `{a,b,c}` up to length 4 against all 8 subsets
  (`strspn`/`strcspn`/`strpbrk`), and every string over `{a,b}` up to length 5
  for every test character (`strchr` family).
* **failure report** includes the function, the case (length, alignments,
  n, character, pattern, needle), and the first differing byte:
  `FAIL strcpy len=17 sa=3 da=9 : dest: payload byte 5 differs: rvv=0x61 glibc=0x00`.

### Guard-page check (why it matters)

`vle8ff.v` is the only safe way to load a vector that may run into an
unmapped page: it loads element 0 (faulting if that is unmapped) and stops at
the first faulting element without raising an exception.  A plain `vle8.v`
whose last lane crosses into the next page raises `SIGSEGV`.  The
`edge:*` cases in `corr/edge.c` are exactly that scenario; the harness was
validated by linking a `strlen` mutant that uses `vle8.v` and confirming it is
reported as a guard-page fault, and by an off-by-one `strlen` mutant that is
reported as a value mismatch.

## Throughput harness

`bench/bench.c` measures `memcpy`, `memset`, `memmove`, `memcmp`, `strlen`,
`strcmp`, `strchr`, `strstr` (and a candidate-heavy `strstr-cand`) at sizes
8/16/32/64/128/256/1024/4096/65536.  Each measurement is a warm-up round
followed by the best of 3 timed rounds; the call goes through a `volatile`
function-pointer table with an `asm` barrier so neither glibc's pure functions
nor the wrappers can be hoisted out of the loop.  Sizes are chosen so each
round moves ~8 MiB (minimum 64 calls, `strstr-cand` is capped at 4096 bytes
and 256 calls because it is quadratic).  "GB/s" is payload bytes per call
divided by the time — one direction only.

`--csv FILE` writes `vlen,function,size,rvv_ns,glibc_ns,rvv_gbs,glibc_gbs,ratio`.

### Caveat: qemu numbers

qemu-user emulates `vle8ff.v` element by element, which makes every
fault-only-first scan roughly an order of magnitude slower than the same
routine on hardware, and it emulates vector loads/stores much more slowly
than scalar `ld`/`sd`.  The qemu table therefore **understates** the vector
routines; use it to catch algorithmic problems (see `strstr-cand`) and run
`bench` natively on the boards for performance decisions.  The fixed cost of
one call under qemu is ~60–70 ns, which dominates the 8/16/32-byte rows.

## Layout

```
build.sh              build both harnesses
run.sh                run the VLEN matrix + throughput
corr/harness.[ch]     buffers, canaries, guard pages, dispatch, reporting
corr/tests_str1..4.c  string routines
corr/tests_mem1..2.c  memory routines
corr/tests_exh.c      exhaustive small-alphabet tests
corr/edge.c           page-boundary (guard page) cases
corr/main.c           driver, per-function summary, scalar cross-check
bench/bench.c         throughput harness
scripts/include/      minimal "newlib.h" stub for the scalar fallbacks
build/ results/       generated (not part of the deliverable)
```
