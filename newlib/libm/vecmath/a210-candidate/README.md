# A210 tested array candidate snapshot

This directory preserves the exact boundary-dispatch source and selection manifest used by the final target66 candidate. It is experimental and is not wired into the standard newlib configure/Makefile build. The manifest selects RVV or scalar array implementations per interface; boundary wrappers override selected ranges. Additional public wrappers exist beyond the 66-interface report scope.

The base repository revision is 1e4bfac6f21a78765ae6f5174568413993073ee3. The tested candidate combines two separately renamed array archives, the dispatch object, and a same-source isolated scalar math archive. Host glibc math is not the scalar baseline. Symbol maps document the archive namespace transformations.

The test archive SHA256 is recorded in tested-artifact-identities.json. This snapshot preserves source and artifact identity, not a claim that the base Git revision alone reproduces every prebuilt archive. Existing kernel edits in the original working tree were not included: their timestamps postdate the archived array builds and exact provenance has not been proved. Binary archives are deliberately not committed.

To assemble against the original prebuilt inputs:

    python3 build.py --rvv-archive /path/library-rvv.a --scalar-array-archive /path/library-scalar.a --output /new/output/directory

Inputs must already contain a210_rvv_* and a210_scalar_* names, with scalar math dependencies renamed to newlib_scalar_*. Link the result with the matching isolated libscalar.a. This command rebuilds only the dispatch object and combined archive, not the underlying libraries. Requires an RV64 Linux GCC with RVV support and ar.

Historical result: target66 non-precision correctness 66/66; precision 42 reached the reference and 24 pending; 462 performance cases, geometric mean 1.072635. This is not all-input certification or standard-build integration.
