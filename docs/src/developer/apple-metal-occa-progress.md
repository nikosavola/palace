```@raw html
<!---
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
--->
```

# Apple Metal / OCCA superbuild wiring: progress and handoff

**Status:** experimental, off-by-default `PALACE_WITH_OCCA` superbuild option added
and verified on a Linux x86 VM with no Apple hardware. This carries the MFEM-side
work in the `occa-metal-apple-support` branch of
[`nikosavola/mfem`](https://github.com/nikosavola/mfem) into Palace's own CMake
superbuild, per the same branch/task in
[`nikosavola/palace`](https://github.com/nikosavola/palace).

**Companion to:** the MFEM-side handoff doc,
`doc/apple-metal-mlx-support-progress.md` in the `nikosavola/mfem` fork's
`occa-metal-apple-support` branch (and its own companion research doc,
`doc/apple-metal-mlx-support.md`). Read those first -- this document assumes the
MFEM-side work and vocabulary (`MFEM_USE_OCCA`, `occa-metal`, etc.).

**Audience:** the next implementation agent, running on real Apple Silicon
hardware, extending this into an actual working `occa-metal` build of Palace.

## 1. The headline finding: this buys Palace nothing today, and that's not a gap to close quickly

Before touching the build system, it's worth stating plainly what item 5 of the
task asked to check honestly: **Palace's own operators have zero OCCA/OKL kernel
coverage, and cannot reach any MFEM OCCA path at all.**

- `grep -rl "occa\|OCCA" palace/` and `find . -iname "*.okl"` (excluding vendored
  build directories under `extern/`) both return nothing. There is no `.okl`
  source and no `occa`/`OCCA` reference anywhere in Palace's own code.
- More fundamentally: Palace does not use MFEM's `mfem::BilinearForm` (the class
  whose partial-assembly dispatch prefers OCCA -- see
  `fem/integ/bilininteg_diffusion_pa.cpp` on the MFEM side) for its own
  electromagnetics operators (curl-curl, mass, etc.) at all. Palace has its own
  `palace::BilinearForm` (`palace/fem/bilinearform.hpp`), whose header comment
  states plainly: *"This class implements bilinear and mixed bilinear forms based
  on integrators assembled using the libCEED library."* It includes
  `fem/libceed/operator.hpp` and builds `ceed::Operator`s, not MFEM PA operators.

**Consequence:** enabling `MFEM_USE_OCCA=YES` in the MFEM subbuild -- everything
this commit wires up -- changes nothing about how Palace actually assembles or
solves anything, even with a fully working `occa-metal` backend on real Apple
hardware. It only makes an OCCA-enabled MFEM available to link against. Getting
any actual acceleration benefit for Palace would require either:

1. Palace switching some or all of its operators from libCEED to MFEM's
   OCCA-dispatching `BilinearForm` for the Apple/Metal case specifically (a
   significant architectural change, and one that would need its own
   feasibility case against libCEED's existing GPU story), or
2. libCEED itself growing an Apple Metal backend, which is a separate,
   independent, and larger body of work with no relationship to MFEM's OCCA
   integration at all.

Do not oversell what this commit does. It is exactly what WP2 (build-system
skeleton) asks for -- making the MFEM-side experiment buildable and pointable
from Palace -- and nothing more.

**Update:** the two routes above (Palace switching to `mfem::BilinearForm`, or
libCEED growing a Metal backend) turned out not to be a clean either/or.
libCEED already ships an OCCA backend of its own (`backends/occa`,
`ceed-occa`) -- Palace's superbuild simply never built it. Section 7 below
wires that up and reports what actually works and what doesn't; section 8
covers Metal specifically. Short version: libCEED-via-OCCA is real but far
more limited than it looks, and Metal has two independent blockers, not one.
Read on rather than re-deriving this from the code -- it took a full pass
through libCEED's own backend source and test harness to pin down.

## 2. What was implemented, file by file

- **`CMakeLists.txt`**: new `PALACE_WITH_OCCA` cache option (`OFF` by default),
  next to `PALACE_WITH_CUDA`/`PALACE_WITH_HIP`, with a docstring stating the
  libCEED-not-OCCA finding above up front so nobody has to rediscover it by
  reading code.
- **`cmake/ExternalGitTags.cmake`**: `EXTERN_OCCA_URL`/`EXTERN_OCCA_GIT_BRANCH`/
  `EXTERN_OCCA_GIT_TAG` entries, pinned at `v2.0.0` -- the same version the
  MFEM-side research doc and its Metal skeleton work targeted. Placed
  alphabetically between MUMPS and ParMETIS, matching this file's existing
  ordering.
- **`cmake/ExternalOCCA.cmake`** (new file, mirrors `cmake/ExternalUmpire.cmake`'s
  shape -- a CMake-based `ExternalProject_Add`, since OCCA 2.0.0 ships its own
  `CMakeLists.txt`, unlike GSLIB's Makefile-based build): builds OCCA with
  `OCCA_ENABLE_TESTS=OFF`/`OCCA_ENABLE_EXAMPLES=OFF` and otherwise its own
  defaults. Deliberately does **not** force `OCCA_ENABLE_METAL=ON` or any other
  backend flag -- OCCA's own `CMakeLists.txt` already auto-detects each backend
  independently (`OCCA_ENABLE_METAL AND APPLE`, `OCCA_ENABLE_OPENMP` if OpenMP is
  found, etc.), so there is nothing platform-specific to add here; on macOS with
  Xcode available this already builds Metal support without any Palace-side
  change.
- **`extern/CMakeLists.txt`**: `include(ExternalOCCA)` guarded by
  `if(PALACE_WITH_OCCA)`, placed immediately before the existing `include
  (ExternalMFEM)` (OCCA must be built and installed before MFEM's own configure
  step runs, since MFEM's `FindOCCA.cmake` module looks for it via
  `OCCA_DIR`).
- **`cmake/ExternalMFEM.cmake`**: two additions, both gated on
  `PALACE_WITH_OCCA` and mirroring the existing GSLIB pattern in the same file
  (`if(PALACE_WITH_GSLIB) ... GSLIB_DIR ... endif()`) line for line:
  1. `occa` appended to `MFEM_DEPENDENCIES` (only meaningful when
     `PALACE_BUILD_EXTERNAL_DEPS` is on, i.e. Palace is building OCCA itself, not
     pointed at a pre-built one) so `ExternalProject_Add`'s dependency ordering
     builds OCCA before MFEM.
  2. `-DMFEM_USE_OCCA=YES` and `-DOCCA_DIR=${CMAKE_INSTALL_PREFIX}` (or
     `-DOCCA_DIR=${OCCA_DIR}` when `PALACE_BUILD_EXTERNAL_DEPS=OFF`, i.e. the
     user supplies their own OCCA and passes `-DOCCA_DIR=...` on the command
     line -- exactly how `GSLIB_DIR` already works in that branch) appended to
     `MFEM_OPTIONS`. When `PALACE_WITH_OCCA` is off, `-DMFEM_USE_OCCA=NO` is
     still explicitly set (this was previously left unset, silently taking
     whatever MFEM's own CMake default is -- explicitly setting it to the same
     value that default already resolves to is a behavior-preserving
     documentation improvement, matching how this same file already explicitly
     sets `-DMFEM_USE_CUDA=NO`/`-DMFEM_USE_HIP=NO` in their own off branches).
- **`docs/src/developer/apple-metal-occa-progress.md`** (this file) and
  **`docs/make.jl`**: added to the "For Developers" navigation section, next to
  `developer/notes.md` etc.

## 3. Does not silently repoint `EXTERN_MFEM_URL`/`EXTERN_MFEM_GIT_TAG`

Per the task's explicit guardrail: this branch does **not** touch
`EXTERN_MFEM_URL`/`EXTERN_MFEM_GIT_TAG` in `cmake/ExternalGitTags.cmake`. They
remain pointed at upstream `mfem/mfem` at the pinned commit
`d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`.

**This has a real, load-bearing consequence you need to know before you build
with `-DPALACE_WITH_OCCA=ON`:** that pinned upstream MFEM commit has the *same*
OCCA-2.0.0 compatibility breaks the MFEM-side handoff doc documents (item 0
there) -- it predates the fixes on the `occa-metal-apple-support` MFEM branch.
**This was independently verified, not just predicted** -- see section 4's MFEM
subproject build, which targets the default (upstream, unfixed) MFEM pin on
purpose and hits exactly the ambiguous-`setup()`/`loadKernels` compile errors
the MFEM-side doc's item 0 fixes. Treat build failure as the expected outcome,
not a surprise, if you enable `PALACE_WITH_OCCA` against the *default* MFEM
pin without also overriding `EXTERN_MFEM_URL`/`EXTERN_MFEM_GIT_TAG`.

### 3.1 Do the Palace-side MFEM patches even apply to the Phase 1 MFEM branch?

Palace's own superbuild applies four local patches to whatever MFEM source it
checks out, via `PATCH_COMMAND git reset --hard && git clean -fd && git apply
"${MFEM_PATCH_FILES}"` in `cmake/ExternalMFEM.cmake`
(`extern/patch/mfem/*.diff`). Since section 5 tells the next agent to point
`EXTERN_MFEM_GIT_TAG` at the Phase 1 `occa-metal-apple-support` MFEM branch
instead of the default pin, it matters whether these four patches still apply
there -- and this was checked directly, not assumed:

```sh
cd ~/dev/mfem   # on occa-metal-apple-support
git apply --check ~/dev/palace/extern/patch/mfem/*.diff
```

Result, per-patch (`git apply --check`, plus `--check --reverse` to
distinguish "already applied upstream" from "genuinely conflicts"):

| Patch | Forward apply | Reverse apply | Verdict |
|---|---|---|---|
| `mfem_pr5246.diff` | fails (`fem/intrules.cpp`/`.hpp`) | **succeeds** | Already applied upstream of the Phase 1 branch -- its content (positive-weight simplex quadrature rules) is already present. No-op needed; not a real conflict. |
| `mfem_pr5353.diff` | **succeeds** | -- | Applies cleanly as-is. |
| `patch_gmsh_parser_performance.diff` | fails (`mesh/mesh_readers.cpp:1539`) | fails | **Genuine conflict.** `mesh/mesh_readers.cpp` has diverged around this region on the Phase 1 branch (the `occa-metal-apple-support` MFEM branch is ~2500 commits ahead of the patch set's target on `fem/`/`mesh/` files alone); the patch's context lines no longer match. Needs manual reconciliation (or dropping, if the perf fix it encodes -- `map`&rarr;`unordered_map` for the vertex lookup, hoisting `gmsh_dim` out of the read loop -- isn't needed for this experiment) before pointing `EXTERN_MFEM_GIT_TAG` at the Phase 1 branch. |
| `patch_par_tet_mesh_fix_dev.diff` | **succeeds** | -- | Applies cleanly as-is. |

**Bottom line:** 3 of 4 patches are fine as-is (one is redundant, two apply
cleanly); `patch_gmsh_parser_performance.diff` will make
`git apply "${MFEM_PATCH_FILES}"` fail as a single invocation (it applies all
patches in one `git apply` call, so one failing patch aborts the whole
`PATCH_COMMAND` and the `mfem` `ExternalProject_Add` step dies before MFEM
configures at all) unless it's fixed or removed first. This is exactly the
kind of thing the task asked to surface rather than gloss over: **verified-here
via `git apply --check`, not assumed.**

The intended integration path, exactly as the task specified, is the existing
CACHE-variable override mechanism these are already built on:

```sh
cmake -S . -B build \
  -DPALACE_WITH_OCCA=ON \
  -DEXTERN_MFEM_URL=https://github.com/nikosavola/mfem.git \
  -DEXTERN_MFEM_GIT_TAG=occa-metal-apple-support
```

(Use the specific commit SHA the Mac-side agent's work lands on, once it does,
rather than the branch name, for a reproducible pin -- `EXTERN_MFEM_GIT_TAG`
accepts either, since it's passed straight through to `ExternalProject_Add`'s
`GIT_TAG`.) This is exactly how `EXTERN_MFEM_URL`/`EXTERN_MFEM_GIT_TAG` were
already designed to be used (they are ordinary user-overridable `CACHE`
variables, like every other `EXTERN_*` entry in
`cmake/ExternalGitTags.cmake`) -- nothing new needed to be built for this, it
just needed to be documented as the path, which is what this section is.

## 4. Validation performed (Linux, no Apple hardware)

All commands from `~/dev/palace` on the `occa-metal-apple-support` branch,
against a **fresh, separate build directory** for each test (not the
pre-existing `build`/`buildB`/`buildC` directories already present in this
checkout, which predate this task and were left untouched).

### Default configure unchanged
```sh
cmake -S . -B build-default
```
Succeeds; "Configure stage complete"; no `OCCA_OPTIONS` printed (the
`if(PALACE_WITH_OCCA)` guard around `include(ExternalOCCA)` correctly skips it);
`MFEM_OPTIONS` includes `-DMFEM_USE_OCCA=NO` (previously: no `MFEM_USE_OCCA`
entry at all, relying on MFEM's own CMake default, which is also `OFF` -- so
this is not a behavior change, only an explicit one). **verified-here.**

### `PALACE_WITH_OCCA=ON` configure
```sh
cmake -S . -B build-occa -DPALACE_WITH_OCCA=ON
```
Succeeds; a `"====================== Configuring OCCA dependency
======================"` section appears with `OCCA_OPTIONS` printed; critically,
`MFEM_OPTIONS` for the same run includes `-DMFEM_USE_OCCA=YES;
-DOCCA_DIR=<prefix>` -- confirmed by grepping the actual printed
`MFEM_OPTIONS` line, not assumed. **verified-here.**

### `PALACE_WITH_OCCA=ON`, MFEM subproject build (partial validation)
Per the task's explicitly allowed partial-validation path ("a successful
configure plus a build of just the MFEM subproject ... is an acceptable,
honestly-labeled partial validation"), and to keep this tractable in the time
available, the heavier default dependencies not required by MFEM's own build
were disabled for this specific check (SuperLU_DIST, SUNDIALS, GSLIB, and
SLEPc/PETSc, none of which `cmake/ExternalMFEM.cmake` requires when off, and
PETSc/SLEPc are not an MFEM dependency in any configuration):

```sh
cmake -S . -B build-mfem-occa \
  -DPALACE_WITH_OCCA=ON \
  -DPALACE_WITH_SUPERLU=OFF -DPALACE_WITH_SUNDIALS=OFF \
  -DPALACE_WITH_GSLIB=OFF -DPALACE_WITH_SLEPC=OFF
cmake --build build-mfem-occa --target mfem -j4
```
This transitively builds `metis`, `hypre`, and `occa` (via `ExternalProject_Add`
`DEPENDS`) before MFEM's own CMake configure step runs, then MFEM itself. This
run used the **default** MFEM pin (upstream `mfem/mfem` at
`d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`), deliberately -- not the Phase 1
`occa-metal-apple-support` branch -- specifically to check the section-3 claim
that the default pin hits the same OCCA-2.0.0 breaks the MFEM-side doc fixes.

**Result: OCCA was found and wired correctly; MFEM's own (unfixed, upstream)
source then failed to compile against OCCA 2.0.0, exactly as predicted.**
Exact output from MFEM's own CMake configure step (`extern/mfem-build`'s
config log), confirming `MFEM_USE_OCCA=YES` genuinely reached and was acted on
by MFEM's own `find_package`, not just passed as a string:

```
-- Looking for OCCA ...
--    in OCCA_DIR = <prefix>
-- Found OCCA: <prefix>/lib/libocca.so
-- OCCA_INCLUDE_DIRS=<prefix>/include
...
-- MFEM: using package OCCA
...
-- MFEM version: v4.9.0
-- MFEM git string: tags/v4.9-0-gd9d6526cc1749980a2ba1da16e2c1ca1e07d82ec-dirty
-- Configuring done (1.7s)
-- Generating done (0.6s)
```

Then the build step itself fails, with the exact compile errors the MFEM-side
`occa-metal-apple-support` branch's item 0 fixes (`general/device.cpp`,
`general/occa.cpp`):

```
general/device.cpp: In function 'void mfem::OccaDeviceSetup(int)':
general/device.cpp:499:33: error: call of overloaded 'setup(const char [15])' is ambiguous
  499 |       internal::occaDevice.setup("mode: 'OpenMP'");
      |       ~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~
note: candidate: 'void occa::device::setup(const std::string&)'
note: candidate: 'void occa::device::setup(const occa::json&)'
general/device.cpp:506:33: error: call of overloaded 'setup(const char [15])' is ambiguous
  506 |       internal::occaDevice.setup("mode: 'Serial'");
general/device.cpp:524:10: error: 'loadKernels' is not a member of 'occa'
  524 |    occa::loadKernels("mfem");
      |          ^~~~~~~~~~~
gmake[6]: *** [CMakeFiles/mfem.dir/build.make:97: CMakeFiles/mfem.dir/general/device.cpp.o] Error 1
gmake: *** [Makefile:293: mfem] Error 2
```

**verified-here**, both halves: (1) `PALACE_WITH_OCCA=ON` correctly builds
OCCA and threads `OCCA_DIR`/`MFEM_USE_OCCA=YES` into MFEM's own configure step
(MFEM's own log says `Found OCCA` / `using package OCCA`, not just Palace's
`MFEM_OPTIONS` list); (2) building MFEM against the *default* upstream pin
fails with precisely the ambiguous-`setup()`/`loadKernels` errors the MFEM-side
doc's item 0 exists to fix -- section 3's compatibility warning was a
prediction going in, and is now an observed result, not a guess. This is also
why section 5 tells the Mac-side agent to override `EXTERN_MFEM_GIT_TAG` to the
Phase 1 branch rather than build against the default pin.

### Not validated
- A full default `PALACE_WITH_OCCA=OFF` build (HYPRE, METIS, SuperLU_DIST,
  PETSc/SLEPc, ...) to completion -- the pre-existing `build`/`buildB`/`buildC`
  directories in this checkout suggest this has been done successfully before
  on this machine (by Niko, prior to this task; not reproduced here), but was
  not re-run end-to-end in this session, only configure-tested (see above).
- Anything on macOS, or against an OCCA built with Metal support -- no Apple
  hardware available. This is the actual WP0/WP1 starting point for the next
  agent.
- A build of the MFEM subproject against the *Phase 1* `occa-metal-apple-support`
  branch from within the Palace superbuild (i.e. with `EXTERN_MFEM_GIT_TAG`
  overridden per section 5) -- only the default upstream pin was built here
  (see section 4), to confirm the section-3 compatibility prediction. Building
  the overridden case is the Mac-side agent's first step (section 5), and
  should also apply the `extern/patch/mfem/*.diff` fix noted in section 3.1
  first.

## 5. Exact commands for the Mac-side agent to run first

```sh
# 0. First check whether extern/patch/mfem/patch_gmsh_parser_performance.diff
#    still applies to the Phase 1 branch's current HEAD (it did not against
#    the commit this doc was written against -- see section 3.1). If it still
#    fails, either fix it up (rebase the hunk onto the current
#    mesh/mesh_readers.cpp) or drop it from cmake/ExternalMFEM.cmake's
#    MFEM_PATCH_FILES for this experiment -- otherwise step 2 below will fail
#    at the PATCH_COMMAND step before MFEM even configures.
cd ~/dev/mfem && git apply --check ~/dev/palace/extern/patch/mfem/*.diff

# 1. Point this superbuild's MFEM at the Phase 1 MFEM branch (see section 3).
#    Get the exact commit SHA from the nikosavola/mfem occa-metal-apple-support
#    branch first, rather than trusting a moving branch name.
cmake -S . -B build-metal \
  -DPALACE_WITH_OCCA=ON \
  -DEXTERN_MFEM_URL=https://github.com/nikosavola/mfem.git \
  -DEXTERN_MFEM_GIT_TAG=<commit-sha-on-occa-metal-apple-support>

# 2. Build the MFEM subproject first, in isolation, before the rest of the
#    superbuild -- this is where the MFEM-side occa-metal skeleton actually
#    gets exercised for the first time in the Palace context.
cmake --build build-metal --target mfem -j

# 3. Confirm MFEM's own build actually picked up occa-metal, don't assume:
#    check build-metal/extern/mfem-build's own CMake config output/cache for
#    MFEM_USE_OCCA=YES, and (once the MFEM-side skeleton is extended to build
#    the Metal mode itself, which it does not yet -- see the MFEM-side
#    handoff doc) for any Metal-specific feature flags.

# 4. Only then build the rest: `cmake --build build-metal -j`.

# 5. Read the MFEM-side handoff doc's own "exact commands" section
#    (doc/apple-metal-mlx-support-progress.md in the MFEM checkout this
#    EXTERN_MFEM_GIT_TAG points at) for what to validate on the MFEM side
#    specifically (occa-metal actually selectable, PA kernel fingerprint
#    comparison, etc.) before assuming any of it works inside the Palace
#    superbuild context too.

# 6. If/when Palace's own operators are ever reconsidered for an OCCA/Metal
#    path (see section 1's "what this would actually require") -- that is a
#    separate, much larger piece of work, starting from palace/fem/
#    bilinearform.hpp and palace/fem/libceed/, not from this build-system
#    wiring.
```

## 6. Guardrails respected

Confirmed via `git diff -U0` showing only newly-added lines, all OCCA-specific:
no existing line touching `PALACE_WITH_CUDA`, `PALACE_WITH_HIP`,
`PALACE_WITH_GPU_AWARE_MPI`, or Hypre/solver configuration was modified. The new
option is off by default and the default configure step was verified unchanged
(section 4). Re-confirmed identically after the section 7 wiring below (a
second, independent `git diff -U0` scan and a second fresh default-configure
run -- see section 7).

## 7. libCEED's own OCCA backend: wiring, what's verified, and what isn't

Section 1 named two routes to get any OCCA execution under Palace's actual
operators: switch Palace to `mfem::BilinearForm` (large, separate,
architectural), or wait for libCEED to grow a Metal backend (someone else's
work). Both undersold the real situation. **libCEED already ships an OCCA
backend** -- `backends/occa/` in the libCEED source tree, registering as
`ceed-occa` -- gated purely on whether `OCCA_DIR` is passed to its Makefile.
Palace's superbuild simply never passed it. This section wires that up and
reports, with real commands and output, what that buys and what it doesn't.

### 7.1 What changed

- **`cmake/ExternalLibCEED.cmake`**: `OCCA_DIR=${CMAKE_INSTALL_PREFIX}` (or
  the user-supplied `OCCA_DIR` when not building deps) appended to
  `LIBCEED_OPTIONS` when `PALACE_WITH_OCCA` is on, and `occa` appended to
  `LIBCEED_DEPENDENCIES` so `ExternalProject_Add`'s dependency graph builds
  OCCA before libCEED -- both following the exact pattern already used in
  this file for `CUDA_DIR`/`ROCM_DIR`/`MAGMA_DIR`/`XSMM_DIR`. libCEED's own
  Makefile does the rest: it runs `$(OCCA_DIR)/bin/occa modes` and registers
  one libCEED backend per detected OCCA mode (`/cpu/self/occa` unconditionally
  once any OCCA is found, then `/cpu/openmp/occa`, `/gpu/cuda/occa`,
  `/gpu/hip/occa`, `/gpu/opencl/occa`, `/gpu/dpcpp/occa` if that mode shows up
  in `occa modes`) -- nothing mode-specific needs to be forced from Palace's
  side, matching how `cmake/ExternalOCCA.cmake` already doesn't force any
  backend flags on OCCA itself (section 2).
- **`extern/CMakeLists.txt`**: moved the `include(ExternalOCCA)` block to
  *before* `include(ExternalLibCEED)` (it was between libCEED and MFEM).
  This is load-bearing, not cosmetic: `ExternalProject_Add(libCEED DEPENDS
  ... occa ...)` needs the `occa` CMake target to already exist when that
  call is evaluated.

### 7.2 Verified-here: the wiring itself

```sh
cmake -S . -B build-default   # unchanged: no OCCA section, no OCCA_DIR in
                               # LIBCEED_OPTIONS or MFEM_OPTIONS -- re-run
                               # fresh after this section's changes, not just
                               # assumed still true from section 4.
cmake -S . -B build-libceed-occa -DPALACE_WITH_OCCA=ON \
  -DPALACE_WITH_SUPERLU=OFF -DPALACE_WITH_SUNDIALS=OFF \
  -DPALACE_WITH_GSLIB=OFF -DPALACE_WITH_SLEPC=OFF
cmake --build build-libceed-occa --target libCEED -j4
```

Configure output shows `LIBCEED_OPTIONS` gained
`OCCA_DIR=<prefix>` (grepped from the printed line, not assumed). The build
succeeds: `backends/occa/*.cpp` compile (with libCEED's own upstream
`#warning "libCEED OCCA backend is experimental; for best performance, use
device native backends"`, worth keeping in mind for everything below) and
link into `libceed.so`.

Confirmed the backend is actually registered, not just compiled in --
libCEED's own `make info-backends` reports it:

```
$ make info-backends OCCA_DIR=<prefix>
make: 'lib' with optional backends: /cpu/self/avx/serial /cpu/self/avx/blocked /cpu/self/occa /cpu/openmp/occa
```

And confirmed it actually initializes at runtime, with a small standalone
program calling `CeedInit()` and `CeedGetResource()` (not just checking for a
non-error return -- confirming the resource string that comes back matches
what was requested):

```
OK: requested /cpu/self/ref/serial     -> actual resource: /cpu/self/ref/serial
OK: requested /cpu/self/occa           -> actual resource: /cpu/self/occa
OK: requested /cpu/openmp/occa         -> actual resource: /cpu/openmp/occa
```

(No CUDA/HIP/Metal on this VM -- `occa modes` here reports only `Serial` and
`OpenMP`, so only those two `ceed-occa` resources register. This part of the
mechanism is platform-generic, though: on a machine with CUDA or HIP, the
same wiring would additionally register `/gpu/cuda/occa`/`/gpu/hip/occa`.)

### 7.3 Verified-here: what actually computes correctly on `ceed-occa` -- basis/vector primitives

libCEED's own test suite (`tests/t318-basis.c`, unmodified, upstream) does a
real numeric check: tensor H1 Lagrange interpolation against an analytic
function, printing a mismatch line only on failure. Compiled once, run against
three resources:

```
$ ./t318 /cpu/self/ref/serial   # baseline
$ ./t318 /cpu/self/occa
$ ./t318 /cpu/openmp/occa
```

All three exit 0 with zero mismatch output. **This is real, verified-here
correctness evidence for `ceed-occa`'s basis-apply (`CeedBasisApply`,
`CEED_EVAL_INTERP`) kernels on both Serial and OpenMP OCCA modes.**

### 7.4 Not what it looks like: `CeedOperatorApply` with a user QFunction does not work on `ceed-occa`, and this is upstream-acknowledged, not just untested

This is the finding that matters most for Palace specifically, because it is
exactly the code path Palace's own operators use.

**Reproduced directly:** compiling and running libCEED's own
`tests/t500-operator.c` (creation, action, and destruction of a mass-matrix
operator -- i.e. exactly a `CeedOperatorApplyAdd` with a user-supplied
QFunction, the architecture Palace's curl-curl/mass assembly is built on)
against `/cpu/self/occa` aborts:

```
.../t500-operator.h:8:2: Error: File does not exist
#include <ceed/types.h>
terminate called after throwing an instance of 'occa::exception'
  what():
---[ Error ]--------------------------------------------------------------------
    Message  : Unable to transform OKL kernel [...]
    Stack
       9 ceed::occa::CpuOperator::buildApplyAddKernel()
       8 ceed::occa::Operator::applyAdd(...)
       7 .../lib/libceed.so(CeedOperatorApplyAdd+0x9c)
...
Aborted (core dumped)
```

Same crash against `/cpu/openmp/occa`.

**This is not a fluke of an ad-hoc test harness -- it's upstream's own
documented position.** libCEED's own test runner,
`tests/junit.py`, has an explicit pre-run skip rule:

```python
if contains_any(resource, ['occa']) and startswith_any(
        test, ['t4', 't5', 'ex', 'mfem', 'nek', 'petsc', 'fluids', 'solids']):
    return 'OCCA mode not supported'
```

`t4*` is libCEED's QFunction-level tests, `t5*` its Operator-level tests, and
`ex`/`mfem`/`nek`/`petsc`/`fluids`/`solids` are its real downstream-application
examples -- i.e. **every test category that exercises a full
QFunction+Operator pipeline is explicitly excluded from OCCA in libCEED's own
CI.** Only the vector/basis/restriction-level categories (`t1*`-`t3*`, section
7.3's territory) run against `ceed-occa` upstream. This independently confirms
what the crash reproduces: the Operator+QFunction path is not a supported
configuration of `ceed-occa` today, not something this session broke or
failed to configure correctly.

**Plausible mechanism (labeled as hypothesis, not a fix):**
`backends/occa/ceed-occa-qfunction.cpp`'s `QFunction::getKernelProps()`
builds the `occa::properties` used to compile a user QFunction's kernel. It
sets `defines/CeedInt`, `defines/CeedScalar`, `defines/CeedPragmaSIMD`, and
injects `#include "<absolute path to the qfunction header>"` as literal
kernel source -- but never adds an include *search path* (OCCA's own
mechanism for this is `props["okl/include_paths"]` or the `OCCA_INCLUDE_PATH`
environment variable, per
`~/dev/occa-ref/src/occa/internal/lang/preprocessor.cpp`). A `grep -rn
"includes\|compiler_flags\|-I" backends/occa/*.cpp` across the whole backend
turns up nothing. So when a QFunction header pulls in a real system-style
header via `#include <...>` -- `tests/t500-operator.h` does exactly this for
`<ceed/types.h>` -- OCCA's own kernel-source parser has nowhere configured to
look for it. This is offered as the likely *mechanism*, not a verified fix:
ad hoc attempts in this session to work around it (setting
`OCCA_INCLUDE_PATH`, registering an extra `CeedAddJitSourceRoot`) gave
inconsistent results across repeated runs, almost certainly due to OCCA's own
kernel-hash build cache (`~/.occa/cache/`) interacting badly with a
hand-rolled test harness outside libCEED's real test driver. Given the
`tests/junit.py` skip list already says plainly that this category isn't
supported, chasing a source patch for an upstream-acknowledged gap was not a
good use of further time in this session -- see "Not attempted" below.

**Does this transfer to Palace's own QFunctions specifically?** Checked
directly rather than assumed: `grep -rn "#include <" palace/fem/qfunctions/`
shows Palace's QFunction headers include exactly one angle-bracket system
header, `<math.h>` (in the `utils_*_qf.h` helpers) -- not `<ceed/types.h>` or
`<ceed.h>` directly, unlike libCEED's own `t500-operator.h`. `<math.h>` is a
standard header OCCA's underlying toolchain resolves by default, so Palace's
specific QFunctions likely would not reproduce this *exact* symptom
verbatim. That doesn't rescue the OCCA-Operator path, though: `tests/junit.py`'s
skip rule is architectural (any Operator+QFunction usage), not
header-content-specific, and Palace's `BilinearForm`/`operator.cpp` is
precisely that architecture (see 7.5). The honest statement is "the specific
crash reproduced here may not be bit-for-bit what Palace would hit," not
"Palace would be fine."

### 7.5 What this means for Palace, concretely

Palace's `palace::BilinearForm` (`palace/fem/bilinearform.hpp`,
`palace/fem/libceed/operator.cpp`) calls `CeedOperatorApplyAdd` with custom
QFunctions for its actual electromagnetics assembly
(`palace/fem/libceed/operator.cpp:170-176`, e.g. `CeedVectorSetArray(u[id],
mem, CEED_USE_POINTER, ...)` then `CeedOperatorApplyAdd(op[id], u[id], v[id],
...)`) -- exactly the category `tests/junit.py` marks unsupported for OCCA and
that crashed in 7.4's reproduction. Combined with section 1's finding that
Palace never goes through `mfem::BilinearForm` at all, there are now two
independent routes for OCCA to reach Palace's real assembly, and **both are
closed today**:

1. Via `mfem::BilinearForm` (MFEM's own OCCA-dispatching partial assembly) --
   closed because Palace doesn't use that class (section 1).
2. Via `ceed-occa` (libCEED's own OCCA backend, wired up in this section) --
   closed because the Operator+QFunction path it would need is not a
   supported configuration of that backend (this section).

So: this section's build wiring is real (7.2), and the basis/vector-level
primitives genuinely compute correctly through it (7.3) -- but none of that
is on any code path Palace's own operators actually call. **The correct,
undiluted statement of where things stand is that Palace still gains zero
functional benefit from any of the OCCA wiring in this document, including
this section's libCEED work.** It is one increment closer to the truth than
section 1's original framing, which left the door open to "libCEED growing a
Metal backend" as if that were the only piece missing; this section shows the
CPU/OpenMP door on libCEED's *existing* OCCA backend is already closed for a
different, more immediate reason.

### 7.6 Not attempted, and why

- **A source patch to `ceed-occa` to add an OCCA include-search-path
  property.** Even if this fixed the exact `<ceed/types.h>` symptom (not
  confirmed -- see 7.4), `tests/junit.py`'s skip list is the stronger signal
  that Operator+QFunction support on OCCA is an upstream-acknowledged gap, not
  a one-line oversight. A real fix belongs upstream, would need libCEED's
  actual maintainers or a proper fork+patch workflow (Palace has no such
  mechanism for libCEED today -- unlike MFEM's `extern/patch/mfem/*.diff`,
  there is no `LIBCEED_PATCH_FILES` in `cmake/ExternalLibCEED.cmake`), and
  should be scoped and tested against libCEED's own CI expectations, not
  guessed at from one VM.
- **CUDA/HIP `ceed-occa` resources.** No CUDA/HIP toolchain on this VM;
  `occa modes` here only ever reports `Serial`/`OpenMP`.

## 8. Metal specifically: two independent blockers, not one

The task framing going in was "get to OCCA, then get to Metal, patching MFEM
as needed." Section 7 already shows the MFEM side isn't where the remaining
work is -- MFEM's own `occa-metal` skeleton (the companion MFEM-side doc)
rejects cleanly on non-Apple platforms exactly as designed, and section 7's
blocker sits entirely inside libCEED. This section adds a second,
independent libCEED-side blocker specific to Metal, and states plainly: no
patch to MFEM gets Palace to a working Metal backend. Both real blockers are
in libCEED.

### 8.1 Blocker 1 (inherited from section 7): the Operator path doesn't work on any `ceed-occa` mode, Metal included

Section 7.4's finding is mode-agnostic -- the missing include-path mechanism
and the `tests/junit.py` skip rule apply to `ceed-occa` as a whole, not
specifically to Serial/OpenMP. There is no reason to expect a hypothetical
`/gpu/metal/occa` resource would behave differently; if anything, GPU modes
in `ceed-occa` are described in the source as *more* experimental than CPU
modes (see 8.3). Getting a real electromagnetics operator to run via
`ceed-occa` on any mode -- CPU, OpenMP, or an eventual Metal -- requires
fixing this first, and it is an upstream libCEED problem, not something
Palace's build wiring or an MFEM patch can route around.

### 8.2 Blocker 2: Metal was never wired into `ceed-occa`'s resource parser, and there's a real reason why

Checked directly: `grep -rniE "metal" backends/occa/` across libCEED's entire
OCCA backend returns exactly one hit,
`ceed-occa.cpp:52`, a bare comment inside `getDefaultDeviceMode()`'s GPU
priority list:

```cpp
if (gpuMode) {
  if (::occa::modeIsEnabled("CUDA")) { return "CUDA"; }
  if (::occa::modeIsEnabled("HIP")) { return "HIP"; }
  if (::occa::modeIsEnabled("dpcpp")) { return "dpcpp"; }
  if (::occa::modeIsEnabled("OpenCL")) { return "OpenCL"; }
  // Metal doesn't support doubles
}
```

`getDeviceMode()` (the function that parses a resource string like
`/gpu/cuda/occa` into an OCCA mode name) has no `"metal"` case at all --
Metal isn't gated off, it was simply never added. The comment says why:
**`CeedScalar` is hardcoded to `double`, and Metal's double-precision support
is poor-to-absent on Apple GPUs.** Verified directly, not inferred from the
comment alone: `include/ceed/types.h:153` does

```c
/// Base scalar type for the library to use: change which header is included to change the precision.
#include "ceed-f64.h"
```

unconditionally -- there is no build-time flag in this libCEED pin that
selects `ceed-f32.h` (which exists, and is installed alongside `ceed-f64.h`,
but is never `#include`d by anything). Changing `CeedScalar` today means
manually editing this line and rebuilding all of libCEED, not passing an
option.

**This is not a libCEED-internal-only concern -- it is load-bearing for
Palace's actual data flow.** Checked directly in
`palace/fem/libceed/operator.cpp:160-176`:

```cpp
const auto *x_data = x.Read(mem == CEED_MEM_DEVICE);
auto *y_data = y.ReadWrite(mem == CEED_MEM_DEVICE);
...
CeedVectorSetArray(u[id], mem, CEED_USE_POINTER, const_cast<CeedScalar *>(x_data));
CeedVectorSetArray(v[id], mem, CEED_USE_POINTER, y_data);
```

`x_data`/`y_data` are `mfem::real_t*` (from an `mfem::Vector`), handed to
`CeedVectorSetArray` with `CEED_USE_POINTER` -- a zero-copy aliasing call,
not a conversion. **`CeedScalar` and `mfem::real_t` must be the exact same
type, bit for bit, or this is silent memory misinterpretation, not just a
slow path.** Palace has zero `MFEM_USE_SINGLE` references anywhere
(`grep -rn "MFEM_USE_SINGLE" palace/` -- empty), meaning MFEM is built double
throughout every Palace configuration today, matching libCEED's hardcoded
`CeedScalar = double`.

**Consequence:** there is no scoped patch that gets Palace to a working Metal
path here. The two coherent options, both far bigger than "patch MFEM":

1. **Flip precision everywhere, together.** Rebuild libCEED with
   `CeedScalar = float` (edit `ceed-f64.h`->`ceed-f32.h` in `types.h`, or add
   a real build-time switch -- itself a libCEED patch, not an MFEM one) *and*
   rebuild MFEM with `MFEM_USE_SINGLE` (already plumbed on the MFEM side --
   see the companion MFEM-side doc's `OccaSetRealTypeDefine` work) *and*
   verify HYPRE still gets the double precision it needs for the actual
   linear solve (Palace's own assembly output would be single, but the
   solver stack downstream may not tolerate that without its own changes --
   not investigated here, out of scope for a build-wiring milestone). This
   is a whole-stack precision decision, not a Metal-specific toggle, and it
   would need sign-off given how far it reaches.
2. **Add a Metal-specific single-precision path inside `ceed-occa` only**,
   keeping the rest of libCEED double -- architecturally cleaner but a real
   feature addition to `ceed-occa` (a new resource-string case, a
   precision-aware kernel-property path, and a defined boundary where
   double-precision host data gets cast to float before crossing into the
   Metal-mode `CeedVector`). This is upstream libCEED feature work, not
   something Palace's superbuild or an MFEM patch can supply.

### 8.3 One more upstream signal worth carrying forward

`ceed-occa.cpp`'s file-level `#warning "libCEED OCCA backend is experimental;
for best performance, use device native backends"` and the GPU-priority
ordering in `getDefaultDeviceMode()` (CUDA, then HIP, then dpcpp, then
OpenCL -- native per-vendor backends are what libCEED actually recommends)
both point the same way: `ceed-occa` reads as a secondary, less-maintained
backend relative to libCEED's native CUDA/HIP backends, even before Metal
enters the picture. Worth knowing going in, not just for the Metal case.

### 8.4 Bottom line for the next agent

**Don't patch MFEM to chase Metal.** Nothing found in this session points at
MFEM. Both real blockers are in libCEED: the Operator+QFunction path doesn't
work on `ceed-occa` at all today (section 7.4), and `CeedScalar` is
hardcoded double with no Metal registration in the resource parser (section
8.2). If Metal-via-libCEED is still the goal, the actual next steps are
upstream libCEED work -- fix or scope the Operator-path gap, then design a
real precision story -- not anything reachable from Palace's or MFEM's build
systems alone.
