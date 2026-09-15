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
[`nikosavola/palace`](https://github.com/nikosavola/palace). A second pass
additionally wires and patches libCEED's own OCCA backend (`ceed-occa`) --
see section 7 for what now genuinely works (`CeedOperatorApply` and
Jacobi/Chebyshev diagonal assembly, verified with real numerics on
`/cpu/self/occa` and `/cpu/openmp/occa`) and what still doesn't (full
sparse-matrix assembly, needed by Palace's default AMS/AMG solves, segfaults
via a separate libCEED bug -- section 7.6). Metal itself remains
untouched/unreachable (section 8) -- no Apple hardware exists to test it,
and it has its own additional blocker beyond what section 7 fixes.

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

**Update (second pass):** the two routes above (Palace switching to
`mfem::BilinearForm`, or libCEED growing a Metal backend) turned out not to be
a clean either/or. libCEED already ships an OCCA backend of its own
(`backends/occa`, `ceed-occa`) -- Palace's superbuild simply never built it.
Section 7 wires that up, and after a second, deeper pass (including finding
and fixing a real bug in `ceed-occa`, and finding and fixing a bug in this
document's own earlier testing), the accurate summary is: **`CeedOperatorApply`
and Palace's Jacobi/Chebyshev-smoother diagonal assembly now work correctly on
`ceed-occa`, verified with real numerics -- but Palace's default AMS/AMG solve
path still cannot complete end to end**, because full sparse-matrix assembly
(needed by every AMS/AMG-preconditioned solve) hits a separate, deeper bug.
Section 8 covers Metal specifically, which has its own, independent blocker on
top of all of this. Read sections 7 and 8 in full -- an earlier draft of this
document got the diagnosis in section 7.4 wrong (attributed to a missing
`ceed-occa` include path; it was actually a bug in this document's own test
harness), and that correction matters for anyone skimming just the headlines.

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

# 6. Before touching Metal at all: try a real Palace run against
#    /cpu/self/occa or /cpu/openmp/occa on real hardware, via the config
#    file's "Backend" field under "Solver" (palace/utils/configfile.cpp,
#    ceed_backend -- no Palace source change needed, ConfigureCeedBackend()
#    in palace/main.cpp already passes this straight through to CeedInit()).
#    Expect: any solve that avoids ParallelAssemble() (no AMS/AMG coarse
#    level) has a real chance of working, per section 7's fixes. Any
#    AMS/AMG-preconditioned solve (Palace's default) will very likely
#    segfault at CeedOperatorLinearAssembleSymbolic -- see section 7.6 for
#    the exact backtrace to expect and its root cause, before assuming this
#    session's fix broke on real hardware.

# 7. Only once 6 is understood on real hardware: revisit section 8 for
#    Metal specifically. Fixing section 7.6's libCEED bug (or working around
#    it Palace-side by avoiding ParallelAssemble() for ceed-occa) comes
#    before the Metal registration/precision work in section 8.2 -- Metal
#    inherits the same full-assembly blocker as CPU/OpenMP, so there is no
#    point reaching for Metal-specific work until 7.6 is resolved one way
#    or another.

# 8. If/when Palace's own operators are ever reconsidered for an OCCA/Metal
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

### 7.4 Correction: the earlier "Operator+QFunction doesn't work" diagnosis was wrong -- it was this document's own test-harness bug

An earlier pass of this document (still visible in git history) claimed
`CeedOperatorApply` with a user QFunction categorically does not work on
`ceed-occa`, citing a crash on libCEED's own `tests/t500-operator.c` and
`tests/junit.py`'s `'OCCA mode not supported'` skip rule for `t4*`/`t5*`/
`ex`/`mfem`/`nek`/`petsc`/`fluids`/`solids`. **That diagnosis was incorrect.**
The crash was real, but its cause was a bug in this document's own test
harness, not in `ceed-occa`:

- `CeedAddJitSourceRoot(ceed, root)` requires `root` to end with a trailing
  slash. `interface/ceed-jit-tools.c`'s `CeedPathConcatenate()` finds the
  *last* `/` in the given root and keeps only up to and including it --
  passing `.../extern/libCEED` (no trailing slash) silently truncates to
  `.../extern/` before concatenating the relative path, so a real,
  existing file is reported as "not found". Confirmed directly with a small
  diagnostic calling `CeedGetJitSourceRoots`/`CeedGetJitAbsolutePath`
  directly: with the root passed exactly as `nikosavola/palace`'s own
  `palace/CMakeLists.txt` already does it
  (`PALACE_LIBCEED_JIT_SOURCE_DIR=".../include/palace/"`, trailing slash
  present), resolution succeeds; without it, it fails with the exact
  "Couldn't find matching JiT source file" error this document originally
  attributed to `ceed-occa`. **Palace's own code already gets this right** --
  the bug was only in this document's ad hoc reproduction, never in Palace.
- Separately, libCEED's own `t500-operator.h`/`t400-qfunction.h` (and the
  installed `include/ceed/jit-source/gallery/*.h` headers) explicitly
  `#include <ceed/types.h>`. `ceed-occa-qfunction.cpp`'s
  `QFunction::getKernelProps()` already predefines `CeedInt`/`CeedScalar`/
  `CEED_QFUNCTION` via `occa::properties` before including the qfunction
  source, making that `#include` redundant for the OCCA path -- and when
  OCCA's own kernel-source parser follows it (resolvable via the
  `OCCA_INCLUDE_PATH` environment variable / `props["okl/include_paths"]`,
  per `~/dev/occa-ref/src/occa/internal/lang/preprocessor.cpp`), it fails on
  *that* header's own further nested quoted `#include "ceed-f64.h"` --
  an OCCA parser limitation specific to files that redundantly re-include
  `<ceed/types.h>`. **Palace's own QFunction headers do not do this**:
  `grep -rn "#include <" palace/fem/qfunctions/` shows exactly one
  angle-bracket include anywhere in that tree, `<math.h>` (in the
  `utils_*_qf.h` helpers) -- never `<ceed/types.h>` or `<ceed.h>`. This
  category of failure is specific to libCEED's own test/gallery headers, not
  a real constraint on Palace's QFunctions.

**Corrected, verified-here result:** building a QFunction in Palace's own
style (no direct `<ceed/types.h>` include, real `<math.h>` usage via
`sqrt`/`tanh`, matching `palace/fem/qfunctions/utils_*_qf.h`) into a real
mass-matrix `CeedOperatorApplyAdd` (adapted from libCEED's own
`tests/t500-operator.c`) and running it, with a fresh, hermetic
`OCCA_CACHE_DIR` per run (ruling out the OCCA kernel-cache cross-contamination
that made earlier ad hoc attempts in this session non-reproducible):

```
$ ./t500_palace_style /cpu/self/ref/serial   # exit 0, no mismatch
$ ./t500_palace_style /cpu/self/occa         # exit 0, no mismatch
$ ./t500_palace_style /cpu/openmp/occa       # exit 0, no mismatch
```

Reproduced identically across 6 independent runs (3 per OCCA mode) with a
fresh cache directory each time. **`CeedOperatorApply`/`CeedOperatorApplyAdd`
with a Palace-style QFunction genuinely works correctly on `/cpu/self/occa`
and `/cpu/openmp/occa`.** `tests/junit.py`'s skip rule still stands as an
accurate description of libCEED's own *test suite* (many of those tests --
`t4*`, `ex`, `mfem`, `petsc`, etc. -- exercise other things this document
does not claim work, and some genuinely don't, see 7.6), but it does not mean
"Operator+QFunction is broken on `ceed-occa`" as a blanket statement, and this
document's earlier reading of it that way was wrong.

### 7.5 A real fix: `CeedOperatorLinearAssembleAddDiagonal` (Jacobi/Chebyshev smoothing) now works via libCEED's own fallback mechanism

Palace's `Operator::AssembleDiagonal()` (`palace/fem/libceed/operator.cpp`)
calls `CeedOperatorLinearAssembleAddDiagonal`, used by
`palace/linalg/jacobi.cpp` and `palace/linalg/chebyshev.cpp` -- i.e. by
Palace's Jacobi and Chebyshev smoothers, standard components of its default
multigrid preconditioning. Before this session's fix, this aborted
deterministically and unconditionally on `ceed-occa`:

```
backends/occa/ceed-occa-ceed-object.cpp:29 in staticCeedError():
(OCCA) Backend does not implement LinearAssembleDiagonal
```

**Root cause, found by reading `interface/ceed-preconditioning.c`'s
dispatcher, not guessed:**

```c
if (op->LinearAssembleAddDiagonal) {
  CeedCall(op->LinearAssembleAddDiagonal(op, assembled, request));  // backend version
} else if (is_composite) { ... }
else {
  CeedOperator op_fallback;
  CeedCall(CeedOperatorGetFallback(op, &op_fallback));
  if (op_fallback) { CeedCall(CeedOperatorLinearAssembleAddDiagonal(op_fallback, ...)); return ...; }
}
// Default interface implementation
CeedCall(CeedOperatorLinearAssembleAddDiagonalSingle(op, request, false, assembled));
```

libCEED has a real, designed-for-this fallback mechanism -- `CeedOperator`s
can delegate operations they don't implement to a fallback `Ceed` context,
exactly what `backends/cuda-gen`/`backends/hip-gen` already do
(`CeedSetOperatorFallbackCeed(ceed, ceed_ref)` in
`backends/cuda-gen/ceed-cuda-gen.c`) for their own unimplemented operations.
But this dispatcher only reaches the fallback branch when
`op->LinearAssembleAddDiagonal` is null. `ceed-occa`'s
`Operator::ceedCreate()` registered a **non-null** function pointer that
always calls `staticCeedError(...)` -- so the top branch always fired, and
the fallback (which `ceed-occa` never registered anyway) was never reachable.

**The fix** (`extern/patch/libceed/patch_occa_operator_fallback.diff`,
applied via a new `PATCH_COMMAND` in `cmake/ExternalLibCEED.cmake`, gated on
`PALACE_WITH_OCCA`, mirroring `ExternalMFEM.cmake`'s mechanism): two small
changes to `backends/occa/ceed-occa-operator.cpp` and
`backends/occa/ceed-occa.cpp`:

1. Stop registering `LinearAssembleQFunction`/`LinearAssembleQFunctionUpdate`/
   `LinearAssembleAddDiagonal`/`LinearAssembleAddPointBlockDiagonal`/
   `CreateFDMElementInverse` as hard-failing stubs (leave the pointers null).
2. In `initCeed()`, register a reference-backend fallback Ceed
   (`/cpu/self/ref/serial`, or `/gpu/cuda/ref`/`/gpu/hip/ref` for those
   modes) via `CeedSetOperatorFallbackCeed`, the same call cuda-gen/hip-gen
   already use.

**Verified-here**, via the *actual* `ExternalProject_Add`/`PATCH_COMMAND`
path (not a hand-edited build tree -- the patch was generated against a
fresh clone at the pinned `95bd1e908b...` commit, confirmed with
`git apply --check` against that clean checkout, then the whole
`libCEED` target was deleted and rebuilt from scratch through Palace's
CMake so the verification below exercises the same mechanism the Mac-side
agent will use):

```
$ ./t500_palace_style /cpu/self/ref/serial
CeedOperatorLinearAssembleAddDiagonal ierr=0
diag: sum=2.040000000000 sumsq=0.082059088502 first=0.003395390927 last=0.010204609073 n=61

$ ./t500_palace_style /cpu/self/occa
CeedOperatorLinearAssembleAddDiagonal ierr=0
diag: sum=2.040000000000 sumsq=0.082059088502 first=0.003395390927 last=0.010204609073 n=61

$ ./t500_palace_style /cpu/openmp/occa
CeedOperatorLinearAssembleAddDiagonal ierr=0
diag: sum=2.040000000000 sumsq=0.082059088502 first=0.003395390927 last=0.010204609073 n=61
```

Bit-identical across all three. The qfunction's `rho` coefficient was
deliberately made non-trivial and varying per quadrature point
(`rho[i] = weight[i] * dxdX[i] * (1.0 + 0.37 * i)`, so `first != last`) before
running this comparison -- a uniform/trivial coefficient could make a wrong,
degenerate implementation (e.g. one that accidentally aliases host pointers
across backends) look right by coincidence. It didn't happen here: the
diagonal is genuinely non-uniform and still matches exactly.

### 7.6 Not fixed: full sparse-matrix assembly still segfaults, and Palace's default AMS/AMG solve path needs exactly that

Palace's `CeedOperatorAssembleCOO` (`palace/fem/libceed/operator.cpp`) calls
`CeedOperatorLinearAssembleSymbolic` then `CeedOperatorLinearAssemble` --
full sparse (COO) matrix assembly, not just the diagonal. Unlike
`LinearAssembleAddDiagonal`, `ceed-occa` never registered stubs for these two
(they were already null), so the section 7.5 fix does make them reach the
fallback branch -- but doing so **segfaults**, with a real backtrace, not a
clean error:

```
$ ./t500_palace_style /cpu/self/occa   # (with LinearAssembleSymbolic added)
CeedOperatorLinearAssembleAddDiagonal ierr=0
diag: ...
Segmentation fault (core dumped)
```

```
$ gdb -batch -ex run -ex bt --args ./t500_palace_style /cpu/self/occa
Program received signal SIGSEGV, Segmentation fault.
0x... in occa::modeMemory_t::addMemoryRef(occa::memory*) () from libocca.so
#0  occa::modeMemory_t::addMemoryRef(occa::memory*) ()
#1  ceed::occa::Vector::getKernelArg() ()
#2  ceed::occa::ElemRestriction::apply(CeedTransposeMode, ...) ()
#3  CeedElemRestrictionApply ()
#4  CeedOperatorAssembleSymbolicSingle ()
#5  CeedOperatorLinearAssembleSymbolic ()   # <- fallback op, on the fallback (ref) Ceed
#6  CeedOperatorLinearAssembleSymbolic ()   # <- original call, on the occa op
#7  main ()
```

**Root cause:** frame 6 is the fallback dispatch (mirrors section 7.5's
pattern, this time reaching `CeedOperatorGetFallback` because `ceed-occa`
never registered these two in the first place). Frame 5 is the *generic*
implementation running against the fallback (`ref`) `CeedOperator` -- but
frame 2 is `ceed::occa::ElemRestriction::apply()`, `ceed-occa`'s *own*
C++ class, not the ref backend's. **`CeedOperatorGetFallback` does not fully
reparent a `CeedElemRestriction` onto the fallback `Ceed`** -- the generic
assembly algorithm ends up calling `ceed-occa`'s restriction-apply against a
`CeedVector` that was allocated on the fallback (host-memory, no OCCA backing)
Ceed, and `ceed::occa::Vector::getKernelArg()` dereferences an OCCA memory
handle that was never actually created for that vector. This is why
`backends/cuda-gen`/`backends/hip-gen` never hit this: their fallback
(`/gpu/cuda/ref`, `/gpu/hip/ref`) shares the same device memory space as the
main backend, so a "mixed" object graph happens to still work; `ceed-occa`'s
fallback (`/cpu/self/ref/serial`) does not share OCCA's own memory
abstraction at all.

**Not fixed here, deliberately.** This is a cross-backend object-lifecycle
bug inside libCEED's own fallback-operator construction
(`CeedOperatorGetFallback` and whatever builds its cloned field list), not a
`ceed-occa`-local registration issue like 7.5's. Fixing it correctly would
mean understanding and changing how libCEED reparents `CeedElemRestriction`/
`CeedBasis` objects onto a fallback `Ceed` for *every* backend that might use
a fallback with a different memory model, not just `ceed-occa` -- real
upstream libCEED surgery, not a scoped, verifiable-on-this-VM patch, and
exactly the kind of change that risks producing wrong numbers silently if
rushed. Documented here with a full backtrace so the next agent (or an
upstream libCEED issue) doesn't have to re-derive it.

**Consequence: this is not a niche gap.** `palace/linalg/solver.cpp` calls
`ParOperator::ParallelAssemble()`/`StealParallelAssemble()` to get a real
assembled `HypreParMatrix` for essentially every preconditioner setup, and
`ParOperator::ParallelAssemble()` (`palace/linalg/rap.cpp`) calls
`BilinearForm::FullAssemble()` -> `CeedOperatorAssembleCOO()` whenever the
underlying operator is `ceed::Operator`-backed rather than an already-assembled
matrix. `palace/linalg/ams.cpp`'s `HypreAmsSolver` -- Palace's standard
Maxwell/curl-curl preconditioner -- explicitly requires a real
`HypreParMatrix` (`MFEM_VERIFY(A, "HypreAmsSolver requires a HypreParMatrix
operator!")`). **Any default, AMS/AMG-preconditioned Palace solve will reach
this segfault** when its operators are libCEED-backed and libCEED is running
on `ceed-occa`, regardless of the section 7.5 fix. The diagonal fix is real
and matters (some solver configurations only need the diagonal, e.g. plain
Jacobi/Chebyshev smoothing without an AMS/AMG coarse level), but it does not
add up to "Palace runs end to end on `ceed-occa`" for Palace's actual default
configuration.

### 7.7 What this means for Palace, concretely

Two independent routes were identified for OCCA to reach Palace's real
assembly (section 1):

1. Via `mfem::BilinearForm` (MFEM's own OCCA-dispatching partial assembly) --
   closed, because Palace doesn't use that class at all (section 1).
2. Via `ceed-occa` (libCEED's own OCCA backend) -- **partially open** after
   this session's fix: `CeedOperatorApply`/`CeedOperatorApplyAdd` (matrix-free
   operator action) and `CeedOperatorLinearAssembleAddDiagonal`
   (Jacobi/Chebyshev smoothing) are verified working; full sparse-matrix
   assembly (needed by Palace's default AMS/AMG preconditioning) segfaults on
   a separate, deeper, not-fixed-here libCEED bug (7.6).

So: **route 2 is real, not zero, but is not "Palace works on `ceed-occa`"
either.** A Palace configuration that never needs `ParallelAssemble()` (a
matrix-free solve using only Jacobi/Chebyshev-smoothed iterations, if Palace
has such a configuration -- not verified here, would need checking against
Palace's actual solver-selection code and config options) could plausibly run
end to end today. Palace's default, AMS/AMG-preconditioned electromagnetics
solves cannot, until either the 7.6 libCEED bug is fixed upstream or Palace's
own solver setup avoids `ParallelAssemble()` for `ceed::Operator`-backed
operators on `ceed-occa` specifically (not attempted here -- a Palace-side
architectural decision, not a build-wiring one).

### 7.8 Remaining gaps, not attempted

- **Fixing the section 7.6 fallback-reparenting bug in libCEED itself.**
  Real upstream surgery in `CeedOperatorGetFallback`'s object-cloning logic,
  affecting every backend that might register a cross-memory-model fallback,
  not scoped to `ceed-occa`. Left as a documented, reproducible bug (7.6's
  backtrace) rather than attempted blind.
- **`CeedElemRestrictionCreateAtPoints`** (a different, unrelated `ceed-occa`
  gap found while surveying `t5*` tests broadly: `"Backend does not implement
  CeedElemRestrictionCreateAtPoints"`). **Not relevant to Palace** -- checked
  directly, `grep -rn "AtPoints" palace/fem/libceed/*.cpp` returns nothing;
  Palace never calls it.
- **CUDA/HIP `ceed-occa` resources.** No CUDA/HIP toolchain on this VM;
  `occa modes` here only ever reports `Serial`/`OpenMP`. The section 7.5 fix's
  `/gpu/cuda/ref`/`/gpu/hip/ref` fallback selection is untested (code
  reviewed, not run) for the same reason.

## 8. Metal specifically: two independent blockers, not one

The task framing going in was "get to OCCA, then get to Metal, patching MFEM
as needed." Section 7 already shows the MFEM side isn't where the remaining
work is -- MFEM's own `occa-metal` skeleton (the companion MFEM-side doc)
rejects cleanly on non-Apple platforms exactly as designed, and both real
remaining blockers found in this session sit entirely inside libCEED, not
MFEM. Section 8.1 restates where things actually stand after section 7's
fixes (better than originally thought, but still blocked for a real-assembly
Palace solve); 8.2 is a second, independent, Metal-specific blocker on top of
that. No patch to MFEM reaches either one.

### 8.1 Blocker 1, corrected: not "the Operator path doesn't work," but "full-assembly doesn't work, and Metal inherits that too"

An earlier pass of this section claimed the `ceed-occa` Operator+QFunction
path was categorically broken on every mode, Metal included, and that this
alone blocked Metal. Section 7.4 corrects the "categorically broken" part:
`CeedOperatorApply`/`CeedOperatorApplyAdd` and diagonal assembly
(`CeedOperatorLinearAssembleAddDiagonal`) both work correctly, verified with
real numerics, after this session's fix (7.5). What's actually still broken,
mode-agnostically, is the section 7.6 bug: full sparse-matrix assembly
(`CeedOperatorLinearAssembleSymbolic`/`CeedOperatorLinearAssemble`) segfaults
via a cross-backend fallback-reparenting bug in libCEED's
`CeedOperatorGetFallback`. That bug is not specific to Serial/OpenMP -- it is
in how libCEED constructs a fallback operator's `CeedElemRestriction`
objects, which would apply identically to a hypothetical `/gpu/metal/occa`
resource. So: **getting a real electromagnetics operator all the way through
Palace's default AMS/AMG-preconditioned solve via `ceed-occa` -- on any mode,
CPU, OpenMP, or an eventual Metal -- still requires fixing the section 7.6
bug first, and that is an upstream libCEED problem**, not something Palace's
build wiring or an MFEM patch can route around. (`CeedOperatorApplyAdd`-only,
non-AMS/AMG configurations, if Palace has any, are not blocked by this --
see 7.7.)

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
MFEM, and this remains true after the deeper second pass. Real progress was
made and is committed: `CeedOperatorApply`/`CeedOperatorApplyAdd` and
Jacobi/Chebyshev diagonal assembly are genuinely fixed and verified working
on `ceed-occa`'s CPU/OpenMP modes (section 7.5). But two real blockers remain
between here and an actual working `occa-metal` build of Palace, and **both
are in libCEED, not MFEM**:

1. Full sparse-matrix assembly (`CeedOperatorLinearAssembleSymbolic`/
   `CeedOperatorLinearAssemble`) segfaults via a cross-backend
   fallback-reparenting bug in `CeedOperatorGetFallback` (section 7.6) --
   and Palace's default AMS/AMG-preconditioned solves need exactly this, via
   `ParOperator::ParallelAssemble()`. This blocks any mode of `ceed-occa`,
   not just Metal.
2. `CeedScalar` is hardcoded to `double` with no Metal registration in
   `ceed-occa`'s resource parser at all (section 8.2), and Palace's
   zero-copy `mfem::real_t`<->`CeedScalar` aliasing (`CEED_USE_POINTER` in
   `palace/fem/libceed/operator.cpp`) means this is a whole-stack precision
   decision, not a local one.

If Metal-via-libCEED is still the goal, the actual next steps are upstream
libCEED work -- fix the fallback-reparenting bug (or find/report it as a
libCEED issue), then separately design a real precision story for Metal --
not anything reachable from Palace's or MFEM's build systems alone. The
`ceed-occa` build wiring and the diagonal-assembly fix in this branch are a
real, durable improvement to build on, not a dead end, but they don't close
either remaining gap.
