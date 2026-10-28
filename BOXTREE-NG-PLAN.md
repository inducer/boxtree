# Boxtree NG: CPU-native trees, particle classes, and traversals

## Goals and decisions

Migrate away from OpenCL in **four pull requests**, retaining the library's
capabilities rather than just replacing its main tree builder. Keep the array
layouts and FMM-facing contracts recognizable; accept explicit API changes for
execution contexts, device arrays, and kernel-source extension points.

Target architecture:

- NumPy-owned host arrays, Python orchestration, and a small **ahead-of-time
  Cython/OpenMP extension** for irregular walks, partitioning, and bulk parallel
  primitives. No runtime compilation for built-in operations.
- A particle-independent `TreeOfBoxes` geometry layer, independently usable
  particle assignment, and a thin adaptive-build coordinator. A particle `Tree`
  **has a** `TreeOfBoxes` in `tree.topology`; it does not inherit from one.
  Particle handling consumes a topology contract, not the implementation of
  `tree_of_boxes.py`.
  Box-topology construction and editing stay pure Python/NumPy for now; the
  Cython boundary starts with particle handling and traversal, not box creation.
- Named **particle classes**, with `source` and `target` compatibility aliases.
  Interaction builders select an ordered pair of classes; emitting and receiving
  are roles of an operation, not properties of the geometry or native kernels.
- Parallel CPU implementations of particle handling, traversal, queries, and
  other bulk stages currently run in parallel, with the explicit exception of
  pure-NumPy box-topology construction/editing. Use box/row/chunk parallelism and
  level barriers instead of reproducing GPU-wide scans and per-particle
  intermediate structures literally.
- Use runtime dimension loops throughout Python and Cython, including axis and
  child loops (`range(dim)` and `range(2**dim)`). No dimension specialization,
  generated kernels, or runtime compilation. The same implementation covers
  1D/2D/3D; validate dimensions and scratch/allocation sizes at the boundary.

Cython/OpenMP is the chosen native implementation, not a performance claim.

## NumPy and Cython/OpenMP responsibilities

NumPy is appropriate for array algebra, validation, small problems, and a clear
reference implementation. It is not a sufficient production implementation of
ragged recursive walks: Python loops retain interpreter/GIL overhead, while
vectorizing a whole walk can require large temporary frontier arrays. NumPy
sorting and many indexing operations also do not guarantee multithreading.
Measure these issues rather than assuming every existing kernel needs a native
replacement. Box-topology construction/editing is explicitly kept in NumPy for
this migration, even if profiling identifies it as a future optimization target.

Cython supplies typed memoryviews, direct NumPy buffer access, `nogil` helpers,
and OpenMP `prange` loops for particle partitioning/sorting, irregular walks,
queries, and bulk parallel primitives. This keeps the implementation close to
existing Python code and supports incremental porting. The costs are manual
bounds/lifetime/race discipline and OpenMP runtime/wheel integration, especially
on macOS and Windows. Dimension-dependent arithmetic uses ordinary runtime
loops, not templates, source generation, or dimension-specific entry points.
Float32/float64 buffer dispatch may use Cython fused scalar types; this is dtype
handling, not dimension specialization.

In PR 1 compare NumPy and Cython/OpenMP stable partitioning and representative
interaction rows. Include allocations and buffer-normalization copies, not just
time inside a tight loop. Keep NumPy reference implementations for correctness;
use measurements to choose native particle/traversal kernels and serial thresholds.

Add an AOT Cython extension build backend/build hook (the current
`pyproject.toml` uses pure hatchling), with Cython and NumPy headers as build
dependencies. Build wheels with a supported
OpenMP runtime and portable CPU baseline, not `-march=native`. Release builds
must actually support parallel execution; a serial developer fallback is not
completion of this migration. NumPy becomes an explicit runtime dependency. Retain unrelated dependencies where still used,
for example symbolic cost modeling; audit rather than indiscriminately remove
Mako, cgen, arraycontext, or loopy.

## Architecture and data contracts

Names below describe proposed APIs, not existing symbols.

### Geometry owns boxes, not populations

Retain `TreeOfBoxes` and the public refinement/coarsening/mesh conversion
functions. Its canonical state is root extent, centers, parent IDs, child IDs,
levels, level starts, structural flags, dtypes, and pruning state. Keep the
current structure-of-arrays layout: centers `(dim, aligned_nboxes)`, children
`(2**dim, aligned_nboxes)`, contiguous scalar arrays, level-major box IDs, root
ID zero, and zero for an absent child. Do not require GPU padding, but retain
`aligned_nboxes` as the actual storage stride.

Keep root creation, box refinement/coarsening, box renumbering, topology pruning,
and edit-map construction in pure Python/NumPy, including when called by the
particle-driven coordinator. Do not move these operations into the Cython
extension in these four PRs. Batch NumPy edits where practical, and profile
geometry separately so its costs remain visible.

`TreeOfBoxes` currently lives in `tree.py`, has source/target extent metadata,
and `_compute_tob_box_flags` treats all boxes as both sources and targets.
Remove this semantic dependency from the geometry core. Derive leaves from
children; make population flag bits compatibility data only. A particle-free
tree must still support box-based traversal via an explicit all-box occupancy
view, rather than fictional particles.

Normalize level starts to `nlevels+1` entries, including the final sentinel, and
choose `-1` for the canonical root parent, as used by `make_tree_of_boxes_root`.
The particle `Tree` documents root parent zero: preserve zero in the legacy
view until that contract is deliberately retired. Test both representations;
do not let sentinel remapping index `-1` or remap absent child zero.

Define a small read-only topology protocol containing these arrays and scalar
geometry operations. `TreeOfBoxes` implements it; particle assignment and
traversal accept it without calling its private refinement functions. External
conforming topologies are possible after validation.

Topology edits return an edit result with the new topology, old-to-new and
new-to-old box maps (explicit deleted/new sentinels), and split/coarsen ancestry.
Existing public functions can continue returning just the tree, with an opt-in
edit-result API for consumers. Expose deterministic remapping; do not rely on
private sorting order. Geometry-only coarsening must remain usable without
particle code. Pruning for particle trees removes empty subtrees only if empty
across all participating classes; mesh adaptivity may intentionally retain empty
leaves. Keep these policies separate.

### A particle tree contains geometry and assignments

Retain the public name `Tree`, but make it a standalone aggregate, not a subclass
of `TreeOfBoxes`. Its canonical fields are `topology: TreeOfBoxes` and
`particle_classes: Mapping[ClassKey, ParticleAssignment]`, plus any pair-selection
metadata required by source/target compatibility views. Geometry arrays exist
only on the contained topology; assignments own only particle/class data. Neither
object holds a back-reference to the aggregate. Several particle trees may share
the same immutable topology with different populations or assignments.

Geometry-only functions receive `tree.topology` explicitly, not the particle
`Tree`. Particle operations receive a topology plus assignments, or unpack them
from a `Tree` convenience argument. The particle aggregate does not itself
implement the topology protocol: convenience must not recreate an implicit
is-a relationship through blanket delegation or `__getattr__` forwarding.

Topology edits create a new topology and remapped assignments, then return a new
`Tree` aggregate atomically. Never replace `tree.topology` while retaining stale
assignments. Serialization/container traversal records the topology once per
aggregate and preserves explicit population aliases; reconstruction validates
that all assignments match that topology version.

### Particle classes and assignments

Use a mapping `class_key -> ParticleAssignment`. Keys are stable names, not an
enum limited to two values and not an ever-growing bit field. Each assignment
contains:

- Coordinates in tree order (retain the object-array-of-axis-arrays interface),
  optional radii, and extent policy (`linf`/`l2`, stick-out factor).
- `box_starts`, `box_counts_nonchild`, `box_counts_cumul`, and optional per-box
  particle bounding-box minima/maxima, using the existing shapes and meanings.
- Explicit `tree_to_user` and `user_to_tree` permutations; per-class particle
  index space, without global concatenation or ID-threshold classification.
- Direct and descendant occupancy derived from counts, plus any explicit
  distributed/global occupancy override. Storage need not allocate a dense
  classes-by-boxes tensor when only some classes require particular metadata.

A class identifies a population, not a Python subclass. Two names may explicitly
alias the same population/assignment: the old `targets=None` case stores and
counts it once. Equal coordinates alone do not imply identity. Separate
populations at the same coordinates remain distinct. The coordinator aggregates
refinement weights once per participating population, with configurable class
participation; the legacy separate-source/target call retains its concatenated
weight semantics. Preserve integer nonnegative weights, validation, and threshold
behavior, including safe accumulation without overflow.

For each box, directly owned particles precede its descendants in particle
order. Every subtree occupies a contiguous interval in each class; box IDs
remain level-major while particle order is depth-first by fixed child order.
This is required by cumulative starts/counts and downstream FMM code. Merely
sorting particles by final level-major box ID is incorrect when particles with
extent stay in internal boxes. Stable ties use original user ID.

Assignment supports an already-built topology, independently of adaptive
construction. A particle descends into its center-selected child only when its
extent fits according to the existing predicates; otherwise it stays in the
parent, including internal boxes. Preserve the asymmetric boundary comparisons
and `l2` versus `linf` semantics from `tree_build_kernels.py`, not a generic
sphere/AABB replacement. Bounding boxes include radii and descend from all
children; retain the empty-box center convention.

On a topology edit, assignments consume edit maps and redistribute affected
subtree ranges; coarsening collects former descendant particles, refinement
partitions only affected ranges. A complete reassignment is a correctness
fallback/reference, not the default at every refinement level. Assignments
reference an immutable topology version; derived lists and bounds cannot be
silently reused after an edit. New populations may be attached without rebuilding
the topology; requesting adaptation is a separate operation.

### Adaptive construction is a coordinator

The coordinator repeats occupancy/weight evaluation, geometry refinement, and
assignment updates. It owns stopping and balancing policy; neither geometry nor
particle storage calls back into the other. The Python coordinator calls the
pure-NumPy geometry API for box edits and Cython/OpenMP particle operations for
assignment updates. Batch edits and reuse particle workspaces where practical;
avoid unnecessary repeated full-tree copies/sorts, but do not introduce a native
box builder to eliminate them. Any shared geometry-only batched edit helper
remains pure NumPy and also serves the public functional API.

Preserve adaptive, adaptive-level-restricted, and non-adaptive modes, automatic
radius-aware bounding boxes and root stretch, explicit boxes, empty-leaf pruning
and unpruned construction. Match the current split decision based on the weight
of particles eligible to descend: particles stuck at the parent must not cause
endless refinement. Preserve the non-adaptive common-depth behavior, and compute
level-restriction closure using the current adjacency definition, not just
face-neighbor balancing. Forced splits, occupancy splits, and particle extent
stopping are separate decisions. Keep precision-based depth limits and
`MaxLevelsExceeded`; no fixed-width Morton key may silently truncate deep trees.

Use stable small-bin partitioning (stay-in-parent plus `2**dim` children) rather
than requiring one global Morton sort. Parallelize over boxes and chunk unusually
large boxes, including the root; otherwise the first and most expensive levels
would be serial. A parallel radix/sort alternative is useful only if benchmarks
justify it and it preserves internal-box ownership and deep-tree behavior.

### Traversals and compatibility views

`build_traversal(topology, particles, emitter_class, receiver_class, ...)` selects
occupancy, bounds, and extents for that pair. Kernels take selected array bundles,
not class names or source/target branches. Cache geometry-only peer information
separately from pair-dependent interaction lists. Three or more classes, reversed
pairs, self-interaction, and explicit aliases must work without new native code.
This does not require a new many-species numerical FMM solver: the existing FMM
wrangler continues to operate on a selected emitter/receiver pair.

Keep `Tree`, `FMMTraversalInfo`, CSR `*_starts`/`*_lists`, level arrays, and the
wrangler/`drive_fmm` boundary. `Tree` is the composed aggregate described above,
not a geometry subtype. Existing `sources`, `targets`, starts/counts/bounds,
and source/target flags are explicit compatibility views of the chosen pair.
For frequently used geometry reads, retain a small documented set of read-only
forwarding properties (for example `tree.nboxes`, `tree.dimensions`,
`tree.box_centers`, and `tree.box_child_ids`) where they materially reduce
wrangler changes. These properties reference `tree.topology` without duplicating
arrays; new core code uses the contained topology explicitly. Do not forward
refinement/coarsening or synthesize the full topology protocol on `Tree`.

Breaking changes are intentional: `isinstance(tree, TreeOfBoxes)` is false,
geometry-only calls become, for example, `refine_tree_of_boxes(tree.topology, ...)`
or `make_meshmode_mesh_from_leaves(tree.topology)`, and dataclass replacement of
geometry fields on `Tree` is no longer valid. A geometry-only edit returns a
geometry object, not an updated particle tree; callers needing particle updates
use the coordinator/edit-and-reassign operation. Update plotting, query, and
serialization call sites to select geometry versus particle data explicitly.
In particular:

- `user_source_ids = source.tree_to_user`;
  `sorted_target_ids = target.user_to_tree`. Their directions differ today.
- `sources_are_targets` denotes explicit population identity, not array equality.
- Generic occupancy is class-local; legacy `box_flags` retain their bit values.
- Preserve CSR row spaces: same-level neighborhoods index global boxes; direct
  and close lists index `target_boxes`; Lists 2/4 index
  `target_or_target_parent_boxes`; List 3 has compressed target rows per source
  level. List entries remain global box IDs.
- Replace `pyopencl.algorithm.BuiltList` with an owned host record retaining its
  used fields, including compressed-row maps and counts. Preserve nested
  dataclass/object-array/optional-array serialization without mandatory OpenCL.

The primary CPU API is synchronous and returns a result, with no context, queue,
allocator, event, or `wait_for`. Provide a documented mechanical migration table
for old calls. Do not make temporary compatibility entry points (such as to accept a host array
context and return `(result, None)` with a warning). Optional explicit arraycontext
transfer utilities let GPU downstream consumers upload host-built trees/lists
once; device computation and synchronization stay outside the CPU core.

OpenCL/Mako custom query strings cannot stay source-compatible. Retain a callable
query capability with a documented ahead-of-time native callback/extension API
and a slower Python callback reference route. Built-in queries ship precompiled;
custom native extensions compile at package build time, not first invocation.
Document ownership, callback lifetime, error propagation, and whether callbacks
may run without the GIL. No Python callbacks in parallel `nogil` loops.

## Parallel execution and correctness rules

- Keep allocations, validation, and exceptions at the Python boundary; release
  the GIL for native work. Read-only input views remain owned for the full call;
  document no concurrent caller mutation and never resize borrowed arrays.
- Use count -> parallel prefix sum -> fill for ragged lists and partitions,
  disjoint row writes, and per-worker scratch. Share predicates between count
  and fill. Check allocation sizes/index overflow before writing.
- Parallelize independent particle chunks, per-box particle operations, query balls, interaction rows,
  and cost/classification reductions. Use dynamic scheduling for skewed walks,
  fixed child order within rows, and level barriers for parent/child dependencies.
  Avoid shared append buffers and racing occupancy writes, even identical writes.
- Use a Cython/OpenMP parallel scan/sort implementation for particle handling
  and list construction where needed; a call to `np.argsort` alone is not a
  parallel implementation. Small-input serial thresholds are acceptable; expose
  thread control and test one-thread and multi-thread results. Box-topology
  editing/renumbering is the explicit pure-NumPy exception: do not require it to
  become OpenMP-parallel in this migration.
- Avoid nested OpenMP/BLAS/MPI oversubscription; document threads per rank and
  reuse a single threading runtime. Maintain MPI communication order and only
  invoke MPI on threads allowed by the requested MPI thread support.
- Do not enable fast-math initially. Preserve geometric epsilon/tangency rules;
  deterministic integer lists and stable permutations are the default. Numerical
  behavior changes, such as fixing float32 space-invader accumulation for float64
  inputs, require separate documented tests rather than accidental port drift.

## Four pull requests

Each PR is mergeable with the existing implementation still usable until the
final cutover. Temporary OpenCL code is an oracle, not a permanent second backend.
The scopes below are intentionally substantial; do not split off functionality
into an indefinite post-migration backlog.

### PR 1 — Contracts, geometry boundary, and native execution foundation

**Changes:** isolate host data records/serialization and topology protocol from
OpenCL imports in `tree.py`, `array_context.py`, and package initialization.
Introduce the standalone `Tree(topology, particle_classes, ...)` aggregate,
remove its `TreeOfBoxes` inheritance, and add class inputs/assignment records,
pair views, and narrowly scoped compatibility properties. Separate geometry and
particle serialization and update geometry-only call sites to use `tree.topology`.
Make `tree_of_boxes.py` genuinely geometry-only, add edit maps and
canonical sentinels/level starts, while retaining its public functional API and
mesh conversion and its pure-NumPy implementation. Add NumPy reference
assignment/predicates, no production claim for a serial particle reference.
Introduce the AOT Cython/OpenMP build, float32/float64 dispatch, runtime dimension
and child loops, threading control, and parallel scan/partition primitives.
Do not introduce generated or dimension-specialized kernels.

**Evidence/exit gate:** existing pure-box behavior and mesh tests remain valid;
new topology/remapping, permutation, alias, empty-class, and three-class tests
pass. Assert that `Tree` is not a `TreeOfBoxes`, geometry arrays are not duplicated,
and multiple particle aggregates can share one topology. Test aggregate
serialization and rejection of mismatched topology/assignment versions; verify
geometry-only APIs require the contained topology rather than implicit forwarding.
Benchmark representative NumPy and Cython/OpenMP partition and ragged-walk
implementations at one and several threads, including skewed inputs. Record
crossover thresholds, memory use, and Cython/OpenMP wheel-build results before
expanding the native implementation. Test the same dimension-loop code in
1D/2D/3D and verify geometry operations need no Cython extension calls. Capture OpenCL golden/reference results and benchmark
inputs while the old code is available; avoid brittle equality of arbitrary box
numbers where equivalent topology ordering is allowed.

### PR 2 — Independent particle assignment and replacement tree builder

**Changes:** implement host particle assignment and coordinator described above,
including parallel root/large-box partitioning, extent stopping, weighted
refinement, all three build modes, level-restriction closure, bounding boxes,
pruning, final permutations/counts/bounds, and update-after-refine/coarsen.
All box creation/editing/pruning stays in the pure-NumPy geometry layer;
Cython/OpenMP accelerates particle operations, not construction of box topology.
The builder returns a composed `Tree`; incremental edits return a new aggregate
with the new topology and consistently remapped class assignments.
Keep the legacy builder available for differential tests. Generalize particle
filtering (user-order CSR and compact tree-order forms), linked point populations,
and associated inverse maps; retain `ParticleListFilter`,
`TreeWithLinkedPointSources`, and `link_point_sources` compatibility interfaces.
Port their scans/gathers and necessary `bounding_box.py`/`tools.py` primitives.

**Evidence/exit gate:** port all tree-building cases from `test/test_tree.py`;
compare invariants/topology/ownership with the old builder on shared and separate
populations, radii in both norms, zero/positive weights, float32/float64,
non-adaptive and balanced trees, explicit bounding boxes, reallocation/pruning,
and 1D. Add empty populations, coincident/indistinguishable points, exact split
planes and root upper boundaries, deep trees, overflow rejection, three classes,
and alias-weight tests. Compare incremental assignments with full reassignment
through randomized refine/coarsen sequences. Assert subtree-contiguous ranges and
particle conservation, not just leaf capacities. Report phase timings, peak memory,
and scaling; no per-particle vector of all child-bin counts is required.

### PR 3 — Native traversals, queries, and interaction metadata

**Changes:** port `traversal.py`, `area_query.py`, `translation_classes.py`, and
`rotation_classes.py` to shared compiled geometric walkers and parallel CSR
builders. Retain Lists 1–4, close Lists 3/4 and merging, source/target ancestor
lists and level starts, 1-/2-away and other supported separation settings,
`static_linf`/`precise_linf`/`static_l2`, particle-adaptive bounds, List 3 source
thresholding, and distributed source masks. Preserve current supported
combinations and explicit rejection of unsupported ones (notably extent-bearing
emitters in FMM traversal); class generality must not quietly invent unsupported
FMM mathematics. List 4 may include same-level boxes for 2-away.

Retain adaptive peer lists, ball-to-leaf queries, their leaf-to-ball transpose,
space-invader maxima, caller-supplied peer lists, and fused custom query
functionality. Include balls centered outside the root and boundary tangency.
Implement the native extension contract for custom queries, with a working
ported example of `AreaQueryElementwiseTemplate` functionality. Preserve
translation/rotation class alignment with List 2 and per-level metadata.

**Evidence/exit gate:** native connectivity/classification tests and small
brute-force query oracles pass; `test/test_traversal.py` and FMM completeness
cases pass for both dtypes, extents, filters, criteria, and selected class pairs.
Add exact per-source/receiver interaction coverage tests: all-one totals alone
can conceal a duplicate canceled by an omission. Compare row-space maps and
compressed List 3 metadata, not just unordered list contents. Validate no missing
or double-counted propagated interactions. Benchmark CSR count/fill separately,
query reductions/transpose, list memory, and parallel scaling on balanced and
strongly adaptive trees.

### PR 4 — Full-stack migration, MPI, packaging, and OpenCL removal

**Changes:** switch FMM orchestration, constant-one and FMMLib integrations,
`cost.py`, visualization, examples, documentation, and remaining utilities to the
host records. Preserve multiple weight vectors, output permutations, gradients,
dipoles, Laplace/Helmholtz, per-level orders, optimized 3D M2L and its memory cap,
per-stage timing/cost calibration, and filtered target counts. Preserve the
wrangler interface where unrelated to device execution.

Port all distributed setup: cost-weighted DFS partitioning, responsibility and
ancestor masks, local particle compaction/counts, local/global maps, masked local
traversals, box-to-multipole-user-rank lists, and parallel communication filtering.
Keep global box numbering and MPI ownership/reduction protocols, sparse
reduce/scatter and full Allreduce, and output gathering. **Global emitter
occupancy must survive local source removal** so remote multipole interactions
are not lost; distinguish global population identity from locally aliased storage.
Keep `AllReduceCommPattern` behavior for non-power-of-two rank counts.

Remove `tree_build_kernels.py` and obsolete OpenCL kernels only after feature
parity. Remove mandatory PyOpenCL imports/dependency and obsolete runtime code
generation from core modules, update build metadata/CI, and separate any optional
external arraycontext/meshmode GPU interoperability tests from core installation.
Publish a migration guide for builders, arrays, result tuples, transfer utilities,
class views, composition (`tree.topology` instead of geometry inheritance),
serialization, and custom extensions. Audit downstream subtype checks and
geometry-field dataclass replacements as well as ordinary attribute access.
Do not replace ndarray with a
wrapper solely to imitate `.get()`/`.events`.

**Evidence/exit gate:** the host suite covers all eight current test modules,
including cost-model reference comparisons, FMMLib numerical stability/optimized
M2L, filtering, linked populations, mesh conversion, and tool utilities. Run MPI
parity with shared-memory results, both multipole communication modes, target
extents, empty local populations and irregular rank counts. Run with one and
multiple threads per rank. Build/install wheels on supported platforms and run
core import/build/query/FMM tests in an environment without PyOpenCL or an OpenCL
runtime. Check optional interoperability separately. CI must not pass merely
because old `opencl` markers skipped the migrated tests.

## Acceptance and performance reporting

Maintain a feature-to-test checklist within these PRs, starting from existing
tests and adding the gaps above. Run old and new implementations on identical
seeded cases during transition; compare semantic invariants and numerical results,
with explicit tolerances, rather than requiring identical incidental allocation
or traversal ordering. Preserve deterministic new ordering across thread counts.
Snapshot/serialization changes are documented compatibility breaks, not silently
accepted stale metadata.

Benchmarks cover 2D/3D, float32/float64, shared/separate/three-class populations,
uniform/clustered/degenerate distributions, radii, balanced/unbalanced trees,
repeated assignment, queries, traversal, costs, and MPI setup. Include small
latency cases and memory-fitting sizes around 10^5–10^7 particles where hardware
permits. Report total wall time, per-phase times, peak RSS/scratch, and thread
scaling, including setup and all copies; compare NumPy and native one-thread
baselines plus the old OpenCL implementation with hardware and compilation/transfer
costs clearly identified. Set regression budgets from PR 1 measurements, not
invented speedups. Report pure-NumPy box construction/editing separately from
Cython/OpenMP particle and traversal work. A stage-by-stage parallelism audit is
required at cutover: outside the explicitly retained NumPy box-topology layer,
replacing parallel particle sort, scan, compaction, or traversal with a serial
production loop is not feature-complete merely because numerical tests pass.
