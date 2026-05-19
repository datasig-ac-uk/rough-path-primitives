# RoughPathPrimitives

`RoughPathPrimitives` is a C++ library for computational rough path theory.

The core of the library is a set of algebraic operations on truncated tensor and Lie objects, exposed through a common operation interface and implemented for different execution devices and strategies. The library is designed to make the mathematical object model explicit while keeping the execution model replaceable.

## Design

The central idea is to separate the problem into four layers:

1. `basis`
   Describes the algebraic indexing structure. A basis knows how tensor or Lie coefficients are arranged, how degrees are laid out, and how combinatorial indexing works.

2. `views`
   Views are lightweight handles onto concrete data. A view combines raw storage with the metadata needed to interpret that storage as a graded vector, tensor, Lie element, or scalar.

3. `batch`
   Batches package data, layout, and metadata so that operations can request a view for a particular batch index. This is the bridge between raw memory and the per-object interfaces used by kernels.

4. `operations`
   Operations express the actual rough path algebra: tensor products, adjoint actions, pairings, exponential and logarithm maps, and related linear algebra utilities.

This split is deliberate. The same algebraic operation should be able to run against different storage layouts and on different execution backends without changing the mathematical interface.

## Philosophy

The library is built around a few constraints.

First, the operation layer is the point of the project. This is not primarily a container library. Data structures exist to serve the kernels that implement rough path computations.

Second, metadata and storage should stay separate. Basis objects, degree ranges, and layout information are different in nature from coefficient buffers. That separation makes it easier to:

- reuse the same operation across CPU and GPU implementations
- construct views cheaply from batches
- change layout and batching without rewriting algebraic code
- keep the algebraic intent visible at call sites

Third, device-specific code should be isolated behind strategies and specializations rather than spread through the public interface. A caller should ask for an operation such as `tensor_pairing` or `ft_mul`; the selected strategy should determine how it is launched.

## Public Structure

The public headers are organized around that model.

- `include/rpp/basis.hpp`
  Umbrella header for basis types.

- `include/rpp/views/views.hpp`
  Umbrella header for dense views such as:
  - `DenseGradedVectorView`
  - `DenseTensorView`
  - `DenseLieView`
  - `ScalarView`

- `include/rpp/views/batch.hpp`
  Generic batch abstraction used to materialize views from data and layout.

- `include/rpp/operations.hpp`
  Umbrella header for the operation interfaces.

- `include/rpp/cpu/strategies.hpp`
  CPU execution strategies.

- `include/rpp/gpu/strategies.hpp`
  GPU execution strategies and launch support.

The current `sparse` directory is different in character from `views`: it mostly holds metadata for operations rather than mutable dynamic objects. It is therefore kept separate from the dense view and batch model.

## Operation Model

Operations are defined in two stages.

1. A public operation interface in `include/rpp/operations/...`
2. One or more device/strategy-specific implementations in headers such as:
   - `include/rpp/cpu/operations/single_thread/...`
   - `include/rpp/gpu/operations/block/...`

This gives the library a stable mathematical API with backend-specific implementations selected by the strategy type and the included specialization headers.

Typical examples include:

- linear algebra utilities
  - `vector_add`
  - `vector_assign`
  - `vector_set_constant`
  - `sparse_matrix_vector`

- basic rough path algebra
  - `ft_mul`
  - `ft_fma`
  - `st_mul`
  - `tensor_pairing`
  - `tensor_antipode`
  - `lie_to_tensor`
  - `tensor_to_lie`

- intermediate constructions
  - `ft_exp`
  - `ft_fmexp`
  - `ft_log`

## Basis, Views, and Batches

The basis layer provides the combinatorial structure for truncated tensor and Lie objects.

The view layer turns raw memory into typed mathematical objects. A view should be cheap, direct, and explicit about what metadata it needs. For dense objects, that usually means:

- a data handle or iterator
- a basis or basis tag
- a degree range

The batch layer exists to create views. A batch is not intended to be a heavy algebraic object in its own right. Its purpose is to store:

- raw data
- layout information
- metadata required to construct a view at a batch index

That makes batches a convenient interface for launch code while keeping the kernels written in terms of the actual mathematical objects they consume.

## Devices and Strategies

The library currently separates execution by strategy rather than by embedding backend logic into every operation.

- CPU support is centered on single-threaded execution.
- GPU support is centered on block-based CUDA execution.

This keeps the public operation names stable while allowing device-specific launch configuration, scratch-space handling, and kernel structure to vary with the backend.

## Build

The project uses CMake and currently builds as a header-only interface library, with an optional CUDA interface target.

Main options:

- `RPP_ENABLE_CUDA`
  Enable CUDA support and the `RPP::GPU` target.

- `RPP_ENABLE_TESTS`
  Build the test suite.

- `RPP_ENABLE_BENCHMARKS`
  Build the benchmark targets.

- `RPP_ADD_STUB_TARGET`
  Build a target that includes the exported headers to catch integration and include-surface issues.

Basic configure/build flow:

```bash
cmake -S . -B build
cmake --build build
```

With CUDA enabled:

```bash
cmake -S . -B build -DRPP_ENABLE_CUDA=ON
cmake --build build
```

## Current Direction

The library is moving toward a cleaner separation between:

- basis metadata
- dense views
- generic batch construction
- operation implementations

That direction is intended to make the kernel code simpler, make view construction more uniform, and reduce type duplication in the batching layer without weakening the mathematical structure of the interfaces.
