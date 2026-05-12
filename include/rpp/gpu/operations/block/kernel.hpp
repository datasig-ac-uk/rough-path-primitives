#ifndef RPP_GPU_OPERATIONS_BLOCK_KERNEL_HPP
#define RPP_GPU_OPERATIONS_BLOCK_KERNEL_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <cuda_runtime.h>

#include <rpp/config.h>
#include <rpp/support/error.hpp>

#include <rpp/basis/basis_pack.hpp>

#include <rpp/gpu/device.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::gpu::block {
namespace detail {

} // namespace detail

template<
    typename Op,
    typename BatchArgs,
    typename BasisPack,
    typename... Extras
>
RPP_KERNEL void kernel(
    const BatchArgs batches,
    const BasisPack bases,
    const typename Op::Index batch_size,
    const Extras... extras
) {
    using Strategy = typename Op::Strategy;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = Strategy::make_context(smem_bytes);

    Op::init_scratch_mem(ctx, bases);

    const auto work_idx = Strategy::object_index(blockIdx.x, threadIdx.x);
    if (work_idx >= batch_size) { return; }

    ops::invoke(Op{}, ctx, [&](auto const &batch) { return batch.view(work_idx, bases); }, batches, extras...);

    Op::destroy_scratch_mem(ctx, bases);
}


} // namespace rpp::gpu::block

#endif //RPP_GPU_OPERATIONS_BLOCK_KERNEL_HPP
