#ifndef RPP_GPU_BLOCK_OPERATIONS_LINALG_VECTOR_ASSIGN_HPP
#define RPP_GPU_BLOCK_OPERATIONS_LINALG_VECTOR_ASSIGN_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/linalg/vector_assign.hpp>

#include <rpp/gpu/block/strategy.hpp>

namespace rpp::ops {
template <typename Accum_,
          unsigned BlockSize,
          unsigned MaxBlockSize,
          typename Architecture>
class VectorAssign<
    gpu::strategies::
        BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>>
    : public BaseOperation<
          gpu::strategies::
              BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>> {
public:
    using Strategy = gpu::strategies::
        BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Architecture_ = typename Strategy::Architecture;
    using Index = typename Strategy::Index;

    static constexpr bool is_implemented = true;


    template <typename VectorOut, typename VectorArg>
    RPP_DEVICE void operator()(Context const& ctx,
                               VectorOut& out,
                               VectorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        const auto begin =
            basis.start_of_degree(std::max(out.min_degree(), arg.min_degree()));
        auto size =
            basis.end_of_degree(std::min(out.max_degree(), arg.max_degree())) -
            begin;

        auto arg_data = arg.data() + begin;
        auto out_data = out.data() + begin;
        if constexpr (std::is_pointer_v<decltype(arg_data)>) {
            const auto count_to_align = static_cast<Index>(
                (Architecture_::sector_alignment -
                 (reinterpret_cast<std::uintptr_t>(arg_data) &
                  (Architecture_::sector_alignment - 1))) /
                sizeof(*arg_data));

            for (Index i = ctx.thread_rank();
                 i < std::min(count_to_align, size);
                 i += ctx.num_threads()) {
                out_data[i] = arg_data[i];
            }
            arg_data += count_to_align;
            out_data += count_to_align;
            size -= count_to_align;

            for (Index i = ctx.thread_rank(); i < size;
                 i += ctx.num_threads()) {
                out_data[i] = arg_data[i];
            }
        }
        else {
            for (Index i = ctx.thread_rank(); i < size;
                 i += ctx.num_threads()) {
                out_data[i] = arg_data[i];
            }
        }
    }
};
} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_LINALG_VECTOR_ASSIGN_HPP