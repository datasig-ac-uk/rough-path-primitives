#ifndef RPP_GPU_BLOCK_OPERATIONS_LINALG_VECTOR_ASSIGN_HPP
#define RPP_GPU_BLOCK_OPERATIONS_LINALG_VECTOR_ASSIGN_HPP

// IWYU pragma: always_keep

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/linalg/vector_assign.hpp>

#include <rpp/gpu/block/operations/linalg/vector_set_constant.hpp>
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

private:
    using SetConstant = VectorSetConstant<Strategy>;
    SetConstant set_constant;

public:
    template <typename VectorOut, typename VectorArg>
    RPP_DEVICE void operator()(Context const& ctx,
                               VectorOut& out,
                               VectorArg const& arg) const noexcept {
        using Scalar = typename VectorOut::value_type;
        auto const& basis = out.basis();

        const auto copy_begin_degree =
            std::max(out.min_degree(), arg.min_degree());
        const auto copy_end_degree =
            std::min(out.max_degree(), arg.max_degree());

        if (copy_begin_degree > copy_end_degree) {
            set_constant(ctx, out, Scalar{0});
            return;
        }

        const auto begin = basis.start_of_degree(copy_begin_degree);
        auto size = basis.end_of_degree(copy_end_degree) - begin;

        auto arg_data = arg.data() + begin;
        auto out_data = out.data() + begin;

        for (Index i = out.begin_index() + ctx.thread_rank(); i < begin;
             i += ctx.num_threads()) {
            out[i] = Scalar{0};
        }

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

        for (Index i = basis.end_of_degree(copy_end_degree) + ctx.thread_rank();
             i < out.end_index();
             i += ctx.num_threads()) {
            out[i] = Scalar{0};
        }
    }
};
} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_LINALG_VECTOR_ASSIGN_HPP
