#ifndef RPP_GPU_BLOCK_OPERATIONS_LINALG_VECTOR_ADD_HPP
#define RPP_GPU_BLOCK_OPERATIONS_LINALG_VECTOR_ADD_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/linalg/vector_add.hpp>

#include <rpp/gpu/block/operations/linalg/vector_set_constant.hpp>
#include <rpp/gpu/block/strategy.hpp>

namespace rpp::ops {
template <typename Accum_,
          unsigned BlockSize,
          unsigned MaxBlockSize,
          typename Architecture>
class VectorAdd<
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
    using Index = typename Strategy::Index;
    static constexpr bool is_implemented = true;

private:
    using SetConstant = VectorSetConstant<Strategy>;
    SetConstant set_constant;

public:
    template <typename VectorOut, typename VectorLhs, typename VectorRhs>
    RPP_DEVICE void operator()(Context const& ctx,
                               VectorOut& out,
                               VectorLhs const& lhs,
                               VectorRhs const& rhs,
                               Accum alpha = Accum{1},
                               Accum beta = Accum{1}) const noexcept {
        using Scalar = typename VectorOut::value_type;
        auto const& basis = out.basis();
        const auto min_degree =
            std::max({out.min_degree(), lhs.min_degree(), rhs.min_degree()});
        const auto max_degree =
            std::min({out.max_degree(), lhs.max_degree(), rhs.max_degree()});
        // if (max_degree < min_degree) {
        //     set_constant(ctx, out, Scalar{0});
        //     return;
        // }
        //

        const auto lhs_begin = lhs.begin_index();
        const auto lhs_end = lhs.end_index();
        const auto rhs_begin = rhs.begin_index();
        const auto rhs_end = rhs.end_index();

        for (Index i = out.begin_index() + ctx.thread_rank();
             i < out.end_index();
             i += ctx.num_threads()) {

            Accum val{0};
            if (lhs_begin <= i && i < lhs_end) {
                val += alpha * static_cast<Accum>(lhs[i]);
            }
            if (rhs_begin <= i && i < rhs_end) {
                val += beta * static_cast<Accum>(rhs[i]);
            }

            out[i] = static_cast<Scalar>(val);
        }

    }
};
} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_LINALG_VECTOR_ADD_HPP
