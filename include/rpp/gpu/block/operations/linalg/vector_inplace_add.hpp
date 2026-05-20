#ifndef RPP_GPU_BLOCK_OPERATIONS_LINALG_VECTOR_INPLACE_ADD_HPP
#define RPP_GPU_BLOCK_OPERATIONS_LINALG_VECTOR_INPLACE_ADD_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/views/batch.hpp>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/linalg/vector_inplace_add.hpp>

#include <rpp/gpu/block/strategy.hpp>

namespace rpp::ops {
template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class VectorInplaceAdd<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize,
            Architecture> > : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize,
            Architecture> > {
public:
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

    static constexpr bool is_implemented = true;

    template<typename VectorLhs, typename VectorRhs>
    RPP_DEVICE void operator()(Context const &ctx, VectorLhs &lhs, VectorRhs const &rhs,
                               Accum alpha = Accum{1}) const noexcept {
        using Scalar = typename VectorLhs::value_type;
        auto const &basis = lhs.basis();
        const auto min_degree = std::max(lhs.min_degree(), rhs.min_degree());
        const auto max_degree = std::min(lhs.max_degree(), rhs.max_degree());
        if (max_degree < min_degree) {
            return;
        }

        const auto begin = basis.start_of_degree(min_degree);
        const auto size = basis.end_of_degree(max_degree) - begin;
        auto lhs_data = lhs.data() + begin;
        auto rhs_data = rhs.data() + begin;

        for (Index i = ctx.thread_rank(); i < size; i += ctx.num_threads()) {
            Accum lhs_val{lhs_data[i]};
            Accum rhs_val{rhs_data[i]};
            Accum result = lhs_val + alpha * rhs_val;
            lhs_data[i] = static_cast<Scalar>(result);
        }
    }
};
} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_LINALG_VECTOR_INPLACE_ADD_HPP