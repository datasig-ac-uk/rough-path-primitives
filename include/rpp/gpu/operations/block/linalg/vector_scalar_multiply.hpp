#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_VECTOR_SCALAR_MULTIPLY_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_VECTOR_SCALAR_MULTIPLY_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/views/batch.hpp>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/linalg/vector_scalar_multiply.hpp>

#include <rpp/gpu/operations/block/strategy.hpp>

namespace rpp::ops {
template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class VectorScalarMultiply<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > : public
        BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

public:
    static constexpr bool is_implemented = true;

    template<typename Vector>
    RPP_DEVICE void operator()(Context const &ctx, Vector &vec, Accum scalar) const noexcept {
        using Scalar = typename Vector::value_type;
        auto const &basis = vec.basis();
        const auto min_degree = vec.min_degree();
        const auto max_degree = vec.max_degree();
        if (max_degree < min_degree) {
            return;
        }

        const auto begin = basis.start_of_degree(min_degree);
        const auto size = basis.end_of_degree(max_degree) - begin;
        auto data = vec.data() + begin;

        for (Index i = ctx.thread_rank(); i < size; i += ctx.num_threads()) {
            Accum val{data[i]};
            Accum result = val * scalar;
            data[i] = static_cast<Scalar>(result);
        }
    }
};
} // namespace rpp::ops

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_VECTOR_SCALAR_MULTIPLY_HPP
