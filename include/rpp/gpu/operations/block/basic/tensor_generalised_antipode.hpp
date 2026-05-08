#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/dense/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/tensor_generalised_antipode.hpp>

#include <rpp/gpu/strategies.hpp>

namespace rpp::ops {
template <typename Accum_, unsigned BlockSize, typename Architecture, TensorAntipodeSigningPolicy Policy>
class TensorGeneralisedAntipode<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>, Policy, void>
    : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {

    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;

public:
    using Context = typename Strategy::Context;

    template <typename TensorOut, typename TensorArg>
    RPP_DEVICE void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
        auto const& basis = out.basis();

        for (auto elt_idx=ctx.thread_rank(); elt_idx<static_cast<typename Strategy::Index>(arg.size()); elt_idx += ctx.num_threads()) {
            const auto degree = basis.degree(elt_idx);
            const auto degree_begin = basis.start_of_degree(degree);
            const auto rev_idx = basis.reverse_index(elt_idx - degree_begin, degree);

            if constexpr (Policy == TensorAntipodeSigningPolicy::SignByDegree) {
                out[rev_idx + degree_begin] = arg[elt_idx] * (degree % 2 == 0 ? 1 : -1);
            } else {
                out[rev_idx + degree_begin] = arg[elt_idx];
            }
        }
    }

};

}

#endif //RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP
