#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/tensor_generalised_antipode.hpp>

#include <rpp/gpu/block/strategy.hpp>

namespace rpp::ops {
template <typename Accum_,
          unsigned BlockSize,
          unsigned MaxBlockSize,
          typename Architecture,
          TensorAntipodeSigningPolicy Policy>
class TensorGeneralisedAntipode<
    gpu::strategies::
        BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>,
    Policy,
    void>
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

    template <typename TensorOut, typename TensorArg>
    RPP_DEVICE void operator()(Context const& ctx,
                               TensorOut& out,
                               TensorArg const& arg) const noexcept {
        auto const& basis = out.basis();

        for (auto elt_idx = ctx.thread_rank();
             elt_idx < static_cast<typename Strategy::Index>(arg.size());
             elt_idx += ctx.num_threads()) {
            const auto degree = basis.degree(elt_idx);
            const auto degree_begin = basis.start_of_degree(degree);
            const auto rev_idx =
                basis.reverse_index(elt_idx - degree_begin, degree);

            if constexpr (Policy == TensorAntipodeSigningPolicy::SignByDegree) {
                out[rev_idx + degree_begin] =
                    arg[elt_idx] * (degree % 2 == 0 ? 1 : -1);
            }
            else {
                out[rev_idx + degree_begin] = arg[elt_idx];
            }
        }
    }
};

} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP