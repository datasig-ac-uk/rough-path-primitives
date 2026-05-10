#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_PAIRING_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_PAIRING_HPP

#include <algorithm>
#include <functional>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/tensor_pairing.hpp>

#include <rpp/gpu/operations/block/strategy.hpp>

namespace rpp::ops {
template<typename Accum, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class TensorPairing<gpu::strategies::BlockStrategy<Accum, BlockSize, MaxBlockSize, Architecture> > : public
        BaseOperation<gpu::strategies::BlockStrategy<Accum, BlockSize, MaxBlockSize, Architecture> > {
    using Strategy = gpu::strategies::BlockStrategy<Accum, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;

public:
    template<typename Basis>
    static constexpr size_t scratch_space_size(Strategy const &strategy, Basis const &basis) noexcept {
        ignore_unused(strategy, basis);
        return sizeof(typename Strategy::BlockReduceArray);
    }

    template<typename Scalar, typename TensorFunc, typename TensorArg>
    RPP_DEVICE void operator()(Context const &ctx, Scalar &out, const TensorFunc &functional,
                               const TensorArg &arg) noexcept {
        auto const &basis = functional.basis();
        const auto min_deg = std::max(functional.min_degree(), arg.min_degree());
        const auto max_deg = std::min(functional.max_degree(), arg.max_degree());

        const auto begin = basis.start_of_degree(min_deg);
        const auto end = basis.end_of_degree(max_deg);

        Accum val{0};
        for (auto i = begin + ctx.thread_rank(); i < end; i += ctx.num_threads()) {
            const Accum func_val{functional[i]};
            const Accum arg_val{arg[i]};
            val += func_val * arg_val;
        }

        Accum result = ctx.reduce(val, std::plus<Accum>{});

        if (ctx.thread_rank() == 0) {
            out = result;
        }
    }
};
} // namespace rpp::ops

#endif //INCLUDE_RPP_GPU_OPERATIONS_BLOCK_TENSOR_PAIRING_HPP
