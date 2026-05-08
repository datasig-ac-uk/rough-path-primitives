#ifndef RPP_CPU_OPS_SINGLE_THREAD_TENSOR_SET_IDENTITY_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_TENSOR_SET_IDENTITY_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/utility.hpp>

#include <rpp/dense/batch.hpp>

#include <rpp/operations/basic/tensor_set_identity.hpp>

#include <rpp/cpu/strategies.hpp>
#include <rpp/cpu/operations/single_thread/detail/batch_wrapper.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class TensorSetIdentity<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

public:
    template <typename Basis>
    static constexpr std::size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template <typename Tensor>
    void operator()(Context const& ctx, Tensor& tensor, Accum scalar = Accum{1}) const noexcept {
        tensor[0] = scalar;
        std::fill(tensor.begin() + 1, tensor.end(), Accum{0});
    }
};

} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchTensor, typename Basis, typename Accum_, typename Architecture>
void tensor_set_identity_kernel(
    const BatchTensor batch_tensor,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ scalar = Accum_{1}
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::TensorSetIdentity<Strategy>;

    detail::apply_batch<Op>(
        basis,
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index tensor_idx) {
            auto tensor = batch_tensor.view(tensor_idx, basis);
            op(ctx, tensor, scalar);
        }
    );
}

} // namespace rpp::cpu::single_thread

#endif // RPP_CPU_OPS_SINGLE_THREAD_TENSOR_SET_IDENTITY_HPP
