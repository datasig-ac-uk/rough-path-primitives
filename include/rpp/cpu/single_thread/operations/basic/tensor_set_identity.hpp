#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_TENSOR_SET_IDENTITY_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_TENSOR_SET_IDENTITY_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/basic/tensor_set_identity.hpp>

#include <rpp/cpu/single_thread/strategy.hpp>
namespace rpp::ops {

template <typename Accum_, typename Architecture>
class TensorSetIdentity<
    cpu::strategies::SingleThreadStrategy<Accum_, Architecture>>
    : public BaseOperation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

public:
    static constexpr bool is_implemented = true;

    template <typename Tensor>
    void operator()(Context const& ctx,
                    Tensor& tensor,
                    Accum scalar = Accum{1}) const noexcept {
        ignore_unused(ctx);
        std::fill(tensor.begin(), tensor.end(), Accum{0});
        if (tensor.has_degree(0)) {
            tensor[0] = scalar;
        }
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_TENSOR_SET_IDENTITY_HPP
