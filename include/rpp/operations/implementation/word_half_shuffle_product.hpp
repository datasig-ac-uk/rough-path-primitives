

#ifndef RPP_OPERATIONS_IMPLEMENTATION_WORD_HALF_SHUFFLE_PRODUCT_HPP
#define RPP_OPERATIONS_IMPLEMENTATION_WORD_HALF_SHUFFLE_PRODUCT_HPP

// IWYU pragma: always_keep


#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/implementation/word_shuffle_product.hpp>

namespace rpp::ops::common {

namespace detail {

template <typename Tensor>
struct PrefixLetterGetter {
    using Index = typename Tensor::Index;
    using Degree = typename Tensor::Degree;
    Index letter;

    RPP_HOST_DEVICE RPP_FORCEINLINE bool
    has_degree(Tensor const& instance, Degree degree) const noexcept {
        ignore_unused(this);
        return instance.has_degree(degree + 1);
    }

    RPP_HOST_DEVICE RPP_FORCEINLINE decltype(auto) operator()(
        Tensor const& instance, Degree degree, Index index) const noexcept {
        const auto stride = instance.basis().size_of_degree(degree);
        return instance.degree_view(degree + 1)[letter * stride + index];
    }
};

} // namespace detail


template <typename Context, typename TensorIntegrator, typename TensorIntegrand>
RPP_HOST_DEVICE RPP_FORCEINLINE typename Context::Accum
word_left_half_shuffle_product(Context const& ctx,
                               typename Context::Index prefix_letter,
                               typename Context::Index trailing_index,
                               typename Context::Degree trailing_degree,
                               TensorIntegrator const& integrator,
                               TensorIntegrand const& integrand) noexcept {

    detail::PrefixLetterGetter<TensorIntegrator> integrator_getter{
        prefix_letter};
    detail::DefaultGetter<TensorIntegrand> integrand_getter{};

    return word_shuffle_product(ctx,
                                trailing_index,
                                trailing_degree,
                                integrator,
                                integrand,
                                std::move(integrator_getter),
                                std::move(integrand_getter));
}


} // namespace rpp::ops::common

#endif // RPP_OPERATIONS_IMPLEMENTATION_WORD_HALF_SHUFFLE_PRODUCT_HPP
