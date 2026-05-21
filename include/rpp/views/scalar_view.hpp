#ifndef RPP_VIEWS_SCALAR_VIEW_HPP
#define RPP_VIEWS_SCALAR_VIEW_HPP

#include <rpp/config.h>

#include <rpp/architecture.hpp>
#include <rpp/support/iterator_traits.hpp>
#include <rpp/views/batch.hpp>


namespace rpp {

template <typename It>
class ScalarView {
    using Traits = traits::IteratorTraits<It>;
    It data_;

public:
    using Architecture = traits::arch_of_t<It>;
    using Index = typename Architecture::Index;
    using Data = It;
    using MetaData = std::tuple<>;


    RPP_HOST_DEVICE
    constexpr explicit ScalarView(Data data) : data_(std::move(data)) {}

    RPP_HOST_DEVICE
    constexpr explicit ScalarView(Data data, MetaData metadata RPP_MAYBE_UNUSED)
        : data_(std::move(data)) {}

    template <typename U,
              typename = std::enable_if_t<
                  std::is_convertible_v<typename Traits::reference, U&>>>
    RPP_HOST_DEVICE constexpr operator U&() const noexcept {
        return *data_;
    }

    template <typename U>
    RPP_HOST_DEVICE constexpr ScalarView&
    operator=(U&& val) noexcept(noexcept(*data_ = std::forward<U>(val))) {
        *data_ = std::forward<U>(val);
        return *this;
    }
};


template <typename Data>
constexpr auto make_scalar_batch(Data&& data) noexcept {
    using StoredData = std::decay_t<Data>;
    using BatchType =
        Batch<ScalarView<StoredData>, layouts::NoStrideLayout, std::tuple<>>;
    return BatchType{
        std::forward<Data>(data), layouts::NoStrideLayout{}, std::make_tuple()};
}


} // namespace rpp

#endif // RPP_VIEWS_SCALAR_VIEW_HPP
