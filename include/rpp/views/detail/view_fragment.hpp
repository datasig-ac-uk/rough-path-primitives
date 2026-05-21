#ifndef RPP_VIEWS_DETAIL_VIEW_FRAGMENT_HPP
#define RPP_VIEWS_DETAIL_VIEW_FRAGMENT_HPP

#include <rpp/architecture.hpp>
#include <rpp/config.h>
#include <rpp/support/iterator_traits.hpp>

namespace rpp::detail {

template <typename It_>
class VectorFragment {
    using Traits = traits::IteratorTraits<It_>;
    using Index = typename Traits::difference_type;

    It_ data_;
    Index size_;

public:
    using value_type = typename Traits::value_type;
    using reference = typename Traits::reference;
    using Architecture = traits::arch_of_t<It_>;

    RPP_HOST_DEVICE
    constexpr VectorFragment(It_ data, Index size) : data_(data), size_(size) {}

    RPP_HOST_DEVICE
    constexpr Index size() const noexcept { return size_; }

    RPP_HOST_DEVICE
    constexpr reference operator[](Index i) const noexcept { return data_[i]; }
};

} // namespace rpp::detail

#endif // RPP_VIEWS_DETAIL_VIEW_FRAGMENT_HPP
