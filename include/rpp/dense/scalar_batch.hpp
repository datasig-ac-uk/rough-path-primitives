#ifndef RPP_DENSE_SCALAR_BATCH_HPP
#define RPP_DENSE_SCALAR_BATCH_HPP

#include <type_traits>
#include <utility>

#include <rpp/architecture.hpp>
#include <rpp/config.h>
#include <rpp/support/iterator_traits.hpp>

namespace rpp::dense {
namespace detail {

template <typename It>
class ScalarView {
    using value_type = typename traits::IteratorTraits<It>::value_type;
    It data_;

public:
    RPP_HOST_DEVICE
    constexpr ScalarView(It iter) noexcept : data_(std::move(iter)) {}

    template <typename U, typename=std::enable_if_t<std::is_convertible_v<value_type&, U&>>>
    RPP_HOST_DEVICE
    constexpr operator U& () const noexcept {
        return *data_;
    }

    template <typename U>
    RPP_HOST_DEVICE
    constexpr ScalarView& operator=(U&& new_val) noexcept {
        *data_ = std::forward<U>(new_val);
        return *this;
    }
};

} // namespace detail

template <typename It_>
class ScalarBatch {
    using Traits = traits::IteratorTraits<It_>;
    using Architecture = traits::arch_of_t<It_>;
    using View = detail::ScalarView<It_>;
    using Index = typename Architecture::Index;

    It_ data_;

public:
    RPP_HOST_DEVICE
    constexpr explicit ScalarBatch(It_ data) noexcept : data_(data) {}

    RPP_HOST_DEVICE
    constexpr decltype(auto) data() const noexcept {
        return data_;
    }

    template <typename Bases>
    constexpr View view(Index index, Bases const& bases RPP_MAYBE_UNUSED) const noexcept {
        return View(data_ + index);
    }
};

} // namespace rpp::dense

#endif // RPP_DENSE_SCALAR_BATCH_HPP
