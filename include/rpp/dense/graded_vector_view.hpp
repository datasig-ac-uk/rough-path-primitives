#ifndef RPP_DENSE_GRADED_VECTOR_VIEW_HPP
#define RPP_DENSE_GRADED_VECTOR_VIEW_HPP

#include <rpp/config.h>
#include <rpp/architecture.hpp>
#include <rpp/support/iterator_traits.hpp>

#include <rpp/dense/detail/view_fragment.hpp>

namespace rpp::dense {

template<typename It_, typename Basis_>
class DenseGradedVectorView {
    using Traits = traits::IteratorTraits<It_>;

public:
    using value_type = typename Traits::value_type;
    using reference = typename Traits::reference;
    using iterator = It_;
    using Scalar = value_type;

    using Architecture = traits::arch_of_t<It_>;
    using Index = typename Traits::difference_type;
    using Basis = Basis_;
    using Degree = typename Basis::Degree;
    using Index_ = typename Basis::Index;

    using Fragment = detail::VectorFragment<It_>;

private:
    It_ data_;
    Degree min_degree_;
    Degree max_degree_;
    Basis const &basis_;

public:
    RPP_HOST_DEVICE
    constexpr DenseGradedVectorView(It_ data, Basis const &basis)
        : data_(data), min_degree_(0), max_degree_(basis.depth), basis_(basis) {
    }

    RPP_HOST_DEVICE
    constexpr DenseGradedVectorView(It_ data, Basis const &basis, Degree min_deg, Degree max_degree)
        : data_(data), min_degree_(min_deg), max_degree_(max_degree), basis_(basis) {
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Basis const &basis() const noexcept { return basis_; }
    RPP_HOST_DEVICE RPP_NODISCARD constexpr iterator data() const noexcept { return data_; }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Index begin_index() const noexcept {
        return basis_.start_of_degree(min_degree_);
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Index end_index() const noexcept {
        return basis_.end_of_degree(max_degree_);
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr iterator begin() const noexcept {
        return data_ + begin_index();
    }
    RPP_HOST_DEVICE RPP_NODISCARD constexpr iterator end() const noexcept {
        return data_ + end_index();
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Degree min_degree() const noexcept { return min_degree_; }
    RPP_HOST_DEVICE RPP_NODISCARD constexpr Degree max_degree() const noexcept { return max_degree_; }

    template<typename Index__>
    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference operator[](Index__ i) noexcept {
        return data_[i];
    }

    template<typename Index__>
    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference operator[](Index__ i) const noexcept {
        return data_[i];
    }

    RPP_HOST_DEVICE RPP_NODISCARD Index size() const noexcept {
        return basis_.degree_begin[max_degree_ + 1] - basis_.degree_begin[min_degree_];
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr bool has_degree(Degree degree) const noexcept {
        return min_degree_ <= degree && degree <= max_degree_;
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Fragment degree_view(Degree degree) const noexcept {
        const auto begin = basis_.start_of_degree(degree);
        const auto end = basis_.end_of_degree(degree);
        return { data() + begin, end - begin };
    }
};

} // namespace rpp::dense

#endif // RPP_DENSE_GRADED_VECTOR_VIEW_HPP
