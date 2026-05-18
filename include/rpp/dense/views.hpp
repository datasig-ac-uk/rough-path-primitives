#ifndef RPP_DENSE_VIEWS_HPP
#define RPP_DENSE_VIEWS_HPP

#include <algorithm>
#include <cstddef>
#include <iterator>

#include <rpp/config.h>
#include <rpp/architecture.hpp>
#include <rpp/support/iterator_traits.hpp>

namespace rpp::dense {
namespace detail {

/**
 * @class VectorFragment
 * @brief Represents a fragment or subsection of a vector with specified boundaries.
 *
 * This class is designed to operate on a subset of a vector, allowing
 * operations to be performed on a contiguous range within the vector without
 * directly modifying the original vector. It maintains information about
 * the starting index and length of the fragment.
 *
 */
template<typename It_>
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
    constexpr VectorFragment(It_ data, Index size)
        : data_(data), size_(size) {
    }

    RPP_HOST_DEVICE
    constexpr Index size() const noexcept {
        return size_;
    }

    RPP_HOST_DEVICE
    constexpr reference operator[](Index i) const noexcept {
        return data_[i];
    }
};
} // namespace detail

/**
 * @brief Read‑write view into a dense graded vector limited to a specified segment.
 *
 * This class provides a lightweight façade over a dense graded vector,
 * exposing only a contiguous sub‑range defined by a start index and length.
 * It enables operations that are logically scoped to that segment (e.g.,
 * reductions, element‑wise updates) without copying the underlying data.
 * All indexed accesses are translated to the corresponding location in the
 * parent container, ensuring that modifications affect the original vector.
 * The view does not own its storage; it merely delegates all queries to the
 * owning container, preserving memory layout and performance.
 */
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

    template<typename Index_>
    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference operator[](Index_ i) noexcept {
        return data_[i];
    }

    template<typename Index_>
    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference operator[](Index_ i) const noexcept {
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


/**
 * @brief Read‑write view into a dense graded tensor limited to a contiguous
 *        degree sub‑range.
 *
 * @tparam It_   Iterator type used by the underlying dense graded vector.
 * @tparam Basis_ Type that defines the grading basis (e.g., degree space).
 *
 * This class derives from @c DenseGradedVectorView and does not own any
 * storage; it merely provides a façade over a segment of a parent container.
 * The view enables operations that are logically scoped to a degree interval
 * without copying data, translating all indexed accesses to the corresponding
 * locations in the parent container.
 *
 * The primary operation offered by this class is @ref truncate, which
 * constructs a new view restricted to the intersection of the requested degree
 * range with the degrees actually present in the underlying vector.
 *
 * @param min_degree  Lower bound of the desired degree interval.
 * @param max_degree  Upper bound of the desired degree interval.
 *
 * @return A new @c DenseTensorView representing the intersection of the
 *         requested degree range with the existing degree range of the
 *         underlying container.
 *
 * @note The truncation operation is constexpr and noexcept, allowing it to be
 *       evaluated at compile time when possible.
 */
template<typename It_, typename Basis_>
class DenseTensorView : public DenseGradedVectorView<It_, Basis_> {
    using Base = DenseGradedVectorView<It_, Basis_>;

public:
    using Base::Base;
    using typename Base::Degree;

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr DenseTensorView truncate(Degree min_degree, Degree max_degree) const noexcept {
        return {
            this->data(),
            this->basis(),
            std::max(min_degree, this->min_degree()),
            std::min(max_degree, this->max_degree())
        };
    }
};


/**
 * @brief A read‑write view specialized for Lie algebras that provides
 *        a limited perspective on a dense graded vector.
 *
 * This class extends @c DenseGradedVectorView to operate only on a
 * contiguous segment of degrees defined by the user. It does not own
 * storage; instead, it references the underlying container and translates
 * all operations to the corresponding positions within that container.
 * The view supports truncation to a requested degree interval, automatically
 * intersecting it with the existing degree range of the parent object.
 * Truncation is performed constexpr and noexcept, enabling compile‑time
 * evaluation when possible.
 *
 * The view facilitates scoped computations on Lie‑type structures while
 * preserving the original memory layout and enabling modifications to
 * propagate to the underlying data.
 */
template <typename It, typename Basis>
class DenseLieView : public DenseGradedVectorView<It, Basis> {
    using Base = DenseGradedVectorView<It, Basis>;

public:
    using Base::Base;
    using typename Base::Degree;


    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr DenseLieView truncate(Degree min_degree, Degree max_degree) const noexcept {
        return {
            this->data(),
            this->basis(),
            std::max(min_degree, this->min_degree()),
            std::min(max_degree, this->max_degree())
        };
    }
};


} //namespace rpp::dense

#endif // RPP_DENSE_VIEWS_HPP
