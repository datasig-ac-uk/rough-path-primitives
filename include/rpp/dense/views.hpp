#ifndef RPP_DENSE_VIEWS_HPP
#define RPP_DENSE_VIEWS_HPP

#include <cstddef>
#include <iterator>

#include <rpp/config.h>

#include <rpp/basis.hpp>

namespace rpp::dense {
namespace detail {
template<typename It_>
class VectorFragment {
    using Traits = std::iterator_traits<It_>;
    using Index = typename Traits::difference_type;

    It_ data_;
    Index size_;

public:
    using value_type = typename Traits::value_type;
    using reference = typename Traits::reference;

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

template<typename It_, typename Basis_>
class DenseVectorView {
    using Traits = std::iterator_traits<It_>;

public:
    using value_type = typename Traits::value_type;
    using reference = typename Traits::reference;
    using iterator = It_;
    using Scalar = value_type;

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
    constexpr DenseVectorView(It_ data, Basis const &basis)
        : data_(data), min_degree_(0), max_degree_(basis.depth), basis_(basis) {
    }

    RPP_HOST_DEVICE
    constexpr DenseVectorView(It_ data, Basis const &basis, Degree min_deg, Degree max_degree)
        : data_(data), min_degree_(min_deg), max_degree_(max_degree), basis_(basis) {
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Basis const &basis() const noexcept { return basis_; }
    RPP_HOST_DEVICE RPP_NODISCARD constexpr iterator data() const noexcept { return data_; }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr iterator begin() const noexcept {
        return data_ + basis_.start_of_degree(min_degree_);
    }
    RPP_HOST_DEVICE RPP_NODISCARD constexpr iterator end() const noexcept {
        return data_ + basis_.end_of_degree(max_degree_);
    }


    RPP_HOST_DEVICE RPP_NODISCARD constexpr Degree min_degree() const noexcept { return min_degree_; }
    RPP_HOST_DEVICE RPP_NODISCARD constexpr Degree max_degree() const noexcept { return max_degree_; }

    template<typename Index_>
    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference operator[](Index_ i) noexcept {
        return *(data_ + i);
    }

    template<typename Index_>
    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference operator[](Index_ i) const noexcept {
        return *(data_ + i);
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


template<typename It_, typename Basis_>
class DenseTensorView : public DenseVectorView<It_, Basis_> {
    using Base = DenseVectorView<It_, Basis_>;

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
} //namespace rpp::dense

#endif // RPP_DENSE_VIEWS_HPP
