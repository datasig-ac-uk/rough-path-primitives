#ifndef RPP_VIEWS_DENSE_GRADED_VECTOR_VIEW_HPP
#define RPP_VIEWS_DENSE_GRADED_VECTOR_VIEW_HPP

#include <tuple>

#include <rpp/architecture.hpp>
#include <rpp/config.h>
#include <rpp/support/iterator_traits.hpp>

#include <rpp/views/batch.hpp>
#include <rpp/views/detail/view_fragment.hpp>

namespace rpp {

template <typename It_, typename Basis_>
class DenseGradedVectorView {
    using Traits = traits::IteratorTraits<It_>;

public:
    using value_type = typename Traits::value_type;
    using reference = typename Traits::reference;
    using iterator = It_;
    using Scalar = value_type;
    using Data = It_;

    using Architecture = traits::arch_of_t<It_>;
    using Basis = Basis_;
    using Degree = typename Architecture::Degree;
    using Index = typename Architecture::Index;

    using Fragment = detail::VectorFragment<It_>;
    using MetaData = std::tuple<Basis const&, Degree, Degree>;

private:
    It_ data_;

    MetaData metadata_;

public:
    RPP_HOST_DEVICE
    constexpr DenseGradedVectorView(It_ data, MetaData meta)
        : data_(data), metadata_(meta) {}

    template <typename BasisMeta,
              typename MinDegreeMeta,
              typename MaxDegreeMeta>
    RPP_HOST_DEVICE constexpr DenseGradedVectorView(
        It_ data,
        std::tuple<BasisMeta, MinDegreeMeta, MaxDegreeMeta> const& meta)
        : data_(data),
          metadata_(std::get<0>(meta),
                    static_cast<Degree>(std::get<1>(meta)),
                    static_cast<Degree>(std::get<2>(meta))) {}

    RPP_HOST_DEVICE
    constexpr DenseGradedVectorView(It_ data, Basis const& basis)
        : data_(data), metadata_(basis, Degree{0}, basis.depth) {}

    RPP_HOST_DEVICE
    constexpr DenseGradedVectorView(It_ data,
                                    Basis const& basis,
                                    Degree min_degree,
                                    Degree max_degree)
        : data_(data), metadata_(basis, min_degree, max_degree) {}

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Basis const&
    basis() const noexcept {
        return std::get<0>(metadata_);
    }
    RPP_HOST_DEVICE RPP_NODISCARD constexpr iterator data() const noexcept {
        return data_;
    }
    RPP_HOST_DEVICE RPP_NODISCARD constexpr MetaData metadata() const noexcept {
        return metadata_;
    }
    RPP_HOST_DEVICE RPP_NODISCARD constexpr Degree min_degree() const noexcept {
        return std::get<1>(metadata_);
    }
    RPP_HOST_DEVICE RPP_NODISCARD constexpr Degree max_degree() const noexcept {
        return std::get<2>(metadata_);
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Index begin_index() const noexcept {
        return basis().start_of_degree(min_degree());
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Index end_index() const noexcept {
        return basis().end_of_degree(max_degree());
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr iterator begin() const noexcept {
        return data() + begin_index();
    }
    RPP_HOST_DEVICE RPP_NODISCARD constexpr iterator end() const noexcept {
        return data() + end_index();
    }


    template <typename I>
    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference operator[](I i) noexcept {
        return data_[i];
    }

    template <typename I>
    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference
    operator[](I i) const noexcept {
        return data_[i];
    }

    RPP_HOST_DEVICE RPP_NODISCARD Index size() const noexcept {
        return basis().degree_begin[max_degree() + 1] -
            basis().degree_begin[min_degree()];
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr bool
    has_degree(Degree degree) const noexcept {
        return min_degree() <= degree && degree <= max_degree();
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Fragment
    degree_view(Degree degree) const noexcept {
        const auto begin = basis().start_of_degree(degree);
        const auto end = basis().end_of_degree(degree);
        return {data() + begin, end - begin};
    }
};


template <typename Data,
          typename Layout,
          typename Basis,
          typename MinDegree,
          typename MaxDegree>
RPP_HOST_DEVICE RPP_NODISCARD constexpr auto
make_graded_vector_batch(Data data,
                         Layout layout,
                         Basis&& basis,
                         MinDegree min_degree,
                         MaxDegree max_degree) noexcept {
    using RealLayout = std::conditional_t<std::is_integral_v<Layout>,
                                          layouts::StrideLayout<Layout>,
                                          Layout>;
    using MetaData = std::tuple<std::decay_t<Basis>, MinDegree, MaxDegree>;
    using BatchType = Batch<DenseGradedVectorView<Data, std::decay_t<Basis>>,
                            RealLayout,
                            MetaData>;


    return BatchType{std::move(data),
                     RealLayout{std::move(layout)},
                     std::make_tuple(std::forward<Basis>(basis),
                                     std::move(min_degree),
                                     std::move(max_degree))};
}

} // namespace rpp

#endif // RPP_VIEWS_DENSE_GRADED_VECTOR_VIEW_HPP
