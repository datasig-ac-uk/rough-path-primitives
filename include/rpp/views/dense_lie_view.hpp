#ifndef RPP_VIEWS_DENSE_LIE_VIEW_HPP
#define RPP_VIEWS_DENSE_LIE_VIEW_HPP

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <utility>

#include <rpp/basis/lie_basis.hpp>
#include <rpp/config.h>

#include <rpp/views/batch.hpp>
#include <rpp/views/dense_graded_vector_view.hpp>

namespace rpp {

template <typename It, typename Basis>
class DenseLieView : public DenseGradedVectorView<It, Basis> {
    using Base = DenseGradedVectorView<It, Basis>;

public:
    using Base::Base;
    using typename Base::Data;
    using typename Base::Degree;
    using typename Base::MetaData;

    RPP_HOST_DEVICE RPP_NODISCARD constexpr DenseLieView
    truncate(Degree min_degree, Degree max_degree) const noexcept {
        return {this->data(),
                this->basis(),
                std::max(min_degree, this->min_degree()),
                std::min(max_degree, this->max_degree())};
    }
};


template <typename Data,
          typename Layout,
          typename BasisOrTag,
          typename MinDegree,
          typename MaxDegree>
constexpr auto make_lie_batch(Data&& data,
                              Layout&& layout,
                              BasisOrTag&& basis_or_tag,
                              MinDegree min_degree,
                              MaxDegree max_degree) noexcept {
    using StoredData = std::decay_t<Data>;
    using StoredLayout = std::decay_t<Layout>;
    using RealLayout = std::conditional_t<std::is_integral_v<StoredLayout>,
                                          layouts::StrideLayout<StoredLayout>,
                                          StoredLayout>;
    using StoredBasisOrTag = std::decay_t<BasisOrTag>;
    using MetaData = std::tuple<StoredBasisOrTag, MinDegree, MaxDegree>;
    using Basis =
        std::conditional_t<basis::is_basis_tag_v<StoredBasisOrTag>,
                           basis::LieBasis<traits::arch_of_t<StoredData>>,
                           StoredBasisOrTag>;
    using BatchType =
        Batch<DenseLieView<StoredData, Basis>, RealLayout, MetaData>;

    return BatchType{std::forward<Data>(data),
                     RealLayout{std::forward<Layout>(layout)},
                     std::make_tuple(std::forward<BasisOrTag>(basis_or_tag),
                                     min_degree,
                                     max_degree)};
}

template <typename Data,
          typename Layout,
          typename MinDegree,
          typename MaxDegree>
constexpr auto make_lie_batch(Data&& data,
                              Layout&& layout,
                              MinDegree min_degree,
                              MaxDegree max_degree) noexcept {
    using StoredData = std::decay_t<Data>;
    using StoredLayout = std::decay_t<Layout>;
    using RealLayout = std::conditional_t<std::is_integral_v<StoredLayout>,
                                          layouts::StrideLayout<StoredLayout>,
                                          StoredLayout>;
    using MetaData = std::tuple<basis::LieBasisTag, MinDegree, MaxDegree>;
    using Basis = basis::LieBasis<traits::arch_of_t<StoredData>>;
    using BatchType =
        Batch<DenseLieView<StoredData, Basis>, RealLayout, MetaData>;

    return BatchType{
        std::forward<Data>(data),
        RealLayout{std::forward<Layout>(layout)},
        std::make_tuple(basis::LieBasisTag{}, min_degree, max_degree)};
}
} // namespace rpp

#endif // RPP_VIEWS_DENSE_LIE_VIEW_HPP
