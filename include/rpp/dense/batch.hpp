#ifndef RPP_DENSE_BATCH_HPP
#define RPP_DENSE_BATCH_HPP

#include <tuple>
#include <type_traits>
#include <utility>

#include <rpp/config.h>
#include <rpp/architecture.hpp>

#include <rpp/basis/basis_pack.hpp>
#include <rpp/basis/basis_tags.hpp>
#include <rpp/basis/tensor_basis.hpp>
#include <rpp/basis/lie_basis.hpp>
#include <rpp/dense/views.hpp>

namespace rpp::dense {
namespace detail {
template<typename T, typename Tag, typename=void>
inline constexpr bool has_matching_basis_tag_v = false;

template<typename T, typename Tag>
inline constexpr bool has_matching_basis_tag_v<T, Tag, std::void_t<typename T::Tag> >
        = std::is_base_of_v<typename T::Tag, Tag>;
} // namespace detail

template<typename It_, typename Stride, typename MinDegree, typename MaxDegree, typename TagOrBasis>
class VectorBatch {
    std::tuple<It_, Stride, MinDegree, MaxDegree, TagOrBasis> data_;

public:
    RPP_HOST_DEVICE
    constexpr VectorBatch(It_ it, Stride stride, MinDegree min, MaxDegree max, TagOrBasis basis_or_tag) noexcept
        : data_(it, stride, min, max, std::move(basis_or_tag)) {
    }


    RPP_HOST_DEVICE constexpr decltype(auto) data() const noexcept {
        return std::get<0>(data_);
    }

    RPP_HOST_DEVICE constexpr decltype(auto) stride() const noexcept {
        return std::get<1>(data_);
    }

    RPP_HOST_DEVICE constexpr decltype(auto) min_degree() const noexcept {
        return std::get<2>(data_);
    }

    RPP_HOST_DEVICE constexpr decltype(auto) max_degree() const noexcept {
        return std::get<3>(data_);
    }

    RPP_HOST_DEVICE constexpr decltype(auto) basis_or_tag() const noexcept {
        return std::get<4>(data_);
    }

    template<template<typename, typename> class ViewT, typename Index, typename BasisPack>
    RPP_HOST_DEVICE constexpr auto view_as(Index index, BasisPack const &basis_pack) const noexcept {
        auto const &basis = get_basis(std::get<4>(data_), basis_pack);
        using Basis = std::remove_cv_t<std::remove_reference_t<decltype(basis)> >;
        using View = ViewT<It_, Basis>;
        using Degree = typename View::Degree;
        return View(data() + index * stride(), basis, static_cast<Degree>(min_degree()),
                    static_cast<Degree>(max_degree()));
    }

    template<template<typename, typename> class ViewT, typename Index, typename BasisPack>
    RPP_HOST_DEVICE constexpr auto view(Index index, BasisPack const &basis_pack) const noexcept {
        return view_as<ViewT>(index, basis_pack);
    }

    template<typename Index, typename BasisPack>
    RPP_HOST_DEVICE constexpr auto view(Index index, BasisPack const &basis_pack) const noexcept {
        return view_as<DenseGradedVectorView>(index, basis_pack);
    }
};

template<typename It, typename Stride, typename MinDegree, typename MaxDegree, typename BasisOrTag>
RPP_HOST_DEVICE
constexpr auto make_vector_batch(It it, Stride stride, MinDegree min, MaxDegree max, BasisOrTag basis_or_tag) noexcept {
    return VectorBatch<It, Stride, MinDegree, MaxDegree, BasisOrTag>{it, stride, min, max, std::move(basis_or_tag)};
}

template<typename It_, typename Stride, typename MinDegree, typename MaxDegree, typename BasisOrTag=TensorBasisTag>
class TensorBatch : public VectorBatch<It_, Stride, MinDegree, MaxDegree, BasisOrTag> {
    using Base = VectorBatch<It_, Stride, MinDegree, MaxDegree, BasisOrTag>;

    static_assert((is_basis_tag_v<BasisOrTag> && std::is_same_v<BasisOrTag, TensorBasisTag>)
                  || detail::has_matching_basis_tag_v<BasisOrTag, TensorBasisTag>,
                  "BasisOrTag must be either a TensorBasisTag or a TensorBasis (appropriately tagged by TensorBasisTag)"
    );

public:
    using Base::Base;

    template<typename Index, typename BasisPack>
    RPP_HOST_DEVICE constexpr auto view(Index index, BasisPack const &basis_pack) const noexcept {
        return this->template view_as<DenseTensorView>(index, basis_pack);
    }
};


template<typename It, typename Stride, typename MinDegree, typename MaxDegree, typename BasisOrTag=TensorBasisTag>
RPP_HOST_DEVICE
constexpr auto make_tensor_batch(It it, Stride stride, MinDegree min, MaxDegree max, BasisOrTag tag = {}) noexcept {
    return TensorBatch<It, Stride, MinDegree, MaxDegree, BasisOrTag>{it, stride, min, max, std::move(tag)};
}


template<typename It_, typename Stride, typename MinDegree, typename MaxDegree, typename BasisOrTag=LieBasisTag>
class LieBatch : public VectorBatch<It_, Stride, MinDegree, MaxDegree, BasisOrTag> {
    using Base = VectorBatch<It_, Stride, MinDegree, MaxDegree, BasisOrTag>;

    static_assert((is_basis_tag_v<BasisOrTag> && std::is_same_v<BasisOrTag, LieBasisTag>)
                  || detail::has_matching_basis_tag_v<BasisOrTag, LieBasisTag>,
                  "BasisOrTag must be either a LieBasisTag or a LieBasis (appropriately tagged by LieBasisTag)"
    );

public:
    using Base::Base;

    template<typename Index, typename BasisPack>
    RPP_HOST_DEVICE constexpr auto view(Index index, BasisPack const &basis_pack) const noexcept {
        return this->template view_as<DenseLieView>(index, basis_pack);
    }
};

template<typename It, typename Stride, typename MinDegree, typename MaxDegree, typename BasisOrTag=LieBasisTag>
RPP_HOST_DEVICE
constexpr auto make_lie_batch(It it, Stride stride, MinDegree min, MaxDegree max, BasisOrTag tag = {}) noexcept {
    return LieBatch<It, Stride, MinDegree, MaxDegree, BasisOrTag>{it, stride, min, max, std::move(tag)};
}

template<typename It, typename Stride, typename MinDegree, typename MaxDegree, typename BasisOrTag, typename Tagger>
constexpr auto tag_batch(VectorBatch<It, Stride, MinDegree, MaxDegree, BasisOrTag> const &batch, Tagger tagger RPP_MAYBE_UNUSED) noexcept {
    using NewTag = apply_tagger<Tagger, BasisOrTag>;
    using NewType = VectorBatch<It, Stride, MinDegree, MaxDegree, NewTag>;
    return NewType(batch.data(), batch.stride(), batch.min_degree(), batch.max_degree(), NewTag{batch.basis_or_tag()});
}

template< typename It, typename Stride, typename MinDegree, typename MaxDegree, typename BasisOrTag, typename Tagger>
constexpr auto tag_batch(TensorBatch<It, Stride, MinDegree, MaxDegree, BasisOrTag> const &batch, Tagger tagger RPP_MAYBE_UNUSED) noexcept {
    using NewTag = apply_tagger<Tagger, BasisOrTag>;
    using NewType = TensorBatch<It, Stride, MinDegree, MaxDegree, NewTag >;
    return NewType(batch.data(), batch.stride(), batch.min_degree(), batch.max_degree(), NewTag{batch.basis_or_tag()});
}

template<typename It, typename Stride, typename MinDegree, typename MaxDegree, typename BasisOrTag, typename Tagger>
constexpr auto tag_batch(LieBatch<It, Stride, MinDegree, MaxDegree, BasisOrTag> const &batch, Tagger tagger RPP_MAYBE_UNUSED) noexcept {
    using NewTag = apply_tagger<Tagger, BasisOrTag>;
    using NewType = LieBatch<It, Stride, MinDegree, MaxDegree, NewTag >;
    return NewType(batch.data(), batch.stride(), batch.min_degree(), batch.max_degree(), NewTag{batch.basis_or_tag()});
}

} // namespace rpp::dense


#endif //RPP_DENSE_BATCH_HPP
