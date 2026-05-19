#ifndef RPP_DENSE_LIE_BATCH_HPP
#define RPP_DENSE_LIE_BATCH_HPP

#include <type_traits>
#include <utility>

#include <rpp/config.h>

#include <rpp/basis/basis_tags.hpp>
#include <rpp/basis/lie_basis.hpp>
#include <rpp/dense/detail/basis_tag_traits.hpp>
#include <rpp/dense/lie_view.hpp>
#include <rpp/dense/vector_batch.hpp>

namespace rpp::dense {

template<typename It_, typename Stride, typename MinDegree, typename MaxDegree, typename BasisOrTag=rpp::basis::LieBasisTag>
class LieBatch : public VectorBatch<It_, Stride, MinDegree, MaxDegree, BasisOrTag> {
    using Base = VectorBatch<It_, Stride, MinDegree, MaxDegree, BasisOrTag>;

    static_assert((is_basis_tag_v<BasisOrTag> && std::is_same_v<BasisOrTag, rpp::basis::LieBasisTag>)
                  || detail::has_matching_basis_tag_v<BasisOrTag, rpp::basis::LieBasisTag>,
                  "BasisOrTag must be either a LieBasisTag or a LieBasis (appropriately tagged by LieBasisTag)"
    );

public:
    using Base::Base;

    template<typename Index, typename BasisPack>
    RPP_HOST_DEVICE constexpr auto view(Index index, BasisPack const &basis_pack) const noexcept {
        return this->template view_as<DenseLieView>(index, basis_pack);
    }
};

template<typename It, typename Stride, typename MinDegree, typename MaxDegree, typename BasisOrTag=rpp::basis::LieBasisTag>
RPP_HOST_DEVICE
constexpr auto make_lie_batch(It it, Stride stride, MinDegree min, MaxDegree max, BasisOrTag tag = {}) noexcept {
    return LieBatch<It, Stride, MinDegree, MaxDegree, BasisOrTag>{it, stride, min, max, std::move(tag)};
}

template<typename It, typename Stride, typename MinDegree, typename MaxDegree, typename BasisOrTag, typename Tagger>
constexpr auto tag_batch(LieBatch<It, Stride, MinDegree, MaxDegree, BasisOrTag> const &batch, Tagger tagger RPP_MAYBE_UNUSED) noexcept {
    using NewTag = apply_tagger<Tagger, BasisOrTag>;
    using NewType = LieBatch<It, Stride, MinDegree, MaxDegree, NewTag >;
    return NewType(batch.data(), batch.stride(), batch.min_degree(), batch.max_degree(), NewTag{batch.basis_or_tag()});
}

} // namespace rpp::dense

#endif // RPP_DENSE_LIE_BATCH_HPP
