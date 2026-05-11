#ifndef RPP_BASIS_BASIS_TAGS_HPP
#define RPP_BASIS_BASIS_TAGS_HPP

#include <cstddef>

namespace rpp {
template<typename T>
inline constexpr bool is_basis_tag_v = false;


template <typename Tag>
struct InputBasisTag : Tag {
    static_assert(is_basis_tag_v<Tag>,
        "only valid tags can be annotated"
        );
};
template <typename Tag>
inline constexpr bool is_basis_tag_v<InputBasisTag<Tag>> = true;

template <typename Tag>
struct OutputBasisTag : Tag {
    static_assert(is_basis_tag_v<Tag>,
        "only valid tags can be annotated"
    );
};
template <typename Tag>
inline constexpr bool is_basis_tag_v<OutputBasisTag<Tag>> = true;

template <size_t Index, typename Tag>
struct IndexedBasisTag : Tag {
    static constexpr size_t index = Index;
    static_assert(is_basis_tag_v<Tag>,
        "only valid tags can be annotated"
        );
};

template <size_t Index, typename Tag>
inline constexpr bool is_basis_tag_v<IndexedBasisTag<Index, Tag>> = true;



} // namespace rpp


#define RPP_MAKE_BASIS_TAG(basis_name) \
    struct basis_name ## Tag {}; \
    template <> \
    inline constexpr bool is_basis_tag_v<basis_name ## Tag> = true; \


#endif //RPP_BASIS_BASIS_TAGS_HPP
