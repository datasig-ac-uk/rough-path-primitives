#ifndef RPP_SPARSE_T2L_MATRIX_HPP
#define RPP_SPARSE_T2L_MATRIX_HPP

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rpp/sparse/detail/compressed_matrix.hpp>
#include <rpp/sparse/matrix.hpp>

namespace rpp::sparse {

namespace t2l_detail {


template <typename FrameBuilder,
          typename Builder,
          typename LieBasis,
          typename MultiplicationCache>
void compute_rbracket(FrameBuilder& frame_builder,
                      Builder const& builder,
                      LieBasis const& basis,
                      MultiplicationCache& cache,
                      typename FrameBuilder::Index lparent,
                      typename FrameBuilder::Index rparent) {
    using Index = typename FrameBuilder::Index;
    using Scalar = typename FrameBuilder::Scalar;

    using LieWord = typename MultiplicationCache::BracketIndex;

    if (lparent == rparent) {
        return;
    }

    const auto lframe = builder[lparent];
    const auto rframe = builder[rparent];


    LieWord word{lparent, 0};

    for (Index j = 0; j < rframe.size; ++j) {
        word.right = 1 + rframe.index[j];

        for (auto&& term : cache.get_bracket(word)) {
            auto index = basis.key_to_idx(term.first);

            frame_builder[index] += static_cast<Scalar>(term.second) *
                lframe.data[0] * rframe.data[j];
        }
    }
}


} // namespace t2l_detail

template <typename Scalar, typename LieBasis, typename MultiplicationCache>
RPP_NODISCARD sparse::GradedMatrixOwned<CSCMatrix,
                                        std::vector<Scalar>,
                                        std::vector<typename LieBasis::Index>,
                                        std::vector<typename LieBasis::Index>>
make_t2l_matrix(LieBasis const& basis, MultiplicationCache& cache) {
    using Index = typename LieBasis::Index;
    using Degree = typename LieBasis::Degree;
    using Offset = Index;
    using difference_type = std::ptrdiff_t;
    using DataContainer = std::vector<Scalar>;
    using IndexContainer = std::vector<Index>;
    using OffsetContainer = std::vector<Offset>;

    using Builder =
        CompressedMatrixBuilder<DataContainer, IndexContainer, OffsetContainer>;
    using FrameBuilder =
        CompressedMatrixFrameBuilder<DataContainer, IndexContainer>;

    std::vector<Scalar> data;
    std::vector<Index> indices;
    std::vector<Offset> offsets;

    Builder builder{data, indices, offsets};

    FrameBuilder frame_builder;
    frame_builder.reserve(basis.depth + 1);

    // empty frame for the empty word
    builder.append_frame(frame_builder);

    if (basis.depth > 0) {
        frame_builder.append(0, Scalar{1});
        for (Index i = 0; i < basis.width; ++i) {
            frame_builder.set_index(0, i);
            builder.append_frame(frame_builder);
        }
    }

    Index offset = 1;
    Index prev_size = basis.width;
    for (Degree degree = 2; degree <= basis.depth; ++degree) {
        const auto deg_size = prev_size * basis.width;
        const auto deg_offset = 1 + offset * basis.width;

        for (Index i = 0; i < deg_size; ++i) {
            frame_builder.clear();

            const auto lparent = 1 + i / prev_size;
            const auto rparent = offset + i % prev_size;

            t2l_detail::compute_rbracket(
                frame_builder, builder, basis, cache, lparent, rparent);

            frame_builder.pop_zeros();
            builder.append_frame(frame_builder);
        }

        offset = deg_offset;
        prev_size = deg_size;
    }

    // normalise by degree
    offset = 1 + basis.width;
    for (Degree degree=2; degree<=basis.depth; ++degree) {
        const auto end_offset = 1 + offset * basis.width;

        const auto divisor = Scalar{1} / static_cast<Scalar>(degree);
        auto [it, end] = builder.data_range(offset, end_offset);
        for (; it!=end; ++it) {
            *it *= divisor;
        }

        offset = end_offset;
    }


    const auto n_offsets = static_cast<difference_type>(offsets.size() - 1);
    const auto nnz = static_cast<difference_type>(data.size());
    return {std::move(data),
            std::move(indices),
            std::move(offsets),
            nnz,
            n_offsets,
            basis.size()};
}

} // namespace rpp::sparse

#endif // RPP_SPARSE_T2L_MATRIX_HPP
