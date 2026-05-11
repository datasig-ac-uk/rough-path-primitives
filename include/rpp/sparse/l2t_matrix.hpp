#ifndef RPP_BASIS_L2T_MATRIX_HPP
#define RPP_BASIS_L2T_MATRIX_HPP

#include <vector>

#include <rpp/config.h>
#include <rpp/basis/lie_basis.hpp>
#include <rpp/basis/tensor_basis.hpp>
#include <rpp/sparse/matrix.hpp>



namespace rpp::sparse {


namespace l2t_detail {

template <typename Builder, typename Basis>
void insert_commutator(Builder& builder, Basis const& basis, typename Basis::Index left_key, typename Basis::Index right_key) {
    using Index = typename Basis::Index;

    const auto left_idx = basis.key_to_idx(left_key);
    const auto right_idx = basis.key_to_idx(right_key);
    auto& left_frame = builder[left_idx];
    auto& right_frame = builder[right_idx];

    const auto left_degree = basis.degree(left_idx);
    const auto right_degree = basis.degree(right_idx);

    const auto left_offset = tensor_degree_size<Index>(basis.width, left_degree);
    const auto right_offset = tensor_degree_size<Index>(basis.width, right_degree);

    for (Index i=0; i<left_frame.size; ++i) {
        for (Index j=0; j<right_frame.size; ++j) {
            builder.get_scalar(left_frame.index[i] * right_offset + right_frame.index[j])
                += left_frame.data[i] * right_frame.data[j];

            builder.get_scalar(right_frame.index[i] * left_offset + left_frame.index[j])
                -= right_frame.data[i] * left_frame.data[j];
        }
    }
}


}

template <typename Scalar, typename LieBasis>
RPP_NODISCARD
sparse::GradedMatrixOwned<CSCMatrix, std::vector<Scalar>, std::vector<typename LieBasis::Index>, std::vector<typename LieBasis::Index>>
make_l2t_matrix(LieBasis const& basis) {
    using Index = typename LieBasis::Index;
    using Degree = typename LieBasis::Degree;
    using Offset = Index;
    using difference_type = std::ptrdiff_t;

    std::vector<Scalar> data;
    std::vector<Index> indices;
    std::vector<Offset> offsets;

    CompressedMatrixBuilder<std::vector<Scalar>, std::vector<Index>, std::vector<Offset>> builder { data, indices, offsets };

    for (Degree letter=1; letter<=basis.width; ++letter) {
        builder.next_frame();
        builder.get_scalar(letter) = Scalar{1};
    }

    for (Degree d=2; d<=basis.depth; ++d) {
        const auto begin = basis.start_of_degree(d);
        const auto end = basis.end_of_degree(d);

        for (auto key=begin; key<end; ++key) {
            auto& [left, right] = basis[key];
            l2t_detail::insert_commutator(builder, basis, left, right);
        }
    }

    const auto nnz = static_cast<difference_type>(data.size());
    return {std::move(data), std::move(indices), std::move(offsets), nnz, basis.size(), tensor_dimension<Index>(basis.width, basis.depth) };
}


} // namespace rpp::sparse



#endif //RPP_BASIS_L2T_MATRIX_HPP
