#ifndef RPP_SPARSE_T2L_MATRIX_HPP
#define RPP_SPARSE_T2L_MATRIX_HPP

#include <vector>
#include <unordered_map>

#include <rpp/sparse/compressed_matrix.hpp>

namespace rpp::sparse {


template <typename Scalar, typename LieBasis, typename MultiplicationCache>
RPP_NODISCARD
sparse::OwnedCompressedMatrix<std::vector<Scalar>, std::vector<typename LieBasis::Index>, std::vector<typename LieBasis::Index>, CompressedFormat::CSC>
make_t2l_matrix(LieBasis const& basis, MultiplicationCache& cache) {
    using Index = typename LieBasis::Index;
    using Degree = typename LieBasis::Degree;
    using Offset = Index;
    using difference_type = std::ptrdiff_t;

    std::vector<Scalar> data;
    std::vector<Index> indices;
    std::vector<Offset> offsets;

    CompressedMatrixBuilder<std::vector<Scalar>, std::vector<Index>, std::vector<Offset>> builder { data, indices, offsets };

    

    const auto n_offsets = static_cast<difference_type>(offsets.size() - 1);
    const auto nnz = static_cast<difference_type>(data.size());
    return {std::move(data), std::move(indices), std::move(offsets), nnz, n_offsets, basis.size()};
}

} // namespace rpp::sparse

#endif //RPP_SPARSE_T2L_MATRIX_HPP
