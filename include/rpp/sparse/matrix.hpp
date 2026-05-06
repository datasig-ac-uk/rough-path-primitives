#ifndef INCLUDE_RPP_SPARSE_MATRIX_HPP
#define INCLUDE_RPP_SPARSE_MATRIX_HPP

#include <rpp/sparse/compressed_matrix.hpp>

#include "rpp/utility.hpp"

namespace rpp::sparse {
enum class MatrixFormat {
    CSR,
    CSC,
};

namespace detail {
template<MatrixFormat Format>
struct SparseMatrixTypeImpl;


template<>
struct SparseMatrixTypeImpl<MatrixFormat::CSR> {
    template<typename Data, typename Indices, typename Offsets>
    using ViewType = CompressedMatrix<Data, Indices, Offsets, CompressedFormat::CSR>;

    template<typename Data, typename Indices, typename Offsets>
    using GradedViewType = GradedCompressedMatrix<Data, Indices, Offsets, CompressedFormat::CSR>;

    template<typename Data, typename Indices, typename Offsets>
    using OwnedType = OwnedCompressedMatrix<Data, Indices, Offsets, CompressedFormat::CSR>;

    template<typename Data, typename Indices, typename Offsets>
    using GradedOwnedType = OwnedGradedCompressedMatrix<Data, Indices, Offsets, CompressedFormat::CSR>;
};

template<>
struct SparseMatrixTypeImpl<MatrixFormat::CSC> {
    template<typename Data, typename Indices, typename Offsets>
    using ViewType = CompressedMatrix<Data, Indices, Offsets, CompressedFormat::CSC>;

    template<typename Data, typename Indices, typename Offsets>
    using GradedViewType = GradedCompressedMatrix<Data, Indices, Offsets, CompressedFormat::CSC>;

    template<typename Data, typename Indices, typename Offsets>
    using OwnedType = OwnedCompressedMatrix<Data, Indices, Offsets, CompressedFormat::CSC>;

    template<typename Data, typename Indices, typename Offsets>
    using GradedOwnedType = OwnedGradedCompressedMatrix<Data, Indices, Offsets, CompressedFormat::CSC>;
};


template<typename Matrix>
struct MatrixFormatImpl {
    static_assert(static_assert_fail<Matrix>, "Unrecognised sparse matrix type");
};


template<CompressedFormat F, typename D, typename I, typename O>
struct MatrixFormatImpl<CompressedMatrix<D, I, O, F> > {
    static constexpr auto value = (F == CompressedFormat::CSC)
                                      ? MatrixFormat::CSC
                                      : MatrixFormat::CSR;
};

} // namespace detail


template<MatrixFormat F, typename... Args>
using MatrixView = typename detail::SparseMatrixTypeImpl<F>::template ViewType<Args...>;

template<MatrixFormat F, typename... Args>
using GradedMatrixView = typename detail::SparseMatrixTypeImpl<F>::template GradedViewType<Args...>;

template<MatrixFormat F, typename... Args>
using MatrixOwned = typename detail::SparseMatrixTypeImpl<F>::template OwnedType<Args...>;

template<MatrixFormat F, typename... Args>
using GradedMatrixOwned = typename detail::SparseMatrixTypeImpl<F>::template GradedOwnedType<Args...>;


template<typename Matrix>
inline constexpr auto matrix_format_v = detail::MatrixFormatImpl<Matrix>::value;
} // namespace rpp::sparse

#endif //INCLUDE_RPP_SPARSE_MATRIX_HPP
