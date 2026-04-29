#ifndef INCLUDE_RPP_SPARSE_COMPRESSED_MATRIX_HPP
#define INCLUDE_RPP_SPARSE_COMPRESSED_MATRIX_HPP

#include <type_traits>
#include <iterator>

#include <rpp/config.h>


namespace rpp::sparse {

enum class CompressedFormat {
    CSC,
    CSR
};

template <typename DataIter, typename IndexIter, typename OffsetsIter, CompressedFormat Format>
class CompressedMatrix {
    using DataTraits = std::iterator_traits<DataIter>;
    using IndexTraits = std::iterator_traits<IndexIter>;
    using OffsetTraits = std::iterator_traits<OffsetsIter>;

public:

    using difference_type = std::common_type_t<
        typename DataTraits::difference_type,
        typename IndexTraits::value_type,
        typename OffsetTraits::value_type
    >;

    using Scalar = typename DataTraits::value_type;
    using value_type = Scalar;
    using reference = typename DataTraits::reference;
    using Offset = typename OffsetTraits::value_type;
    using Index = typename IndexTraits::value_type;

    static constexpr auto format = Format;

private:

    DataIter data_;
    IndexIter indices_;
    OffsetsIter offsets_;

    difference_type nnz_;
    difference_type outer_dim_;
    difference_type inner_dim_;

public:

    RPP_HOST_DEVICE constexpr CompressedMatrix(
        DataIter data,
        IndexIter indices,
        OffsetsIter offsets,
        difference_type nnz,
        difference_type outer_dim,
        difference_type inner_dim)
            : data_(data), indices_(indices), offsets_(offsets), nnz_(nnz), outer_dim_(outer_dim), inner_dim_(inner_dim)
    {}

    RPP_HOST_DEVICE
    constexpr auto rows() const noexcept {
        if constexpr (Format == CompressedFormat::CSC) {
            return inner_dim_;
        } else {
            return outer_dim_;
        }
    }

    RPP_HOST_DEVICE
    constexpr auto cols() const noexcept {
        if constexpr (Format == CompressedFormat::CSC) {
            return outer_dim_;
        } else {
            return inner_dim_;
        }
    }

    RPP_HOST_DEVICE
    constexpr auto num_non_zero() const noexcept { return nnz_; }

    RPP_HOST_DEVICE
    constexpr auto outer_dim() const noexcept { return outer_dim_; }

    RPP_HOST_DEVICE
    constexpr auto inner_dim() const noexcept { return inner_dim_; }

    RPP_HOST_DEVICE
    constexpr DataIter data() const noexcept { return data_; }

    RPP_HOST_DEVICE
    constexpr IndexIter indices() const noexcept { return indices_; }

    RPP_HOST_DEVICE
    constexpr OffsetsIter offsets() const noexcept { return offsets_; }

    RPP_HOST_DEVICE
    constexpr reference value(difference_type index) const noexcept { return data_[index]; }

    RPP_HOST_DEVICE
    constexpr Index inner_index(difference_type index) const noexcept { return indices_[index]; }

    RPP_HOST_DEVICE
    constexpr Offset offset(difference_type index) const noexcept { return offsets_[index]; }
};

template <typename DataIter, typename IndexIter, typename OffsetsIter>
using CSRMatrix = CompressedMatrix<DataIter, IndexIter, OffsetsIter, CompressedFormat::CSR>;

template <typename DataIter, typename IndexIter, typename OffsetsIter>
using CSCMatrix = CompressedMatrix<DataIter, IndexIter, OffsetsIter, CompressedFormat::CSC>;

template <CompressedFormat Format, typename DataIter, typename IndexIter, typename OffsetsIter, typename Nnz, typename OuterDim, typename InnerDim>
RPP_HOST_DEVICE constexpr auto make_compressed_matrix(
    DataIter data,
    IndexIter indices,
    OffsetsIter offsets,
    Nnz nnz,
    OuterDim outer_dim,
    InnerDim inner_dim
) noexcept {
    return CompressedMatrix<DataIter, IndexIter, OffsetsIter, Format>{
        data,
        indices,
        offsets,
        static_cast<typename CompressedMatrix<DataIter, IndexIter, OffsetsIter, Format>::difference_type>(nnz),
        static_cast<typename CompressedMatrix<DataIter, IndexIter, OffsetsIter, Format>::difference_type>(outer_dim),
        static_cast<typename CompressedMatrix<DataIter, IndexIter, OffsetsIter, Format>::difference_type>(inner_dim)
    };
}

template <typename DataIter, typename IndexIter, typename OffsetsIter, typename Nnz, typename Rows, typename Cols>
RPP_HOST_DEVICE constexpr auto make_csr_matrix(
    DataIter data,
    IndexIter indices,
    OffsetsIter offsets,
    Nnz nnz,
    Rows rows,
    Cols cols
) noexcept {
    return make_compressed_matrix<CompressedFormat::CSR>(data, indices, offsets, nnz, rows, cols);
}

template <typename DataIter, typename IndexIter, typename OffsetsIter, typename Nnz, typename Rows, typename Cols>
RPP_HOST_DEVICE constexpr auto make_csc_matrix(
    DataIter data,
    IndexIter indices,
    OffsetsIter offsets,
    Nnz nnz,
    Rows rows,
    Cols cols
) noexcept {
    return make_compressed_matrix<CompressedFormat::CSC>(data, indices, offsets, nnz, cols, rows);
}

} // namespace rpp::sparse

#endif //INCLUDE_RPP_SPARSE_COMPRESSED_MATRIX_HPP
