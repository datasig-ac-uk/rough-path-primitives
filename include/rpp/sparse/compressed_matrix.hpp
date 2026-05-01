#ifndef INCLUDE_RPP_SPARSE_COMPRESSED_MATRIX_HPP
#define INCLUDE_RPP_SPARSE_COMPRESSED_MATRIX_HPP

#include <type_traits>
#include <iterator>
#include <vector>
#include <algorithm>
#include <memory>

#include <rpp/config.h>


namespace rpp::sparse {
enum class CompressedFormat {
    CSC,
    CSR
};

template<typename DataIter, typename IndexIter, typename OffsetsIter, CompressedFormat Format>
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
    DataIter data_; // size = nnz_
    IndexIter indices_; // size = nnz_
    OffsetsIter offsets_; // size = outer_dim_ + 1

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
        : data_(data), indices_(indices), offsets_(offsets), nnz_(nnz), outer_dim_(outer_dim), inner_dim_(inner_dim) {
    }

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

template<typename DataIter, typename IndexIter, typename OffsetsIter>
using CSRMatrix = CompressedMatrix<DataIter, IndexIter, OffsetsIter, CompressedFormat::CSR>;

template<typename DataIter, typename IndexIter, typename OffsetsIter>
using CSCMatrix = CompressedMatrix<DataIter, IndexIter, OffsetsIter, CompressedFormat::CSC>;

template<CompressedFormat Format, typename DataIter, typename IndexIter, typename OffsetsIter, typename Nnz, typename
    OuterDim, typename InnerDim>
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

template<typename DataIter, typename IndexIter, typename OffsetsIter, typename Nnz, typename Rows, typename Cols>
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

template<typename DataIter, typename IndexIter, typename OffsetsIter, typename Nnz, typename Rows, typename Cols>
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


template<typename DataContainer, typename IndexContainer, typename OffsetContainer, CompressedFormat Format>
class OwnedCompressedMatrix {
    DataContainer data_;
    IndexContainer indices_;
    OffsetContainer offsets_;

public:
    using Scalar = typename DataContainer::value_type;
    using Index = typename IndexContainer::value_type;
    using Offset = typename OffsetContainer::value_type;

    using difference_type = typename DataContainer::difference_type;
    using value_type = Scalar;
    using reference = typename DataContainer::reference;

    static constexpr auto format = Format;

    using DataPointer = decltype(std::declval<DataContainer const &>().data());
    using IndexPointer = decltype(std::declval<IndexContainer const &>().data());
    using OffsetPointer = decltype(std::declval<OffsetContainer const &>().data());

private:
    difference_type nnz_;
    difference_type outer_dim_;
    difference_type inner_dim_;

public:
    constexpr OwnedCompressedMatrix(
        DataContainer &&data, IndexContainer &&indices, OffsetContainer &&offsets,
        difference_type nnz, difference_type outer_dim, difference_type inner_dim
    ) noexcept
        : data_(std::move(data)), indices_(std::move(indices)), offsets_(std::move(offsets)),
          nnz_(nnz), outer_dim_(outer_dim), inner_dim_(inner_dim) {
    }

    constexpr explicit operator CompressedMatrix<Scalar const *, Index const *, Offset const *, Format>() const noexcept {
        return {data_.data(), indices_.data(), offsets_.data(), nnz_, outer_dim_, inner_dim_};
    }
};

template<typename Scalar_, typename DataDeleter, typename Index_, typename IndexDeleter, typename Offset_, typename
    OffsetDeleter, CompressedFormat Format>
class OwnedCompressedMatrix<std::unique_ptr<Scalar_[], DataDeleter>, std::unique_ptr<Index_[], IndexDeleter>,
            std::unique_ptr<Offset_[], OffsetDeleter>, Format> {
    using DataPointer = std::unique_ptr<Scalar_[], DataDeleter>;
    using IndexPointer = std::unique_ptr<Index_[], IndexDeleter>;
    using OffsetPointer = std::unique_ptr<Offset_[], OffsetDeleter>;

    DataPointer data_;
    IndexPointer indices_;
    OffsetPointer offsets_;

    std::ptrdiff_t nnz_;
    std::ptrdiff_t outer_dim_;
    std::ptrdiff_t inner_dim_;

public:
    using Scalar = Scalar_;
    using Index = Index_;
    using Offset = Offset_;
    using difference_type = std::ptrdiff_t;

    constexpr OwnedCompressedMatrix(
        DataPointer &&data,
        IndexPointer &&indices,
        OffsetPointer &&offsets,
        difference_type nnz,
        difference_type outer_dim,
        difference_type inner_dim
    ) noexcept : data_(std::move(data)), indices_(std::move(indices)), offsets_(std::move(offsets)),
                 nnz_(nnz), outer_dim_(outer_dim), inner_dim_(inner_dim) {
    }

    constexpr explicit operator CompressedMatrix<Scalar const *, Index const *, Offset const *, Format>() const noexcept {
        return { data_.get(), indices_.get(), offsets_.get(), nnz_, outer_dim_, inner_dim_ };
    }

};




template <typename DataContainer, typename IndexContainer, typename OffsetContainer>
class CompressedMatrixBuilder {
    DataContainer& data_;
    IndexContainer& indices_;
    OffsetContainer& offsets_;
    using difference_type = typename DataContainer::difference_type;

    using data_iterator = typename DataContainer::iterator;
    using index_iterator = typename IndexContainer::iterator;
    using offset_iterator = typename OffsetContainer::iterator;

    using Scalar = typename DataContainer::value_type;
    using Index = typename IndexContainer::value_type;
    using Offset = typename OffsetContainer::value_type;

public:
    struct MatrixFrame {
        data_iterator data;
        index_iterator index;
        difference_type size;
    };

private:
    std::vector<MatrixFrame> frames_;

    MatrixFrame& current_frame() noexcept { return frames_.back(); }

    auto get_or_insert(Index index) noexcept {
        auto& frame = current_frame();

        auto idx_it = std::lower_bound(frame.index, indices_.end(), index);
        const auto offset = static_cast<difference_type>(idx_it - frame.index);
        auto data_it = frame.data + offset;

        if (idx_it == indices_.end() || *idx_it != index) {
            data_it = data_.insert(data_it, Scalar{0});
            idx_it = indices_.insert(idx_it, index);
            ++frame.size;
        }

        return std::make_pair(data_it, idx_it);
    }


public:

    RPP_HOST
    constexpr CompressedMatrixBuilder(DataContainer& data, IndexContainer& indices, OffsetContainer& offsets)
        : data_(data), indices_(indices), offsets_(offsets) {
        offsets_.push_back(0);
    }

    RPP_HOST
    MatrixFrame const& operator[](difference_type index) const noexcept {
        return frames_[index];
    }

    RPP_HOST
    void next_frame() {
        offsets_.push_back(offsets_.back() + current_frame().size());
        frames_.emplace_back(data_.end(), indices_.end(), 0);
    }

    RPP_HOST RPP_NODISCARD
    Scalar& get_scalar(Index index) noexcept {
        auto [data_it, _] = get_or_insert(index);
        return *data_it;
    }

};



} // namespace rpp::sparse

#endif //INCLUDE_RPP_SPARSE_COMPRESSED_MATRIX_HPP
