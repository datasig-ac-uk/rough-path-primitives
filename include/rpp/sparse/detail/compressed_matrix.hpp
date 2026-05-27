#ifndef RPP_SPARSE_COMPRESSED_MATRIX_HPP
#define RPP_SPARSE_COMPRESSED_MATRIX_HPP

#include <algorithm>
#include <memory>
#include <type_traits>
#include <vector>

#include <rpp/architecture.hpp>
#include <rpp/config.h>
#include <rpp/support/data_mapping.hpp>
#include <rpp/support/iterator_traits.hpp>
#include <rpp/utility.hpp>

#include "rpp/utility.hpp"


namespace rpp::sparse {
namespace detail {
enum class CompressedFormat {
    CSR,
    CSC,
};

template <CompressedFormat F>
inline constexpr auto swapped_format = CompressedFormat::CSR;

template <>
inline constexpr auto swapped_format<CompressedFormat::CSR> =
    CompressedFormat::CSC;
} // namespace detail

template <typename DataIter,
          typename IndexIter,
          typename OffsetsIter,
          detail::CompressedFormat Format>
class CompressedMatrix {
    using DataTraits = traits::IteratorTraits<DataIter>;
    using IndexTraits = traits::IteratorTraits<IndexIter>;
    using OffsetTraits = traits::IteratorTraits<OffsetsIter>;

public:
    using Architecture =
        traits::all_same_type_t<traits::arch_of_t<DataIter>,
                                traits::arch_of_t<IndexIter>,
                                traits::arch_of_t<OffsetsIter>>;

    using difference_type = typename Architecture::Index;

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
    RPP_HOST_DEVICE constexpr CompressedMatrix(DataIter data,
                                               IndexIter indices,
                                               OffsetsIter offsets,
                                               difference_type nnz,
                                               difference_type outer_dim,
                                               difference_type inner_dim)
        : data_(data), indices_(indices), offsets_(offsets), nnz_(nnz),
          outer_dim_(outer_dim), inner_dim_(inner_dim) {}

    RPP_HOST_DEVICE
    constexpr auto rows() const noexcept {
        if constexpr (Format == detail::CompressedFormat::CSC) {
            return inner_dim_;
        }
        else {
            return outer_dim_;
        }
    }

    RPP_HOST_DEVICE
    constexpr auto cols() const noexcept {
        if constexpr (Format == detail::CompressedFormat::CSC) {
            return outer_dim_;
        }
        else {
            return inner_dim_;
        }
    }

    RPP_HOST_DEVICE
    constexpr auto nnz() const noexcept { return nnz_; }


    /**
     * Get the outer dimension of the matrix.
     *
     * When the matrix is a CSC matrix this is the number of columns.
     * When the matrix is a CSR matrix this is the number of rows.
     *
     * @return integer outer dimension
     */
    RPP_HOST_DEVICE
    constexpr auto outer_dim() const noexcept { return outer_dim_; }

    /**
     * Get the inner dimension of the matrix.
     *
     * When the matrix is a CSC matrix this is the number of rows.
     * When the matrix is a CSR matrix this is the number of columns.
     *
     * @return integer inner dimension
     */
    RPP_HOST_DEVICE
    constexpr auto inner_dim() const noexcept { return inner_dim_; }

    RPP_HOST_DEVICE
    constexpr DataIter data() const noexcept { return data_; }

    RPP_HOST_DEVICE
    constexpr IndexIter indices() const noexcept { return indices_; }

    RPP_HOST_DEVICE
    constexpr OffsetsIter offsets() const noexcept { return offsets_; }

    RPP_HOST_DEVICE
    constexpr reference value(difference_type index) const noexcept {
        return data_[index];
    }

    RPP_HOST_DEVICE
    constexpr Index inner_index(difference_type index) const noexcept {
        return indices_[index];
    }

    RPP_HOST_DEVICE
    constexpr Offset offset(difference_type index) const noexcept {
        return offsets_[index];
    }

    RPP_HOST_DEVICE
    constexpr auto offsets(difference_type inner_dim) const noexcept {
        return std::tie(offsets_[inner_dim], offsets_[inner_dim + 1]);
    }

    template <typename DataMapper>
    RPP_NODISCARD friend auto map_data(DataMapper& mapper,
                                       CompressedMatrix const& matrix) noexcept
        -> traits::data_map_result_t<
            DataMapper,
            CompressedMatrix<traits::data_map_value_t<DataMapper, DataIter>,
                             traits::data_map_index_t<DataMapper, IndexIter>,
                             traits::data_map_index_t<DataMapper, OffsetsIter>,
                             Format>> {
        using Result = traits::data_map_result_t<
            DataMapper,
            CompressedMatrix<traits::data_map_value_t<DataMapper, DataIter>,
                             traits::data_map_index_t<DataMapper, IndexIter>,
                             traits::data_map_index_t<DataMapper, OffsetsIter>,
                             Format>>;
        using MappedIndex = typename DataMapper::Index;

        auto mapped_data = map_value_range(mapper, matrix.data(), matrix.nnz());
        if (!mapped_data) {
            return Result{std::move(mapped_data).error()};
        }

        auto mapped_indices =
            map_index_range(mapper, matrix.indices(), matrix.nnz());
        if (!mapped_indices) {
            return Result{std::move(mapped_indices).error()};
        }

        auto mapped_offsets =
            map_index_range(mapper, matrix.offsets(), matrix.outer_dim() + 1);

        if (!mapped_offsets) {
            return Result{std::move(mapped_offsets).error()};
        }

        return Result{CompressedMatrix<
            traits::data_map_value_t<DataMapper, DataIter>,
            traits::data_map_index_t<DataMapper, IndexIter>,
            traits::data_map_index_t<DataMapper, OffsetsIter>,
            Format>{std::move(mapped_data).value(),
                    std::move(mapped_indices).value(),
                    std::move(mapped_offsets).value(),
                    static_cast<MappedIndex>(matrix.nnz()),
                    static_cast<MappedIndex>(matrix.outer_dim()),
                    static_cast<MappedIndex>(matrix.inner_dim())}};
    }
};

template <detail::CompressedFormat Format,
          typename DataIter,
          typename IndexIter,
          typename OffsetsIter,
          typename Nnz,
          typename OuterDim,
          typename InnerDim>
RPP_HOST_DEVICE constexpr auto
make_compressed_matrix(DataIter data,
                       IndexIter indices,
                       OffsetsIter offsets,
                       Nnz nnz,
                       OuterDim outer_dim,
                       InnerDim inner_dim) noexcept {
    return CompressedMatrix<DataIter, IndexIter, OffsetsIter, Format>{
        data,
        indices,
        offsets,
        static_cast<typename CompressedMatrix<DataIter,
                                              IndexIter,
                                              OffsetsIter,
                                              Format>::difference_type>(nnz),
        static_cast<typename CompressedMatrix<DataIter,
                                              IndexIter,
                                              OffsetsIter,
                                              Format>::difference_type>(
            outer_dim),
        static_cast<typename CompressedMatrix<DataIter,
                                              IndexIter,
                                              OffsetsIter,
                                              Format>::difference_type>(
            inner_dim)};
}

template <typename DataIter,
          typename IndexIter,
          typename OffsetsIter,
          typename Nnz,
          typename Rows,
          typename Cols>
RPP_HOST_DEVICE constexpr auto make_csr_matrix(DataIter data,
                                               IndexIter indices,
                                               OffsetsIter offsets,
                                               Nnz nnz,
                                               Rows rows,
                                               Cols cols) noexcept {
    return make_compressed_matrix<detail::CompressedFormat::CSR>(
        data, indices, offsets, nnz, rows, cols);
}

template <typename DataIter,
          typename IndexIter,
          typename OffsetsIter,
          typename Nnz,
          typename Rows,
          typename Cols>
RPP_HOST_DEVICE constexpr auto make_csc_matrix(DataIter data,
                                               IndexIter indices,
                                               OffsetsIter offsets,
                                               Nnz nnz,
                                               Rows rows,
                                               Cols cols) noexcept {
    return make_compressed_matrix<detail::CompressedFormat::CSC>(
        data, indices, offsets, nnz, cols, rows);
}


template <typename DataContainer,
          typename IndexContainer,
          typename OffsetContainer,
          detail::CompressedFormat Format>
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

    using DataPointer = decltype(std::declval<DataContainer const&>().data());
    using IndexPointer = decltype(std::declval<IndexContainer const&>().data());
    using OffsetPointer =
        decltype(std::declval<OffsetContainer const&>().data());

private:
    difference_type nnz_;
    difference_type outer_dim_;
    difference_type inner_dim_;

public:
    constexpr OwnedCompressedMatrix(DataContainer&& data,
                                    IndexContainer&& indices,
                                    OffsetContainer&& offsets,
                                    difference_type nnz,
                                    difference_type outer_dim,
                                    difference_type inner_dim) noexcept
        : data_(std::move(data)), indices_(std::move(indices)),
          offsets_(std::move(offsets)), nnz_(nnz), outer_dim_(outer_dim),
          inner_dim_(inner_dim) {}

    constexpr auto data() const noexcept { return data_.data(); }
    constexpr auto indices() const noexcept { return indices_.data(); }
    constexpr auto offsets() const noexcept { return offsets_.data(); }
    constexpr auto nnz() const noexcept { return nnz_; }
    constexpr auto outer_dim() const noexcept { return outer_dim_; }
    constexpr auto inner_dim() const noexcept { return inner_dim_; }

    constexpr CompressedMatrix<DataPointer, IndexPointer, OffsetPointer, Format>
    view() const noexcept {
        return {data(), indices(), offsets(), nnz(), outer_dim(), inner_dim()};
    }

    constexpr explicit operator CompressedMatrix<DataPointer,
                                                 IndexPointer,
                                                 OffsetPointer,
                                                 Format>() const noexcept {
        return {data(), indices(), offsets(), nnz(), outer_dim(), inner_dim()};
    }
};

template <typename Scalar_,
          typename DataDeleter,
          typename Index_,
          typename IndexDeleter,
          typename Offset_,
          typename OffsetDeleter,
          detail::CompressedFormat Format>
class OwnedCompressedMatrix<std::unique_ptr<Scalar_[], DataDeleter>,
                            std::unique_ptr<Index_[], IndexDeleter>,
                            std::unique_ptr<Offset_[], OffsetDeleter>,
                            Format> {
    using OwnedDataPointer = std::unique_ptr<Scalar_[], DataDeleter>;
    using OwnedIndexPointer = std::unique_ptr<Index_[], IndexDeleter>;
    using OwnedOffsetPointer = std::unique_ptr<Offset_[], OffsetDeleter>;

    OwnedDataPointer data_;
    OwnedIndexPointer indices_;
    OwnedOffsetPointer offsets_;

    std::ptrdiff_t nnz_;
    std::ptrdiff_t outer_dim_;
    std::ptrdiff_t inner_dim_;

public:
    using Scalar = Scalar_;
    using Index = Index_;
    using Offset = Offset_;
    using difference_type = std::ptrdiff_t;
    using DataPointer = Scalar const*;
    using IndexPointer = Index const*;
    using OffsetPointer = Offset const*;

    constexpr OwnedCompressedMatrix(OwnedDataPointer&& data,
                                    OwnedIndexPointer&& indices,
                                    OwnedOffsetPointer&& offsets,
                                    difference_type nnz,
                                    difference_type outer_dim,
                                    difference_type inner_dim) noexcept
        : data_(std::move(data)), indices_(std::move(indices)),
          offsets_(std::move(offsets)), nnz_(nnz), outer_dim_(outer_dim),
          inner_dim_(inner_dim) {}


    constexpr auto data() const noexcept { return data_.get(); }
    constexpr auto indices() const noexcept { return indices_.get(); }
    constexpr auto offsets() const noexcept { return offsets_.get(); }
    constexpr auto nnz() const noexcept { return nnz_; }
    constexpr auto outer_dim() const noexcept { return outer_dim_; }
    constexpr auto inner_dim() const noexcept { return inner_dim_; }

    constexpr CompressedMatrix<Scalar const*,
                               Index const*,
                               Offset const*,
                               Format>
    view() const noexcept {
        return {data(), indices(), offsets(), nnz(), outer_dim(), inner_dim()};
    }

    constexpr explicit operator CompressedMatrix<Scalar const*,
                                                 Index const*,
                                                 Offset const*,
                                                 Format>() const noexcept {
        return {data(), indices(), offsets(), nnz(), outer_dim(), inner_dim()};
    }
};


template <typename DataIter,
          typename IndexIter,
          typename OffsetIter,
          detail::CompressedFormat Format>
class GradedCompressedMatrix
    : public CompressedMatrix<DataIter, IndexIter, OffsetIter, Format> {
    using Base = CompressedMatrix<DataIter, IndexIter, OffsetIter, Format>;

public:
    using Base::Base;
};

template <typename DataContainer,
          typename IndexContainer,
          typename OffsetContainer,
          detail::CompressedFormat Format>
class OwnedGradedCompressedMatrix
    : public OwnedCompressedMatrix<DataContainer,
                                   IndexContainer,
                                   OffsetContainer,
                                   Format> {
    using Base = OwnedCompressedMatrix<DataContainer,
                                       IndexContainer,
                                       OffsetContainer,
                                       Format>;

    using GradedMatrix =
        ::rpp::sparse::GradedCompressedMatrix<typename Base::DataPointer,
                                              typename Base::IndexPointer,
                                              typename Base::OffsetPointer,
                                              Format>;

public:
    using Base::Base;

    using Base::data;
    using Base::indices;
    using Base::inner_dim;
    using Base::nnz;
    using Base::offsets;
    using Base::outer_dim;

    explicit operator GradedMatrix() const noexcept {
        return {data(), indices(), offsets(), nnz(), outer_dim(), inner_dim()};
    }

    constexpr GradedMatrix view() const noexcept {
        return {data(), indices(), offsets(), nnz(), outer_dim(), inner_dim()};
    }
};


template <typename DataContainer,
          typename IndexContainer,
          typename OffsetContainer>
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
    constexpr CompressedMatrixBuilder(DataContainer& data,
                                      IndexContainer& indices,
                                      OffsetContainer& offsets)
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

    RPP_HOST RPP_NODISCARD Scalar& get_scalar(Index index) noexcept {
        auto [data_it, _] = get_or_insert(index);
        return *data_it;
    }
};


namespace detail {
template <typename DataOut,
          typename IndexOut,
          typename OffsetsOut,
          typename Matrix>
void rearrange_data(DataOut data,
                    IndexOut indices,
                    OffsetsOut offsets,
                    Matrix const& matrix) noexcept {
    using OffsetOutTraits = std::iterator_traits<OffsetsOut>;

    const auto outer_dim = matrix.outer_dim();
    using OuterDim = std::remove_cv_t<decltype(outer_dim)>;

    const auto inner_dim = matrix.inner_dim();
    using InnerDim = std::remove_cv_t<decltype(inner_dim)>;

    const auto offsets_in = matrix.offsets();
    const auto indices_in = matrix.indices();

    for (OuterDim i = 0; i < outer_dim; ++i) {
        const auto begin = offsets_in[i];
        const auto end = offsets_in[i + 1];
        for (auto j = begin; j < end; ++j) {
            auto idx = indices_in[j];
            ++offsets[idx];
        }
    }

    typename OffsetOutTraits::value_type sum{0};
    for (InnerDim i = 0; i < inner_dim; ++i) {
        auto val = offsets[i];
        offsets[i] = sum;
        sum += val;
    }
    offsets[inner_dim] = sum;

    const auto data_in = matrix.data();

    std::vector write_locs(offsets, offsets + inner_dim);

    for (OuterDim i = 0; i < outer_dim; ++i) {
        const auto begin = offsets_in[i];
        const auto end = offsets_in[i + 1];

        for (auto j = begin; j < end; ++j) {
            const auto inner_index = indices_in[j];
            const auto dest = write_locs[inner_index]++;

            indices[dest] = i;
            data[dest] = data_in[j];
        }
    }
}
} // namespace detail


template <typename DataIt,
          typename IndexIt,
          typename OffsetIt,
          detail::CompressedFormat Format>
auto swap_format(CompressedMatrix<DataIt, IndexIt, OffsetIt, Format> const&
                     matrix) noexcept {
    using DataTraits = std::iterator_traits<DataIt>;
    using IndexTraits = std::iterator_traits<IndexIt>;
    using OffsetTraits = std::iterator_traits<OffsetIt>;

    using DataVec = std::vector<typename DataTraits::value_type>;
    using IndexVec = std::vector<typename IndexTraits::value_type>;
    using OffsetVec = std::vector<typename OffsetTraits::value_type>;

    using MatrixOut = OwnedCompressedMatrix<DataVec,
                                            IndexVec,
                                            OffsetVec,
                                            detail::swapped_format<Format>>;

    DataVec data(matrix.nnz(), typename DataTraits::value_type{});
    IndexVec indices(matrix.nnz(), 0);
    OffsetVec offsets(matrix.inner_dim() + 1, 0);

    detail::rearrange_data(
        data.begin(), indices.begin(), offsets.begin(), matrix);

    return MatrixOut{std::move(data),
                     std::move(indices),
                     std::move(offsets),
                     matrix.nnz(),
                     matrix.inner_dim(),
                     matrix.outer_dim()};
}


template <typename DataIt,
          typename IndexIt,
          typename OffsetIt,
          detail::CompressedFormat Format>
auto swap_format(
    GradedCompressedMatrix<DataIt, IndexIt, OffsetIt, Format> const&
        matrix) noexcept {
    using DataTraits = std::iterator_traits<DataIt>;
    using IndexTraits = std::iterator_traits<IndexIt>;
    using OffsetTraits = std::iterator_traits<OffsetIt>;

    using DataVec = std::vector<typename DataTraits::value_type>;
    using IndexVec = std::vector<typename IndexTraits::value_type>;
    using OffsetVec = std::vector<typename OffsetTraits::value_type>;

    using MatrixOut =
        OwnedGradedCompressedMatrix<DataVec,
                                    IndexVec,
                                    OffsetVec,
                                    detail::swapped_format<Format>>;

    DataVec data(matrix.nnz(), typename DataTraits::value_type{});
    IndexVec indices(matrix.nnz(), 0);
    OffsetVec offsets(matrix.inner_dim() + 1, 0);

    detail::rearrange_data(
        data.begin(), indices.begin(), offsets.begin(), matrix);

    return MatrixOut{std::move(data),
                     std::move(indices),
                     std::move(offsets),
                     matrix.nnz(),
                     matrix.inner_dim(),
                     matrix.outer_dim()};
}
} // namespace rpp::sparse

#endif // RPP_SPARSE_COMPRESSED_MATRIX_HPP
