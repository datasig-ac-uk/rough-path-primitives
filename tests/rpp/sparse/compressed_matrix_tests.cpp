#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/sparse/compressed_matrix.hpp>

namespace {

using rpp::sparse::CompressedFormat;

template<typename T, typename Size>
[[nodiscard]] auto copy_storage(T const* values, Size size)
{
    const auto count = static_cast<std::ptrdiff_t>(size);
    if (count == 0) {
        return std::vector<T>{};
    }
    return std::vector<T>(values, values + count);
}

template<typename Matrix>
void expect_storage(
    Matrix const& matrix,
    std::vector<typename Matrix::Scalar> const& data,
    std::vector<typename Matrix::Index> const& indices,
    std::vector<typename Matrix::Offset> const& offsets
)
{
    EXPECT_EQ(copy_storage(matrix.data(), matrix.nnz()), data);
    EXPECT_EQ(copy_storage(matrix.indices(), matrix.nnz()), indices);
    EXPECT_EQ(copy_storage(matrix.offsets(), matrix.outer_dim() + 1), offsets);
}

TEST(CompressedMatrixConversionTests, ConvertsCsrToCsc)
{
    using Scalar = int;
    using Index = int;
    using Offset = int;

    std::vector<Scalar> const data{2, -1, 5, 4, 3, 1};
    std::vector<Index> const indices{0, 3, 2, 0, 1, 3};
    std::vector<Offset> const offsets{0, 2, 3, 6};

    auto const csr = rpp::sparse::make_csr_matrix(
        data.data(),
        indices.data(),
        offsets.data(),
        data.size(),
        3,
        4
    );

    auto const csc = rpp::sparse::swap_format(csr);

    static_assert(decltype(csc)::format == CompressedFormat::CSC);
    EXPECT_EQ(csc.nnz(), 6);
    EXPECT_EQ(csc.outer_dim(), 4);
    EXPECT_EQ(csc.inner_dim(), 3);

    auto const csc_view = static_cast<rpp::sparse::CompressedMatrix<
        Scalar const*,
        Index const*,
        Offset const*,
        CompressedFormat::CSC
    > >(csc);
    EXPECT_EQ(csc_view.rows(), 3);
    EXPECT_EQ(csc_view.cols(), 4);

    expect_storage(
        csc,
        std::vector<Scalar>{2, 4, 3, 5, -1, 1},
        std::vector<Index>{0, 2, 2, 1, 0, 2},
        std::vector<Offset>{0, 2, 3, 4, 6}
    );
}

TEST(CompressedMatrixConversionTests, ConvertsCscToCsr)
{
    using Scalar = int;
    using Index = int;
    using Offset = int;

    std::vector<Scalar> const data{2, 4, 3, 5, -1, 1};
    std::vector<Index> const indices{0, 2, 2, 1, 0, 2};
    std::vector<Offset> const offsets{0, 2, 3, 4, 6};

    auto const csc = rpp::sparse::make_csc_matrix(
        data.data(),
        indices.data(),
        offsets.data(),
        data.size(),
        3,
        4
    );

    auto const csr = rpp::sparse::swap_format(csc);

    static_assert(decltype(csr)::format == CompressedFormat::CSR);
    EXPECT_EQ(csr.nnz(), 6);
    EXPECT_EQ(csr.outer_dim(), 3);
    EXPECT_EQ(csr.inner_dim(), 4);

    auto const csr_view = static_cast<rpp::sparse::CompressedMatrix<
        Scalar const*,
        Index const*,
        Offset const*,
        CompressedFormat::CSR
    > >(csr);
    EXPECT_EQ(csr_view.rows(), 3);
    EXPECT_EQ(csr_view.cols(), 4);

    expect_storage(
        csr,
        std::vector<Scalar>{2, -1, 5, 4, 3, 1},
        std::vector<Index>{0, 3, 2, 0, 1, 3},
        std::vector<Offset>{0, 2, 3, 6}
    );
}

TEST(CompressedMatrixConversionTests, PreservesEmptyRowsAndColumns)
{
    using Scalar = int;
    using Index = int;
    using Offset = int;

    std::vector<Scalar> const data{8, 9, 7, 6};
    std::vector<Index> const indices{3, 4, 0, 3};
    std::vector<Offset> const offsets{0, 0, 2, 2, 4};

    auto const csr = rpp::sparse::make_csr_matrix(
        data.data(),
        indices.data(),
        offsets.data(),
        data.size(),
        4,
        5
    );

    auto const csc = rpp::sparse::swap_format(csr);

    EXPECT_EQ(csc.outer_dim(), 5);
    EXPECT_EQ(csc.inner_dim(), 4);
    expect_storage(
        csc,
        std::vector<Scalar>{7, 8, 6, 9},
        std::vector<Index>{3, 1, 3, 1},
        std::vector<Offset>{0, 1, 1, 1, 3, 4}
    );
}

TEST(CompressedMatrixConversionTests, RoundTripsCsrThroughCsc)
{
    using Scalar = int;
    using Index = int;
    using Offset = int;

    std::vector<Scalar> const data{8, 9, 7, 6};
    std::vector<Index> const indices{3, 4, 0, 3};
    std::vector<Offset> const offsets{0, 0, 2, 2, 4};

    auto const csr = rpp::sparse::make_csr_matrix(
        data.data(),
        indices.data(),
        offsets.data(),
        data.size(),
        4,
        5
    );

    auto const csc = rpp::sparse::swap_format(csr);
    auto const roundtrip = rpp::sparse::swap_format(static_cast<rpp::sparse::CompressedMatrix<
        Scalar const*,
        Index const*,
        Offset const*,
        CompressedFormat::CSC
    > >(csc));

    static_assert(decltype(roundtrip)::format == CompressedFormat::CSR);
    EXPECT_EQ(roundtrip.outer_dim(), csr.outer_dim());
    EXPECT_EQ(roundtrip.inner_dim(), csr.inner_dim());
    expect_storage(roundtrip, data, indices, offsets);
}

TEST(CompressedMatrixConversionTests, RoundTripsCscThroughCsr)
{
    using Scalar = int;
    using Index = int;
    using Offset = int;

    std::vector<Scalar> const data{7, 8, 6, 9};
    std::vector<Index> const indices{3, 1, 3, 1};
    std::vector<Offset> const offsets{0, 1, 1, 1, 3, 4};

    auto const csc = rpp::sparse::make_csc_matrix(
        data.data(),
        indices.data(),
        offsets.data(),
        data.size(),
        4,
        5
    );

    auto const csr = rpp::sparse::swap_format(csc);
    auto const roundtrip = rpp::sparse::swap_format(static_cast<rpp::sparse::CompressedMatrix<
        Scalar const*,
        Index const*,
        Offset const*,
        CompressedFormat::CSR
    > >(csr));

    static_assert(decltype(roundtrip)::format == CompressedFormat::CSC);
    EXPECT_EQ(roundtrip.outer_dim(), csc.outer_dim());
    EXPECT_EQ(roundtrip.inner_dim(), csc.inner_dim());
    expect_storage(roundtrip, data, indices, offsets);
}

TEST(CompressedMatrixConversionTests, ConvertsZeroNnzMatrix)
{
    using Scalar = int;
    using Index = int;
    using Offset = int;

    std::vector<Scalar> const data;
    std::vector<Index> const indices;
    std::vector<Offset> const offsets{0, 0, 0};

    auto const csr = rpp::sparse::make_csr_matrix(
        data.data(),
        indices.data(),
        offsets.data(),
        data.size(),
        2,
        3
    );

    auto const csc = rpp::sparse::swap_format(csr);

    EXPECT_EQ(csc.nnz(), 0);
    EXPECT_EQ(csc.outer_dim(), 3);
    EXPECT_EQ(csc.inner_dim(), 2);
    expect_storage(
        csc,
        std::vector<Scalar>{},
        std::vector<Index>{},
        std::vector<Offset>{0, 0, 0, 0}
    );
}

TEST(CompressedMatrixConversionTests, ConvertsGradedCompressedMatrix)
{
    using Scalar = int;
    using Index = int;
    using Offset = int;

    std::vector<Scalar> const data{2, -1, 5, 4, 3, 1};
    std::vector<Index> const indices{0, 3, 2, 0, 1, 3};
    std::vector<Offset> const offsets{0, 2, 3, 6};

    auto const csr = rpp::sparse::GradedCompressedMatrix<
        Scalar const*,
        Index const*,
        Offset const*,
        CompressedFormat::CSR
    >{data.data(), indices.data(), offsets.data(), static_cast<std::ptrdiff_t>(data.size()), 3, 4};

    auto const csc = rpp::sparse::swap_format(csr);

    static_assert(decltype(csc)::format == CompressedFormat::CSC);
    EXPECT_EQ(csc.outer_dim(), 4);
    EXPECT_EQ(csc.inner_dim(), 3);

    auto const csc_view = static_cast<rpp::sparse::GradedCompressedMatrix<
        Scalar const*,
        Index const*,
        Offset const*,
        CompressedFormat::CSC
    > >(csc);
    EXPECT_EQ(csc_view.rows(), 3);
    EXPECT_EQ(csc_view.cols(), 4);

    expect_storage(
        csc,
        std::vector<Scalar>{2, 4, 3, 5, -1, 1},
        std::vector<Index>{0, 2, 2, 1, 0, 2},
        std::vector<Offset>{0, 2, 3, 4, 6}
    );
}

} // namespace
