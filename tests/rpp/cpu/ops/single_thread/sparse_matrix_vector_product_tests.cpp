#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/operations/single_thread/linalg/sparse_matrix_vector.hpp>
#include <rpp/sparse/matrix.hpp>
#include <rpp/views/batch.hpp>
#include <rpp/views/dense_graded_vector_view.hpp>
#include <rpp/views/dense_tensor_view.hpp>
#include <rpp/views/views.hpp>

#include "polynomial_tensor_helper.hpp"

namespace {
class SparseMatrixVectorProductTests
    : public testing::Test,
      public rpp::tests::PolynomialTensorHelper {
protected:
    using Vector = std::vector<Scalar>;

    [[nodiscard]] static Vector make_vector(std::initializer_list<int> values) {
        Vector result;
        result.reserve(values.size());
        for (const int value : values) {
            result.emplace_back(value);
        }
        return result;
    }

    [[nodiscard]] static auto make_csr(Vector const& values,
                                       std::vector<Index> const& indices,
                                       std::vector<Index> const& offsets,
                                       Index rows,
                                       Index cols) {
        return rpp::sparse::make_csr_matrix(values.data(),
                                            indices.data(),
                                            offsets.data(),
                                            static_cast<Index>(values.size()),
                                            rows,
                                            cols);
    }

    [[nodiscard]] static auto make_csc(Vector const& values,
                                       std::vector<Index> const& indices,
                                       std::vector<Index> const& offsets,
                                       Index rows,
                                       Index cols) {
        return rpp::sparse::make_csc_matrix(values.data(),
                                            indices.data(),
                                            offsets.data(),
                                            static_cast<Index>(values.size()),
                                            rows,
                                            cols);
    }
};

TEST_F(SparseMatrixVectorProductTests, AppliesCsrMatrix) {
    auto const out_basis_data = BasisData(2, 1);
    auto const arg_basis_data = BasisData(3, 1);
    auto const& out_basis = out_basis_data.basis;
    auto const& arg_basis = arg_basis_data.basis;

    auto out = make_vector({100, 200, 300});
    auto const arg = make_vector({7, 11, 13, 17});

    auto const values = make_vector({2, -1, 5, 4, 3, 1});
    std::vector<Index> const indices{0, 3, 2, 0, 1, 3};
    std::vector<Index> const offsets{0, 2, 3, 6};
    auto const matrix =
        make_csr(values, indices, offsets, out_basis.size(), arg_basis.size());

    VectorView<Scalar*> out_view(out.data(), out_basis);
    VectorView<Scalar const*> arg_view(arg.data(), arg_basis);

    rpp::ops::SparseMatrixVectorProduct<Strategy, rpp::sparse::CSRMatrix>{}(
        make_context(), out_view, arg_view, matrix);

    EXPECT_EQ(out, make_vector({-3, 65, 78}));
}

TEST_F(SparseMatrixVectorProductTests, AppliesCscMatrix) {
    auto const out_basis_data = BasisData(2, 1);
    auto const arg_basis_data = BasisData(3, 1);
    auto const& out_basis = out_basis_data.basis;
    auto const& arg_basis = arg_basis_data.basis;

    auto out = make_vector({100, 200, 300});
    auto const arg = make_vector({7, 11, 13, 17});

    auto const values = make_vector({2, 4, 3, 5, -1, 1});
    std::vector<Index> const indices{0, 2, 2, 1, 0, 2};
    std::vector<Index> const offsets{0, 2, 3, 4, 6};
    auto const matrix =
        make_csc(values, indices, offsets, out_basis.size(), arg_basis.size());

    VectorView<Scalar*> out_view(out.data(), out_basis);
    VectorView<Scalar const*> arg_view(arg.data(), arg_basis);

    rpp::ops::SparseMatrixVectorProduct<Strategy, rpp::sparse::CSCMatrix>{}(
        make_context(), out_view, arg_view, matrix);

    EXPECT_EQ(out, make_vector({-3, 65, 78}));
}

TEST_F(SparseMatrixVectorProductTests, ScalesResult) {
    auto const out_basis_data = BasisData(2, 1);
    auto const arg_basis_data = BasisData(3, 1);
    auto const& out_basis = out_basis_data.basis;
    auto const& arg_basis = arg_basis_data.basis;

    auto out = make_vector({5, 7, 11});
    auto const arg = make_vector({7, 11, 13, 17});

    auto const values = make_vector({2, -1, 5, 4, 3, 1});
    std::vector<Index> const indices{0, 3, 2, 0, 1, 3};
    std::vector<Index> const offsets{0, 2, 3, 6};
    auto const matrix =
        make_csr(values, indices, offsets, out_basis.size(), arg_basis.size());

    VectorView<Scalar*> out_view(out.data(), out_basis);
    VectorView<Scalar const*> arg_view(arg.data(), arg_basis);

    rpp::ops::SparseMatrixVectorProduct<Strategy, rpp::sparse::CSRMatrix>{}(
        make_context(), out_view, arg_view, matrix, Scalar{2});

    EXPECT_EQ(out, make_vector({-6, 130, 156}));
}

TEST_F(SparseMatrixVectorProductTests, AppliesToLogicalViewRanges) {
    auto const basis_data = BasisData(2, 2);
    auto const& basis = basis_data.basis;

    auto out = make_vector({-1, -1, -1, -1, -1, -1, -1});
    auto const arg = make_vector({-1, -1, -1, 5, 7, 11, 13});

    auto const values = make_vector({1, 2, -1, 1});
    std::vector<Index> const indices{0, 3, 1, 2};
    std::vector<Index> const offsets{0, 2, 4};
    auto const matrix = make_csr(values,
                                 indices,
                                 offsets,
                                 basis.size_of_degree(1),
                                 basis.size_of_degree(2));

    VectorView<Scalar*> out_view(out.data(), basis, 1, 1);
    VectorView<Scalar const*> arg_view(arg.data(), basis, 2, 2);

    rpp::ops::SparseMatrixVectorProduct<Strategy, rpp::sparse::CSRMatrix>{}(
        make_context(), out_view, arg_view, matrix);

    EXPECT_EQ(out, make_vector({-1, 31, 4, -1, -1, -1, -1}));
}

TEST_F(SparseMatrixVectorProductTests, KernelWrapperMatchesDirectOperation) {
    static constexpr Index tensor_count = 2;

    auto const out_basis_data = BasisData(2, 1);
    auto const arg_basis_data = BasisData(3, 1);
    auto const& out_basis = out_basis_data.basis;
    auto const& arg_basis = arg_basis_data.basis;
    const Strategy strategy;

    auto actual = make_vector({5, 7, 11, 2, 3, 5});
    auto expected = actual;
    auto const arg = make_vector({7, 11, 13, 17, 3, 5, 7, 11});

    auto const values = make_vector({2, -1, 5, 4, 3, 1});
    std::vector<Index> const indices{0, 3, 2, 0, 1, 3};
    std::vector<Index> const offsets{0, 2, 3, 6};
    auto const matrix =
        make_csr(values, indices, offsets, out_basis.size(), arg_basis.size());

    auto const out_batch = rpp::make_graded_vector_batch(
        actual.data(), out_basis.size(), out_basis, 0, out_basis.depth);
    auto const arg_batch = rpp::make_graded_vector_batch(
        arg.data(), arg_basis.size(), arg_basis, 0, arg_basis.depth);

    const auto err = rpp::ops::sparse_matrix_vector_product(strategy,
                                                            {},
                                                            out_batch,
                                                            arg_batch,
                                                            out_basis,
                                                            arg_basis,
                                                            tensor_count,
                                                            matrix,
                                                            Scalar{2});

    rpp::ops::SparseMatrixVectorProduct<Strategy, rpp::sparse::CSRMatrix> op;
    auto const ctx = make_context();
    for (Index tensor_idx = 0; tensor_idx < tensor_count; ++tensor_idx) {
        auto out = rpp::DenseGradedVectorView<Scalar*, Basis>(
            expected.data() +
                static_cast<std::size_t>(tensor_idx * out_basis.size()),
            out_basis);
        auto rhs = rpp::DenseGradedVectorView<Scalar const*, Basis>(
            arg.data() +
                static_cast<std::size_t>(tensor_idx * arg_basis.size()),
            arg_basis);
        op(ctx, out, rhs, matrix, Scalar{2});
    }

    EXPECT_EQ(actual, expected);
}
} // namespace
