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

    [[nodiscard]] static Scalar constant(int numerator, int denominator = 1)
    {
        return make_scalar({
            {{}, numerator, denominator}
        });
    }

    [[nodiscard]] static std::size_t matrix_position(Index row, Index col) noexcept
    {
        return static_cast<std::size_t>((row + 1) * 10 + (col + 1));
    }

    [[nodiscard]] static Scalar matrix_symbol(char marker, Index row, Index col)
    {
        return make_scalar({
            {{{marker, matrix_position(row, col)}}, 1, 1}
        });
    }

    [[nodiscard]] static auto make_matrix_values(
        char marker,
        std::vector<Index> const& row_indices,
        std::vector<Index> const& col_indices
    )
    {
        Vector result;
        result.reserve(row_indices.size());
        for (std::size_t i = 0; i < row_indices.size(); ++i) {
            result.push_back(matrix_symbol(marker, row_indices[i], col_indices[i]));
        }
        return result;
    }

    [[nodiscard]] static auto make_csr(Vector const& values,
                                       std::vector<Index> const& indices,
                                       std::vector<Index> const& offsets,
                                       Index rows,
                                       Index cols)
    {
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
                                       Index cols)
    {
        return rpp::sparse::make_csc_matrix(values.data(),
                                            indices.data(),
                                            offsets.data(),
                                            static_cast<Index>(values.size()),
                                            rows,
                                            cols);
    }
};

TEST_F(SparseMatrixVectorProductTests, AppliesCsrMatrix)
{
    auto const out_basis_data = BasisData(2, 1);
    auto const arg_basis_data = BasisData(3, 1);
    auto const& out_basis = out_basis_data.basis;
    auto const& arg_basis = arg_basis_data.basis;

    auto out = make_tensor('o', out_basis);
    auto const arg = make_tensor('x', arg_basis);

    std::vector<Index> const row_indices{0, 0, 1, 2, 2, 2};
    std::vector<Index> const col_indices{0, 3, 2, 0, 1, 3};
    auto const values = make_matrix_values('m', row_indices, col_indices);
    std::vector<Index> const indices{0, 3, 2, 0, 1, 3};
    std::vector<Index> const offsets{0, 2, 3, 6};
    auto const matrix =
        make_csr(values, indices, offsets, out_basis.size(), arg_basis.size());

    VectorView<Scalar*> out_view(out.data(), out_basis);
    VectorView<Scalar const*> arg_view(arg.data(), arg_basis);

    rpp::ops::SparseMatrixVectorProduct<Strategy, rpp::sparse::CSRMatrix>{}(
        make_context(), out_view, arg_view, matrix);

    std::vector<Scalar> const expected{
        matrix_symbol('m', 0, 0) * symbol('x', arg_basis, 0, 0)
            + matrix_symbol('m', 0, 3) * symbol('x', arg_basis, 1, 2),
        matrix_symbol('m', 1, 2) * symbol('x', arg_basis, 1, 1),
        matrix_symbol('m', 2, 0) * symbol('x', arg_basis, 0, 0)
            + matrix_symbol('m', 2, 1) * symbol('x', arg_basis, 1, 0)
            + matrix_symbol('m', 2, 3) * symbol('x', arg_basis, 1, 2)
    };

    EXPECT_EQ(out, expected);
}

TEST_F(SparseMatrixVectorProductTests, AppliesCscMatrix)
{
    auto const out_basis_data = BasisData(2, 1);
    auto const arg_basis_data = BasisData(3, 1);
    auto const& out_basis = out_basis_data.basis;
    auto const& arg_basis = arg_basis_data.basis;

    auto out = make_tensor('o', out_basis);
    auto const arg = make_tensor('x', arg_basis);

    std::vector<Index> const row_indices{0, 2, 2, 1, 0, 2};
    std::vector<Index> const col_indices{0, 0, 1, 2, 3, 3};
    auto const values = make_matrix_values('m', row_indices, col_indices);
    std::vector<Index> const indices{0, 2, 2, 1, 0, 2};
    std::vector<Index> const offsets{0, 2, 3, 4, 6};
    auto const matrix =
        make_csc(values, indices, offsets, out_basis.size(), arg_basis.size());

    VectorView<Scalar*> out_view(out.data(), out_basis);
    VectorView<Scalar const*> arg_view(arg.data(), arg_basis);

    rpp::ops::SparseMatrixVectorProduct<Strategy, rpp::sparse::CSCMatrix>{}(
        make_context(), out_view, arg_view, matrix);

    std::vector<Scalar> const expected{
        matrix_symbol('m', 0, 0) * symbol('x', arg_basis, 0, 0)
            + matrix_symbol('m', 0, 3) * symbol('x', arg_basis, 1, 2),
        matrix_symbol('m', 1, 2) * symbol('x', arg_basis, 1, 1),
        matrix_symbol('m', 2, 0) * symbol('x', arg_basis, 0, 0)
            + matrix_symbol('m', 2, 1) * symbol('x', arg_basis, 1, 0)
            + matrix_symbol('m', 2, 3) * symbol('x', arg_basis, 1, 2)
    };

    EXPECT_EQ(out, expected);
}

TEST_F(SparseMatrixVectorProductTests, ScalesResult)
{
    auto const out_basis_data = BasisData(2, 1);
    auto const arg_basis_data = BasisData(3, 1);
    auto const& out_basis = out_basis_data.basis;
    auto const& arg_basis = arg_basis_data.basis;

    auto out = make_tensor('o', out_basis);
    auto const arg = make_tensor('x', arg_basis);

    std::vector<Index> const row_indices{0, 0, 1, 2, 2, 2};
    std::vector<Index> const col_indices{0, 3, 2, 0, 1, 3};
    auto const values = make_matrix_values('m', row_indices, col_indices);
    std::vector<Index> const indices{0, 3, 2, 0, 1, 3};
    std::vector<Index> const offsets{0, 2, 3, 6};
    auto const matrix =
        make_csr(values, indices, offsets, out_basis.size(), arg_basis.size());

    VectorView<Scalar*> out_view(out.data(), out_basis);
    VectorView<Scalar const*> arg_view(arg.data(), arg_basis);

    rpp::ops::SparseMatrixVectorProduct<Strategy, rpp::sparse::CSRMatrix>{}(
        make_context(), out_view, arg_view, matrix, constant(2));

    std::vector<Scalar> const expected{
        constant(2) * (matrix_symbol('m', 0, 0) * symbol('x', arg_basis, 0, 0)
            + matrix_symbol('m', 0, 3) * symbol('x', arg_basis, 1, 2)),
        constant(2) * (matrix_symbol('m', 1, 2) * symbol('x', arg_basis, 1, 1)),
        constant(2) * (matrix_symbol('m', 2, 0) * symbol('x', arg_basis, 0, 0)
            + matrix_symbol('m', 2, 1) * symbol('x', arg_basis, 1, 0)
            + matrix_symbol('m', 2, 3) * symbol('x', arg_basis, 1, 2))
    };

    EXPECT_EQ(out, expected);
}

TEST_F(SparseMatrixVectorProductTests, AppliesToLogicalViewRanges)
{
    auto const basis_data = BasisData(2, 2);
    auto const& basis = basis_data.basis;

    auto out = make_tensor('o', basis);
    auto const arg = make_tensor('x', basis);

    std::vector<Index> const row_indices{0, 0, 1, 1};
    std::vector<Index> const col_indices{0, 3, 1, 2};
    auto const values = make_matrix_values('m', row_indices, col_indices);
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

    auto expected = out;
    expected[1] = matrix_symbol('m', 0, 0) * symbol('x', basis, 2, 0)
        + matrix_symbol('m', 0, 3) * symbol('x', basis, 2, 3);
    expected[2] = matrix_symbol('m', 1, 1) * symbol('x', basis, 2, 1)
        + matrix_symbol('m', 1, 2) * symbol('x', basis, 2, 2);

    EXPECT_EQ(out, expected);
}

TEST_F(SparseMatrixVectorProductTests, KernelWrapperMatchesDirectOperation)
{
    static constexpr Index tensor_count = 2;

    auto const out_basis_data = BasisData(2, 1);
    auto const arg_basis_data = BasisData(3, 1);
    auto const& out_basis = out_basis_data.basis;
    auto const& arg_basis = arg_basis_data.basis;
    const Strategy strategy;

    auto actual = make_tensor('o', out_basis);
    auto second_actual = make_tensor('p', out_basis);
    actual.insert(actual.end(), second_actual.begin(), second_actual.end());

    auto expected = actual;

    auto arg = make_tensor('x', arg_basis);
    auto second_arg = make_tensor('y', arg_basis);
    arg.insert(arg.end(), second_arg.begin(), second_arg.end());

    std::vector<Index> const row_indices{0, 0, 1, 2, 2, 2};
    std::vector<Index> const col_indices{0, 3, 2, 0, 1, 3};
    auto const values = make_matrix_values('m', row_indices, col_indices);
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
                                                            constant(2));
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();

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
        op(ctx, out, rhs, matrix, constant(2));
    }

    EXPECT_EQ(actual, expected);
}
} // namespace
