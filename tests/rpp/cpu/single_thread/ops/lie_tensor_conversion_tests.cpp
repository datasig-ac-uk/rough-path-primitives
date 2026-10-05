#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/basis/hall_basis.hpp>
#include <rpp/cpu/single_thread/operations/basic/lie_to_tensor.hpp>
#include <rpp/cpu/single_thread/operations/basic/tensor_to_lie.hpp>
#include <rpp/views/views.hpp>

#include "../../../lie_tensor_conversion_test_helper.hpp"
#include "cpu_typed_ft_ops_test_helper.hpp"

namespace {

template <typename Config>
class CpuLieTensorConversionTests : public rpp::tests::TypedCpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuFreeTensorOpTestBase<Config>;
    using typename Base::Scalar;
    using typename Base::Basis;
    using typename Base::Index;
    using typename Base::Degree;
    using typename Base::Strategy;
    using Architecture = rpp::arch::NativeArchitecture;

    template <bool LieToTensor, rpp::sparse::MatrixFormat Format>
    static void check_initialized_output() {
        using L2T = rpp::ops::L2TImplementationType;
        using T2L = rpp::ops::T2LImplementationType;
        using Op = std::conditional_t<LieToTensor,
            rpp::ops::LieToTensor<Strategy, Format == rpp::sparse::CSRMatrix
                ? L2T::CSRSparseMatrix : L2T::CSCSparseMatrix>,
            rpp::ops::TensorToLie<Strategy, Format == rpp::sparse::CSRMatrix
                ? T2L::CSRSparseMatrix : T2L::CSCSparseMatrix>>;
        struct BasisConfig { Degree width, depth; };
        BasisConfig const configs[] = {{2, 2}, {4, 5}};
        for (auto const& config : configs) {
            SCOPED_TRACE(testing::Message() << "width=" << config.width
                                           << ", depth=" << config.depth);
            auto const tensor_data = typename Base::BasisData(config.width, config.depth);
            auto const& tensor_basis = tensor_data.basis;
            auto const hall = rpp::basis::HallBasis<Architecture>(config.width, config.depth);
            auto const lie_basis = hall.to_lie_basis();
            // Lie storage includes the unused degree-zero slot. Full views are
            // used throughout; no truncation behavior is assumed by these tests.
            auto const tensor_size = tensor_basis.true_size();
            auto const lie_size = lie_basis.true_size();
            auto const input_size = LieToTensor ? lie_size : tensor_size;
            auto const output_size = LieToTensor ? tensor_size : lie_size;
            auto const bracket = lie_basis.find_bracket(1, 2, 2);
            auto const xy = tensor_basis.start_of_degree(2) + 1;
            auto const yx = tensor_basis.start_of_degree(2) + config.width;
            ASSERT_NE(bracket, Index{0});
            for (auto scenario : {rpp::tests::ConversionInput::Nonzero,
                                  rpp::tests::ConversionInput::Zero,
                                  rpp::tests::ConversionInput::EmptyMatrix,
                                  rpp::tests::ConversionInput::Cancelling}) {
                if (LieToTensor && scenario == rpp::tests::ConversionInput::Cancelling) {
                    continue;
                }
                SCOPED_TRACE(static_cast<int>(scenario));
                auto const empty = scenario == rpp::tests::ConversionInput::EmptyMatrix;
                auto const matrix_data = rpp::tests::conversion_matrix_data<Format, Scalar>(
                    LieToTensor, tensor_size, lie_size, bracket, xy, yx, empty);
                std::vector<Scalar> arg(input_size, Scalar{0.125});
                arg[1] = Scalar{0.5};
                arg[2] = Scalar{-0.25};
                if (LieToTensor) {
                    arg[bracket] = Scalar{0.75};
                }
                else {
                    arg[xy] = Scalar{0.75};
                    arg[yx] = scenario == rpp::tests::ConversionInput::Cancelling
                        ? Scalar{0.75} : Scalar{-0.25};
                }
                if (scenario == rpp::tests::ConversionInput::Zero) {
                    std::fill(arg.begin(), arg.end(), Scalar{0});
                }
                auto const expected = rpp::tests::conversion_expected(
                    LieToTensor, arg, output_size, bracket, xy, yx, empty);
                // Guards detect overruns; NaNs expose any coefficient not written.
                std::vector<Scalar> initial_out(output_size + 2,
                                               std::numeric_limits<Scalar>::quiet_NaN());
                initial_out.front() = Scalar{7};
                initial_out.back() = Scalar{7};
                auto actual = initial_out;
                using Matrix = rpp::sparse::GradedMatrixView<Format,
                    Scalar const*, Index const*, Index const*>;
                Matrix const matrix(matrix_data.values.data(), matrix_data.indices.data(),
                    matrix_data.offsets.data(), static_cast<Index>(matrix_data.values.size()),
                    Format == rpp::sparse::CSRMatrix ? output_size : input_size,
                    Format == rpp::sparse::CSRMatrix ? input_size : output_size);
                auto const ctx = Base::make_context();
                if constexpr (LieToTensor) {
                    rpp::DenseTensorView<Scalar*, Basis> out(actual.data() + 1, tensor_basis);
                    rpp::DenseLieView<Scalar const*, decltype(lie_basis)> in(arg.data(), lie_basis);
                    Op{}(ctx, out, in, matrix);
                }
                else {
                    rpp::DenseLieView<Scalar*, decltype(lie_basis)> out(actual.data() + 1, lie_basis);
                    rpp::DenseTensorView<Scalar const*, Basis> in(arg.data(), tensor_basis);
                    Op{}(ctx, out, in, matrix);
                }
                ASSERT_EQ(actual.size(), expected.size() + 2);
                EXPECT_EQ(static_cast<double>(actual.front()), 7.0);
                EXPECT_EQ(static_cast<double>(actual.back()), 7.0);
                for (std::size_t i = 0; i < expected.size(); ++i) {
                    // Exact dyadic results allow equality, which also rejects NaNs.
                    EXPECT_EQ(static_cast<double>(actual[i + 1]),
                              static_cast<double>(expected[i])) << "coefficient " << i;
                }
            }
        }
    }
};

using ConversionTypes = rpp::tests::TypedCpuFreeTensorTestTypes;

TYPED_TEST_SUITE(CpuLieTensorConversionTests, ConversionTypes);

TYPED_TEST(CpuLieTensorConversionTests, LieToTensorCSRInitializesEveryCoefficient) {
    TestFixture::template check_initialized_output<true,
        rpp::sparse::CSRMatrix>();
}

TYPED_TEST(CpuLieTensorConversionTests, LieToTensorCSCInitializesEveryCoefficient) {
    TestFixture::template check_initialized_output<true,
        rpp::sparse::CSCMatrix>();
}

TYPED_TEST(CpuLieTensorConversionTests, TensorToLieCSRInitializesEveryCoefficient) {
    TestFixture::template check_initialized_output<false,
        rpp::sparse::CSRMatrix>();
}

TYPED_TEST(CpuLieTensorConversionTests, TensorToLieCSCInitializesEveryCoefficient) {
    TestFixture::template check_initialized_output<false,
        rpp::sparse::CSCMatrix>();
}

} // namespace
