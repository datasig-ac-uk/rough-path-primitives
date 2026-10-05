#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/basis/hall_basis.hpp>
#include <rpp/gpu/block/operations/basic/lie_to_tensor.hpp>
#include <rpp/gpu/block/operations/basic/tensor_to_lie.hpp>
#include <rpp/views/views.hpp>

#include "../../../lie_tensor_conversion_test_helper.hpp"
#include "gpu_block_test_helper.cuh"
#include "gpu_typed_adjoint_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockLieTensorConversionTests : public rpp::tests::TypedGpuAdjointTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuAdjointTestBase<Config>;
    using typename Base::Scalar;
    using typename Base::Basis;
    using typename Base::Index;
    using typename Base::Degree;
    using typename Base::Helper;
    using typename Base::GpuStrategy;
    using typename Base::HostVector;
    using typename Base::DeviceVector;
    using Architecture = typename GpuStrategy::Architecture;

    template <bool LieToTensor, rpp::sparse::MatrixFormat Format>
    static void check_initialized_output() {
        using L2T = rpp::ops::L2TImplementationType;
        using T2L = rpp::ops::T2LImplementationType;
        using Op = std::conditional_t<LieToTensor,
            rpp::ops::LieToTensor<GpuStrategy, Format == rpp::sparse::CSRMatrix
                ? L2T::CSRSparseMatrix : L2T::CSCSparseMatrix>,
            rpp::ops::TensorToLie<GpuStrategy, Format == rpp::sparse::CSRMatrix
                ? T2L::CSRSparseMatrix : T2L::CSCSparseMatrix>>;
        struct BasisConfig { Degree width, depth; };
        BasisConfig const configs[] = {{2, 2}, {4, 5}};
        for (auto const& config : configs) {
            SCOPED_TRACE(testing::Message() << "width=" << config.width
                                           << ", depth=" << config.depth);
            auto const tensor_data = typename Helper::BasisData(config.width, config.depth);
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
                DeviceVector device_out(initial_out.begin(), initial_out.end());
                DeviceVector device_arg(arg.begin(), arg.end());
                auto out_data = Helper::device_data(device_out) + 1;
                auto arg_data = Helper::device_data(device_arg);
                thrust::device_vector<Scalar> values(matrix_data.values.begin(), matrix_data.values.end());
                thrust::device_vector<Index> indices(matrix_data.indices.begin(), matrix_data.indices.end());
                thrust::device_vector<Index> offsets(matrix_data.offsets.begin(), matrix_data.offsets.end());
                auto value_data = Helper::device_data(values);
                auto index_data = Helper::device_data(indices);
                auto offset_data = Helper::device_data(offsets);
                using Matrix = rpp::sparse::GradedMatrixView<Format,
                    decltype(value_data), decltype(index_data), decltype(offset_data)>;
                Matrix const matrix(value_data, index_data, offset_data,
                    static_cast<Index>(values.size()),
                    Format == rpp::sparse::CSRMatrix ? output_size : input_size,
                    Format == rpp::sparse::CSRMatrix ? input_size : output_size);
                auto const strategy = GpuStrategy{Helper::block_size};
                rpp::gpu::DeviceLaunchConfig launch;
                launch.stream = nullptr;
                auto const tensor_batch = rpp::make_tensor_batch(
                    LieToTensor ? out_data : arg_data, tensor_size, Degree{0}, config.depth);
                auto const lie_batch = rpp::make_lie_batch(
                    std::move(LieToTensor ? arg_data : out_data), Index{lie_size},
                    Degree{0}, config.depth);
                auto const err = [&] {
                    if constexpr (LieToTensor) {
                        return rpp::ops::lie_to_tensor(strategy, launch,
                            tensor_batch, lie_batch, tensor_basis, lie_basis,
                            Index{1}, matrix);
                    }
                    else {
                        return rpp::ops::tensor_to_lie(strategy, launch,
                            lie_batch, tensor_batch, lie_basis, tensor_basis,
                            Index{1}, matrix);
                    }
                }();
                ASSERT_TRUE(static_cast<bool>(err)) << err.message();
                RPP_CUDA_ASSERT(cudaDeviceSynchronize());
                auto const actual = Helper::copy_to_host(device_out);
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

using ConversionTypes = testing::Types<
    rpp::tests::TypedScalarAccumConfig<float, float>,
    rpp::tests::TypedScalarAccumConfig<double, double>>;

TYPED_TEST_SUITE(GpuBlockLieTensorConversionTests, ConversionTypes);

TYPED_TEST(GpuBlockLieTensorConversionTests, LieToTensorCSRInitializesEveryCoefficient) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_initialized_output<true,
        rpp::sparse::CSRMatrix>();
}

TYPED_TEST(GpuBlockLieTensorConversionTests, LieToTensorCSCInitializesEveryCoefficient) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_initialized_output<true,
        rpp::sparse::CSCMatrix>();
}

TYPED_TEST(GpuBlockLieTensorConversionTests, TensorToLieCSRInitializesEveryCoefficient) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_initialized_output<false,
        rpp::sparse::CSRMatrix>();
}

TYPED_TEST(GpuBlockLieTensorConversionTests, TensorToLieCSCInitializesEveryCoefficient) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_initialized_output<false,
        rpp::sparse::CSCMatrix>();
}

} // namespace
