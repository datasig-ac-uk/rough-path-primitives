#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/ft_adj_rmul.hpp>
#include <rpp/gpu/block/operations/basic/ft_adj_rmul.hpp>
#include <rpp/gpu/block/operations/basic/ft_mul.hpp>
#include <rpp/gpu/block/operations/basic/tensor_pairing.hpp>

#include "gpu_block_test_helper.cuh"
#include "gpu_typed_adjoint_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockFtAdjRMulTypedTests
    : public rpp::tests::TypedGpuAdjointTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuAdjointTestBase<Config>;
    using typename Base::Basis;
    using typename Base::Degree;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using typename Base::Index;
    using typename Base::PairingDeviceVector;
    using Base::expect_scalar_near;
    using Base::make_batch;
    using Base::make_identity_operator;
    using Base::make_zero_batch;

    [[nodiscard]] static HostVector
    make_letter_operator(Basis const& basis, Index letter_index) {
        auto result = make_zero_batch(basis);
        result[static_cast<std::size_t>(basis.start_of_degree(1) + letter_index)] =
            rpp::tests::cast_scalar<typename Base::Scalar>(1.0f);
        return result;
    }

    [[nodiscard]] static HostVector
    expected_right_shift(Basis const& basis,
                         HostVector const& arg,
                         Index letter_index) {
        auto result = make_zero_batch(basis);
        for (Degree degree = 0; degree < basis.depth; ++degree) {
            const auto level_size = basis.size_of_degree(degree);
            const auto src_begin = basis.start_of_degree(degree + 1);
            const auto dst_begin = basis.start_of_degree(degree);
            for (Index idx = 0; idx < level_size; ++idx) {
                result[static_cast<std::size_t>(dst_begin + idx)] =
                    arg[static_cast<std::size_t>(
                        src_begin + idx * basis.width + letter_index)];
            }
        }
        return result;
    }

    static void expect_adjoint_pairing_identity(Basis const& basis,
                                                GpuStrategy const& gpu_strategy,
                                                HostVector const& op,
                                                HostVector const& t,
                                                HostVector const& arg) {
        auto adjoint = make_zero_batch(basis);
        auto product = make_zero_batch(basis);

        DeviceVector device_adjoint(adjoint);
        DeviceVector device_product(product);
        DeviceVector device_op(op);
        DeviceVector device_t(t);
        DeviceVector device_arg(arg);
        PairingDeviceVector device_lhs_pairing(1);
        PairingDeviceVector device_rhs_pairing(1);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const adj_err = rpp::ops::ft_adj_rmul(
            gpu_strategy,
            launch_config,
            Helper::device_tensor_batch(device_adjoint, basis),
            Helper::device_tensor_batch(device_op, basis),
            Helper::device_tensor_batch(device_arg, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(adj_err)) << adj_err.message();

        auto const mul_err = rpp::ops::ft_mul(
            gpu_strategy,
            launch_config,
            Helper::device_tensor_batch(device_product, basis),
            Helper::device_tensor_batch(device_t, basis),
            Helper::device_tensor_batch(device_op, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(mul_err)) << mul_err.message();

        auto const lhs_err = rpp::ops::tensor_pairing(
            gpu_strategy,
            launch_config,
            Helper::device_scalar_batch(device_lhs_pairing),
            Helper::device_tensor_batch(device_adjoint, basis),
            Helper::device_tensor_batch(device_t, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(lhs_err)) << lhs_err.message();

        auto const rhs_err = rpp::ops::tensor_pairing(
            gpu_strategy,
            launch_config,
            Helper::device_scalar_batch(device_rhs_pairing),
            Helper::device_tensor_batch(device_arg, basis),
            Helper::device_tensor_batch(device_product, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(rhs_err)) << rhs_err.message();

        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const lhs_pairing = Helper::copy_to_host(device_lhs_pairing);
        auto const rhs_pairing = Helper::copy_to_host(device_rhs_pairing);
        ASSERT_EQ(lhs_pairing.size(), std::size_t{1});
        ASSERT_EQ(rhs_pairing.size(), std::size_t{1});
        RPP_EXPECT_GPU_TYPED_SCALAR_NEAR(GpuBlockFtAdjRMulTypedTests, lhs_pairing[0], rhs_pairing[0]);
    }
};

TYPED_TEST_SUITE(GpuBlockFtAdjRMulTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockFtAdjRMulTypedTests, SatisfiesAdjointPairingCriterionOnGpu) {
    RPP_REQUIRE_CUDA_DEVICE();

    constexpr unsigned seeds[][3] = {
        {1, 2, 3},
        {5, 8, 13},
        {21, 34, 55},
        {89, 144, 233},
    };

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        for (auto const& triple : seeds) {
            auto const op = TestFixture::make_batch(triple[0], basis);
            auto const t = TestFixture::make_batch(triple[1], basis);
            auto const arg = TestFixture::make_batch(triple[2], basis);

            TestFixture::expect_adjoint_pairing_identity(
                basis, gpu_strategy, op, t, arg);
        }
    }
}

TYPED_TEST(GpuBlockFtAdjRMulTypedTests,
           IdentityOperatorReturnsArgumentForTruncatedView) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto actual = TestFixture::make_zero_batch(basis);
        auto const op = TestFixture::make_identity_operator(basis);
        auto const arg = TestFixture::make_batch(7, basis);

        typename TestFixture::DeviceVector device_actual(actual);
        typename TestFixture::DeviceVector device_op(op);
        typename TestFixture::DeviceVector device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_adj_rmul(
            gpu_strategy,
            std::move(launch_config),
            TestFixture::Helper::device_tensor_batch(device_actual, basis),
            rpp::make_tensor_batch(TestFixture::Helper::device_data(device_op),
                                   basis.size(),
                                   typename TestFixture::Degree{0},
                                   typename TestFixture::Degree{0}),
            TestFixture::Helper::device_tensor_batch(device_arg, basis),
            basis,
            TestFixture::Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        actual = TestFixture::Helper::copy_to_host(device_actual);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, arg);
    }
}

TYPED_TEST(GpuBlockFtAdjRMulTypedTests, LetterOperatorShiftsCoefficientsRight) {
    RPP_REQUIRE_CUDA_DEVICE();

    constexpr typename TestFixture::Index letter_index = 1;

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto actual = TestFixture::make_zero_batch(basis);
        auto const op = TestFixture::make_letter_operator(basis, letter_index);
        auto const arg = TestFixture::make_batch(11, basis);

        typename TestFixture::DeviceVector device_actual(actual);
        typename TestFixture::DeviceVector device_op(op);
        typename TestFixture::DeviceVector device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_adj_rmul(
            gpu_strategy,
            std::move(launch_config),
            TestFixture::Helper::device_tensor_batch(device_actual, basis),
            rpp::make_tensor_batch(TestFixture::Helper::device_data(device_op),
                                   basis.size(),
                                   typename TestFixture::Degree{1},
                                   typename TestFixture::Degree{1}),
            TestFixture::Helper::device_tensor_batch(device_arg, basis),
            basis,
            TestFixture::Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        actual = TestFixture::Helper::copy_to_host(device_actual);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, 
            actual,
            TestFixture::expected_right_shift(basis, arg, letter_index));
    }
}

TEST(GpuBlockFtAdjRMulTests, MatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();

        auto expected = Helper::make_zero_batch(basis);
        auto actual = expected;
        auto const op = Helper::make_batch(1, basis, Helper::Scalar{0.01});
        auto const arg = Helper::make_batch(2, basis, Helper::Scalar{0.01});

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_op(op);
        Helper::DeviceVector<Helper::Scalar> device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_adj_rmul(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            Helper::device_tensor_batch(device_op, basis),
            Helper::device_tensor_batch(device_arg, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err = rpp::ops::ft_adj_rmul(
            cpu_strategy,
            Helper::CpuStrategy::LaunchConfig{},
            Helper::host_tensor_batch(expected, basis),
            Helper::host_tensor_batch(op, basis),
            Helper::host_tensor_batch(arg, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
    }
}

} // namespace
