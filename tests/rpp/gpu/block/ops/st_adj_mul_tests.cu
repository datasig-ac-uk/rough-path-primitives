#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/st_adj_mul.hpp>
#include <rpp/gpu/block/operations/basic/st_adj_mul.hpp>
#include <rpp/gpu/block/operations/basic/st_mul.hpp>
#include <rpp/gpu/block/operations/basic/tensor_pairing.hpp>

#include "gpu_block_test_helper.cuh"
#include "gpu_typed_adjoint_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockStAdjMulTypedTests
    : public rpp::tests::TypedGpuAdjointTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuAdjointTestBase<Config>;
    using typename Base::Basis;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using typename Base::PairingDeviceVector;
    using Base::expect_scalar_near;
    using Base::make_batch;
    using Base::make_identity_operator;
    using Base::make_zero_batch;

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
        auto const adj_err = rpp::ops::st_adj_mul(
            gpu_strategy,
            launch_config,
            Helper::device_tensor_batch(device_adjoint, basis),
            Helper::device_tensor_batch(device_op, basis),
            Helper::device_tensor_batch(device_arg, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(adj_err)) << adj_err.message();

        auto const mul_err = rpp::ops::st_mul(
            gpu_strategy,
            launch_config,
            Helper::device_tensor_batch(device_product, basis),
            Helper::device_tensor_batch(device_op, basis),
            Helper::device_tensor_batch(device_t, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(mul_err)) << mul_err.message();

        auto const lhs_err = rpp::ops::tensor_pairing(
            gpu_strategy,
            launch_config,
            Helper::device_scalar_batch(device_lhs_pairing),
            Helper::device_tensor_batch(device_t, basis),
            Helper::device_tensor_batch(device_adjoint, basis),
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
        expect_scalar_near(lhs_pairing[0], rhs_pairing[0]);
    }
};

TYPED_TEST_SUITE(GpuBlockStAdjMulTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockStAdjMulTypedTests, SatisfiesAdjointPairingCriterionOnGpu) {
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

TYPED_TEST(GpuBlockStAdjMulTypedTests,
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
        auto const err = rpp::ops::st_adj_mul(
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
        TestFixture::expect_tensor_near(actual, arg);
    }
}

TEST(GpuBlockStAdjMulTests, MatchesCpuForSingleElementBatches) {
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
        auto const err = rpp::ops::st_adj_mul(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            Helper::device_tensor_batch(device_op, basis),
            Helper::device_tensor_batch(device_arg, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err =
             rpp::ops::st_adj_mul(cpu_strategy,
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

TEST(GpuBlockStAdjMulTests, IdentityOperatorMatchesCpuForTruncatedView) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();

        auto expected = Helper::make_zero_batch(basis);
        auto actual = expected;
        auto op = Helper::make_zero_batch(basis);
        auto const arg = Helper::make_batch(7, basis, Helper::Scalar{0.01});
        op[0] = Helper::Scalar{1};

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_op(op);
        Helper::DeviceVector<Helper::Scalar> device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::st_adj_mul(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            rpp::make_tensor_batch(
                Helper::device_data(device_op),
                basis.size(),
                Helper::Degree{0},
                Helper::Degree{0}),
            Helper::device_tensor_batch(device_arg, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err = rpp::ops::st_adj_mul(
            cpu_strategy,
            Helper::CpuStrategy::LaunchConfig{},
            Helper::host_tensor_batch(expected, basis),
            rpp::make_tensor_batch(
                Helper::host_data(op),
                basis.size(),
                basis,
                Helper::Degree{0},
                Helper::Degree{0}),
            Helper::host_tensor_batch(arg, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
    }
}

} // namespace
