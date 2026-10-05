#include <gtest/gtest.h>

#include "../../../shuffle_adjoint_test_helper.hpp"

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
    using Scalar = typename Config::Scalar;
    using Accum = typename Config::Accum;
    using typename Base::Basis;
    using typename Base::DegreeRange;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using typename Base::PairingDeviceVector;
    using Base::expect_scalar_near;
    using Base::make_batch;
    using Base::make_identity_operator;
    using Base::make_zero_batch;

    static constexpr float pairing_identity_input_scale() noexcept {
        if constexpr (std::is_same_v<Scalar, __nv_bfloat16> &&
                      std::is_same_v<Accum, float>) {
            return 0.5f;
        }
        else {
            return 1.0f;
        }
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
        RPP_EXPECT_GPU_TYPED_SCALAR_NEAR(GpuBlockStAdjMulTypedTests, lhs_pairing[0], rhs_pairing[0]);
    }
    static HostVector run_gpu_adj_mul(
        Basis const& basis, GpuStrategy const& strategy,
        HostVector const& initial_out, HostVector const& op, HostVector const& arg,
        DegreeRange out_range, DegreeRange op_range, DegreeRange arg_range) {
        DeviceVector device_out(initial_out), device_op(op), device_arg(arg);
        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::st_adj_mul(
            strategy, std::move(launch_config),
            rpp::make_tensor_batch(Helper::device_data(device_out), basis.size(),
                                   out_range.min, out_range.max),
            rpp::make_tensor_batch(Helper::device_data(device_op), basis.size(),
                                   op_range.min, op_range.max),
            rpp::make_tensor_batch(Helper::device_data(device_arg), basis.size(),
                                   arg_range.min, arg_range.max),
            basis, Helper::tensor_count);
        if (!static_cast<bool>(err)) {
            ADD_FAILURE() << err.message();
            return initial_out;
        }
        auto const sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ADD_FAILURE() << cudaGetErrorString(sync_err);
            return initial_out;
        }
        return Helper::copy_to_host(device_out);
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
        auto const input_scale = TestFixture::pairing_identity_input_scale();

        for (auto const& triple : seeds) {
            auto const op = TestFixture::make_batch(triple[0], basis, input_scale);
            auto const t = TestFixture::make_batch(triple[1], basis, input_scale);
            auto const arg = TestFixture::make_batch(triple[2], basis, input_scale);

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
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, arg);
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

TYPED_TEST(GpuBlockStAdjMulTypedTests, OverwritesEveryOutputDegreeForRestrictedOperands) {
    RPP_REQUIRE_CUDA_DEVICE();

    using Range = typename TestFixture::DegreeRange;
    for (auto const width : {2, 4}) {
        SCOPED_TRACE(width);
        auto const basis_data = typename TestFixture::Helper::BasisData(width, 4);
        auto const& basis = basis_data.basis;
        auto const strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};
        auto const initial_out = TestFixture::make_batch(61, basis);
        auto const op = rpp::tests::make_sparse_shuffle_adjoint_operand(
            TestFixture::make_batch(62, basis), basis);
        auto const arg = rpp::tests::make_sparse_shuffle_adjoint_operand(
            TestFixture::make_batch(63, basis), basis);

        for (auto const& ranges : rpp::tests::shuffle_adjoint_range_cases<Range>()) {
            SCOPED_TRACE(ranges.name);
            auto const actual = TestFixture::run_gpu_adj_mul(
                basis, strategy, initial_out, op, arg,
                ranges.out, ranges.op, ranges.arg);
            auto const coefficients = rpp::tests::reference_shuffle_adjoint<
                typename TestFixture::Accum>(
                basis, op, arg, ranges.out, ranges.op, ranges.arg);
            auto expected = initial_out;
            auto const begin = basis.start_of_degree(ranges.out.min);
            auto const end = basis.end_of_degree(ranges.out.max);
            for (auto idx = begin; idx < end; ++idx) {
                expected[idx] = static_cast<typename TestFixture::Scalar>(coefficients[idx]);
            }
            RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
        }
    }
}

} // namespace
