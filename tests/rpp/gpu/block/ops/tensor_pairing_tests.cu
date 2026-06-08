#include <algorithm>

#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/basic/tensor_pairing.hpp>

#include "gpu_typed_adjoint_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockTensorPairingTypedTests
    : public rpp::tests::TypedGpuAdjointTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuAdjointTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Degree;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using typename Base::PairingDeviceVector;
    using Base::expect_scalar_near;
    using Base::make_batch;

    struct DegreeRange {
        Degree min;
        Degree max;
    };

    static HostVector linear_combo(HostVector const& lhs,
                                   Accum lhs_scale,
                                   HostVector const& rhs,
                                   Accum rhs_scale) {
        HostVector result(lhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            auto const value = lhs_scale * static_cast<Accum>(lhs[i]) +
                               rhs_scale * static_cast<Accum>(rhs[i]);
            if constexpr (std::is_same_v<typename Base::Scalar, __half> ||
                          std::is_same_v<typename Base::Scalar, __nv_bfloat16>) {
                result[i] = rpp::tests::cast_scalar<typename Base::Scalar>(
                    static_cast<float>(value));
            }
            else {
                result[i] = static_cast<typename Base::Scalar>(value);
            }
        }
        return result;
    }

    static Accum reference_pairing(Basis const& basis,
                                   HostVector const& functional,
                                   HostVector const& arg,
                                   DegreeRange functional_range,
                                   DegreeRange arg_range) {
        auto const min_degree =
            std::max(functional_range.min, arg_range.min);
        auto const max_degree =
            std::min(functional_range.max, arg_range.max);
        if (min_degree > max_degree) {
            return Accum{0};
        }

        Accum result{0};
        for (auto idx = basis.start_of_degree(min_degree);
             idx < basis.end_of_degree(max_degree);
             ++idx) {
            result += static_cast<Accum>(
                          functional[static_cast<std::size_t>(idx)]) *
                      static_cast<Accum>(arg[static_cast<std::size_t>(idx)]);
        }
        return result;
    }

    static Accum run_gpu_pairing(Basis const& basis,
                                 GpuStrategy const& gpu_strategy,
                                 HostVector const& functional,
                                 HostVector const& arg,
                                 DegreeRange functional_range,
                                 DegreeRange arg_range) {
        DeviceVector device_functional(functional);
        DeviceVector device_arg(arg);
        PairingDeviceVector device_actual(1, Accum{0});

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::tensor_pairing(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_scalar_batch(device_actual),
            rpp::make_tensor_batch(Helper::device_data(device_functional),
                                   basis.size(),
                                   functional_range.min,
                                   functional_range.max),
            rpp::make_tensor_batch(Helper::device_data(device_arg),
                                   basis.size(),
                                   arg_range.min,
                                   arg_range.max),
            basis,
            Helper::tensor_count);
        if (!static_cast<bool>(err)) {
            ADD_FAILURE() << err.message();
            return Accum{0};
        }
        auto const sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ADD_FAILURE() << "cudaDeviceSynchronize failed: "
                          << cudaGetErrorString(sync_err);
            return Accum{0};
        }

        auto const actual = Helper::copy_to_host(device_actual);
        EXPECT_EQ(actual.size(), std::size_t{1});
        return actual[0];
    }
};

TYPED_TEST_SUITE(GpuBlockTensorPairingTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockTensorPairingTypedTests, IsSymmetric) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};

        auto const lhs = TestFixture::make_batch(1, basis);
        auto const rhs = TestFixture::make_batch(2, basis);
        auto const full = typename TestFixture::DegreeRange{0, basis.depth};

        auto const lhs_rhs = TestFixture::run_gpu_pairing(
            basis, gpu_strategy, lhs, rhs, full, full);
        auto const rhs_lhs = TestFixture::run_gpu_pairing(
            basis, gpu_strategy, rhs, lhs, full, full);

        RPP_EXPECT_GPU_TYPED_SCALAR_NEAR(TestFixture, lhs_rhs, rhs_lhs);
    }
}

TYPED_TEST(GpuBlockTensorPairingTypedTests, IsLinearInEachArgument) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const alpha = typename TestFixture::Accum{0.75};
    auto const beta = typename TestFixture::Accum{-1.25};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};

        auto const x = TestFixture::make_batch(3, basis);
        auto const y = TestFixture::make_batch(4, basis);
        auto const z = TestFixture::make_batch(5, basis);
        auto const combo = TestFixture::linear_combo(x, alpha, y, beta);
        auto const full = typename TestFixture::DegreeRange{0, basis.depth};

        auto const lhs = TestFixture::run_gpu_pairing(
            basis, gpu_strategy, combo, z, full, full);
        auto const xz = TestFixture::run_gpu_pairing(
            basis, gpu_strategy, x, z, full, full);
        auto const yz = TestFixture::run_gpu_pairing(
            basis, gpu_strategy, y, z, full, full);
        auto const expected = alpha * static_cast<typename TestFixture::Accum>(xz) +
                              beta * static_cast<typename TestFixture::Accum>(yz);
        RPP_EXPECT_GPU_TYPED_SCALAR_NEAR(TestFixture, lhs, expected);

        auto const rhs = TestFixture::run_gpu_pairing(
            basis, gpu_strategy, z, combo, full, full);
        auto const zx = TestFixture::run_gpu_pairing(
            basis, gpu_strategy, z, x, full, full);
        auto const zy = TestFixture::run_gpu_pairing(
            basis, gpu_strategy, z, y, full, full);
        auto const expected_rhs =
            alpha * static_cast<typename TestFixture::Accum>(zx) +
            beta * static_cast<typename TestFixture::Accum>(zy);
        RPP_EXPECT_GPU_TYPED_SCALAR_NEAR(TestFixture, rhs, expected_rhs);
    }
}

TYPED_TEST(GpuBlockTensorPairingTypedTests, RespectsTruncatedOperandViews) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};

        auto const functional = TestFixture::make_batch(6, basis);
        auto const arg = TestFixture::make_batch(7, basis);

        auto const functional_range =
            typename TestFixture::DegreeRange{1, basis.depth};
        auto const arg_range = typename TestFixture::DegreeRange{
            static_cast<typename TestFixture::Degree>(
                std::min<typename TestFixture::Degree>(2, basis.depth)),
            basis.depth};
        auto const expected = TestFixture::reference_pairing(
            basis, functional, arg, functional_range, arg_range);
        auto const actual = TestFixture::run_gpu_pairing(
            basis, gpu_strategy, functional, arg, functional_range, arg_range);
        RPP_EXPECT_GPU_TYPED_SCALAR_NEAR(TestFixture, actual, expected);

        if (basis.depth >= 1) {
            auto const disjoint_functional =
                typename TestFixture::DegreeRange{0, 0};
            auto const disjoint_arg =
                typename TestFixture::DegreeRange{1, basis.depth};
            auto const zero = TestFixture::run_gpu_pairing(
                basis,
                gpu_strategy,
                functional,
                arg,
                disjoint_functional,
                disjoint_arg);
            RPP_EXPECT_GPU_TYPED_SCALAR_NEAR(TestFixture, 
                zero, typename TestFixture::Accum{0});
        }
    }
}

} // namespace
