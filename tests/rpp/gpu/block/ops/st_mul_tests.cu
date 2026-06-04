#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/basic/st_mul.hpp>
#include <rpp/gpu/block/operations/basic/tensor_pairing.hpp>

#include "gpu_typed_st_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockStMulTypedTests
    : public rpp::tests::TypedGpuShuffleTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuShuffleTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Degree;
    using typename Base::DegreeRange;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using typename Base::PairingDeviceVector;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::make_batch;
    using Base::make_unit_tensor;
    using Base::reference_mul;

    static typename Base::Scalar scalar_from_accum(Accum value) {
        if constexpr (std::is_same_v<typename Base::Scalar, __half> ||
                      std::is_same_v<typename Base::Scalar, __nv_bfloat16>) {
            return rpp::tests::cast_scalar<typename Base::Scalar>(
                static_cast<float>(value));
        }
        else {
            return static_cast<typename Base::Scalar>(value);
        }
    }

    static constexpr double pairing_character_tolerance() {
        if constexpr (std::is_same_v<typename Base::Scalar, double>) {
            return 1e-7;
        }
        else if constexpr (std::is_same_v<typename Base::Scalar, float>) {
            return 2e-4;
        }
        else if constexpr (std::is_same_v<typename Base::Scalar, __half>) {
            return std::is_same_v<typename Base::Accum, float> ? 2e-3 : 2e-2;
        }
        else {
            return std::is_same_v<typename Base::Accum, float> ? 5e-3 : 4e-2;
        }
    }

    static HostVector make_group_like_argument(Basis const& basis,
                                               unsigned seed) {
        auto group_like = Base::make_zero_batch(basis);
        auto const lambda =
            static_cast<Accum>((static_cast<unsigned>(seed % 3u) + 1u) *
                               0.03125);

        group_like[0] = scalar_from_accum(Accum{1});
        Accum coeff{1};
        for (Degree degree = 1; degree <= basis.depth; ++degree) {
            coeff = coeff * lambda / static_cast<Accum>(degree);
            group_like[static_cast<std::size_t>(basis.start_of_degree(degree))] =
                scalar_from_accum(coeff);
        }
        return group_like;
    }

    static HostVector make_character_test_batch(Basis const& basis,
                                                unsigned seed) {
        auto result = Base::make_zero_batch(basis);
        for (std::size_t i = 0; i < result.size(); ++i) {
            auto const bucket =
                static_cast<unsigned>((seed * 13u +
                                       static_cast<unsigned>(i) * 7u) %
                                      5u);
            auto const value =
                static_cast<Accum>(bucket + 1u) * static_cast<Accum>(0.00390625);
            result[i] = scalar_from_accum(value);
        }
        return result;
    }

    static HostVector run_gpu_mul(Basis const& basis,
                                  GpuStrategy const& gpu_strategy,
                                  HostVector const& lhs,
                                  HostVector const& rhs,
                                  DegreeRange out_range,
                                  DegreeRange lhs_range,
                                  DegreeRange rhs_range,
                                  Accum beta = Accum{1}) {
        auto actual = Base::make_zero_batch(basis);

        DeviceVector device_actual(actual);
        DeviceVector device_lhs(lhs);
        DeviceVector device_rhs(rhs);
        auto lhs_batch = Helper::device_tensor_batch(device_lhs, basis);
        auto rhs_batch = Helper::device_tensor_batch(device_rhs, basis);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::st_mul(
            gpu_strategy,
            std::move(launch_config),
            rpp::make_tensor_batch(
                Helper::device_data(device_actual), basis.size(), out_range.min,
                out_range.max),
            rpp::make_tensor_batch(
                lhs_batch.data(), lhs_batch.layout(), lhs_range.min, lhs_range.max),
            rpp::make_tensor_batch(
                rhs_batch.data(), rhs_batch.layout(), rhs_range.min, rhs_range.max),
            basis,
            Helper::tensor_count,
            beta);
        if (!static_cast<bool>(err)) {
            ADD_FAILURE() << err.message();
            return actual;
        }
        auto const sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ADD_FAILURE() << "cudaDeviceSynchronize failed: "
                          << cudaGetErrorString(sync_err);
            return actual;
        }

        return Helper::copy_to_host(device_actual);
    }

    static void expect_shuffle_character_identity(
        Basis const& basis,
        GpuStrategy const& gpu_strategy,
        HostVector const& s,
        HostVector const& t,
        HostVector const& a) {
        auto product = Base::make_zero_batch(basis);

        DeviceVector device_product(product);
        DeviceVector device_s(s);
        DeviceVector device_t(t);
        DeviceVector device_a(a);
        PairingDeviceVector device_lhs_pairing(1);
        PairingDeviceVector device_s_pairing(1);
        PairingDeviceVector device_t_pairing(1);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;

        auto const mul_err = rpp::ops::st_mul(
            gpu_strategy,
            launch_config,
            Helper::device_tensor_batch(device_product, basis),
            Helper::device_tensor_batch(device_s, basis),
            Helper::device_tensor_batch(device_t, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(mul_err)) << mul_err.message();

        auto const lhs_err = rpp::ops::tensor_pairing(
            gpu_strategy,
            launch_config,
            Helper::device_scalar_batch(device_lhs_pairing),
            Helper::device_tensor_batch(device_product, basis),
            Helper::device_tensor_batch(device_a, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(lhs_err)) << lhs_err.message();

        auto const s_err = rpp::ops::tensor_pairing(
            gpu_strategy,
            launch_config,
            Helper::device_scalar_batch(device_s_pairing),
            Helper::device_tensor_batch(device_s, basis),
            Helper::device_tensor_batch(device_a, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(s_err)) << s_err.message();

        auto const t_err = rpp::ops::tensor_pairing(
            gpu_strategy,
            launch_config,
            Helper::device_scalar_batch(device_t_pairing),
            Helper::device_tensor_batch(device_t, basis),
            Helper::device_tensor_batch(device_a, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(t_err)) << t_err.message();

        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const lhs_pairing = Helper::copy_to_host(device_lhs_pairing);
        auto const s_pairing = Helper::copy_to_host(device_s_pairing);
        auto const t_pairing = Helper::copy_to_host(device_t_pairing);
        ASSERT_EQ(lhs_pairing.size(), std::size_t{1});
        ASSERT_EQ(s_pairing.size(), std::size_t{1});
        ASSERT_EQ(t_pairing.size(), std::size_t{1});

        auto const actual_d =
            rpp::tests::scalar_to_double(lhs_pairing[0]);
        auto const expected_d =
            rpp::tests::scalar_to_double(static_cast<Accum>(s_pairing[0]) *
                                         static_cast<Accum>(t_pairing[0]));
        auto const scale =
            std::max({1.0, std::abs(actual_d), std::abs(expected_d)});
        auto const tolerance = pairing_character_tolerance() * scale;
        EXPECT_NEAR(actual_d, expected_d, tolerance);
    }
};

TYPED_TEST_SUITE(GpuBlockStMulTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockStMulTypedTests, MatchesHostReferenceForSingleElementBatches) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const beta = typename TestFixture::Accum{1.5};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const lhs = TestFixture::make_batch(1, basis);
        auto const rhs = TestFixture::make_batch(2, basis);

        auto const actual = TestFixture::run_gpu_mul(
            basis, gpu_strategy, lhs, rhs, TestFixture::full_range(basis),
            TestFixture::full_range(basis), TestFixture::full_range(basis), beta);
        auto const expected = TestFixture::reference_mul(basis, lhs, rhs, beta);
        TestFixture::expect_tensor_near(actual, expected);
    }
}

TYPED_TEST(GpuBlockStMulTypedTests, UnitActsAsTwoSidedIdentity) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const unit = TestFixture::make_unit_tensor(basis);
        auto const arg = TestFixture::make_batch(11, basis);

        auto const left_actual = TestFixture::run_gpu_mul(
            basis, gpu_strategy, unit, arg, TestFixture::full_range(basis),
            TestFixture::full_range(basis), TestFixture::full_range(basis));
        auto const right_actual = TestFixture::run_gpu_mul(
            basis, gpu_strategy, arg, unit, TestFixture::full_range(basis),
            TestFixture::full_range(basis), TestFixture::full_range(basis));

        TestFixture::expect_tensor_near(left_actual, arg);
        TestFixture::expect_tensor_near(right_actual, arg);
    }
}

TYPED_TEST(GpuBlockStMulTypedTests, RespectsTruncatedOperandAndOutputViews) {
    RPP_REQUIRE_CUDA_DEVICE();

    using Range = typename TestFixture::DegreeRange;

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const lhs = TestFixture::make_batch(3, basis);
        auto const rhs = TestFixture::make_batch(5, basis);
        auto const out_range = Range{1, std::min<typename TestFixture::Degree>(
                                            3, basis.depth)};
        auto const lhs_range = Range{1, basis.depth};
        auto const rhs_range = Range{0, std::min<typename TestFixture::Degree>(
                                            2, basis.depth)};

        auto const actual = TestFixture::run_gpu_mul(
            basis, gpu_strategy, lhs, rhs, out_range, lhs_range, rhs_range);
        auto const expected = TestFixture::reference_mul(
            basis, lhs, rhs, out_range, lhs_range, rhs_range);
        TestFixture::expect_tensor_near(actual, expected);
    }
}

TYPED_TEST(GpuBlockStMulTypedTests, PairingTurnsShuffleProductIntoCharacter) {
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
            auto const s =
                TestFixture::make_character_test_batch(basis, triple[0]);
            auto const t =
                TestFixture::make_character_test_batch(basis, triple[1]);
            auto const a = TestFixture::make_group_like_argument(basis, triple[2]);

            TestFixture::expect_shuffle_character_identity(
                basis, gpu_strategy, s, t, a);
        }
    }
}

} // namespace
