#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/linalg/vector_scalar_multiply.hpp>

#include "gpu_typed_vector_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockVectorScalarMultiplyTypedTests
    : public rpp::tests::TypedGpuVectorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuVectorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::make_batch;

    static HostVector reference_scalar_multiply(
        HostVector const& vec,
        Basis const& basis,
        typename Base::DegreeRange range,
        Accum scalar) {
        auto result = vec;
        for (auto idx = basis.start_of_degree(range.min);
             idx < basis.end_of_degree(range.max);
             ++idx) {
            auto const i = static_cast<std::size_t>(idx);
            result[i] =
                Base::scalar_from_accum(static_cast<Accum>(result[i]) * scalar);
        }
        return result;
    }

    static HostVector run_gpu_scalar_multiply(
        Basis const& basis,
        GpuStrategy const& gpu_strategy,
        HostVector const& vec,
        typename Base::DegreeRange range,
        Accum scalar) {
        DeviceVector device_vec(vec);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::vector_scalar_multiply(
            gpu_strategy,
            std::move(launch_config),
            rpp::make_graded_vector_batch(Helper::device_data(device_vec),
                                          basis.size(),
                                          basis,
                                          range.min,
                                          range.max),
            basis,
            Helper::tensor_count,
            scalar);
        if (!static_cast<bool>(err)) {
            ADD_FAILURE() << err.message();
            return vec;
        }
        auto const sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ADD_FAILURE() << "cudaDeviceSynchronize failed: "
                          << cudaGetErrorString(sync_err);
            return vec;
        }

        return Helper::copy_to_host(device_vec);
    }
};

TYPED_TEST_SUITE(GpuBlockVectorScalarMultiplyTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockVectorScalarMultiplyTypedTests, MatchesReferenceOnFullView) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const scalar = typename TestFixture::Accum{-1.125};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const vec = TestFixture::make_batch(1, basis);

        auto const actual = TestFixture::run_gpu_scalar_multiply(
            basis, gpu_strategy, vec, TestFixture::full_range(basis), scalar);
        auto const expected = TestFixture::reference_scalar_multiply(
            vec, basis, TestFixture::full_range(basis), scalar);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

TYPED_TEST(GpuBlockVectorScalarMultiplyTypedTests, ZeroScalarZerosActiveSlice) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const vec = TestFixture::make_batch(2, basis);
        auto const max_degree =
            static_cast<typename TestFixture::Degree>(std::min<int>(basis.depth, 1));
        auto const range = typename TestFixture::DegreeRange{0, max_degree};

        auto const actual = TestFixture::run_gpu_scalar_multiply(
            basis, gpu_strategy, vec, range, typename TestFixture::Accum{0});
        auto const expected = TestFixture::reference_scalar_multiply(
            vec, basis, range, typename TestFixture::Accum{0});
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

TYPED_TEST(GpuBlockVectorScalarMultiplyTypedTests, RespectsTruncatedView) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const scalar = typename TestFixture::Accum{0.625};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        if (basis.depth < 1) {
            continue;
        }
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const vec = TestFixture::make_batch(3, basis);
        auto const range = typename TestFixture::DegreeRange{1, basis.depth};

        auto const actual = TestFixture::run_gpu_scalar_multiply(
            basis, gpu_strategy, vec, range, scalar);
        auto const expected = TestFixture::reference_scalar_multiply(
            vec, basis, range, scalar);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

} // namespace
