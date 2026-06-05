#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/linalg/vector_add.hpp>

#include "gpu_typed_vector_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockVectorAddTypedTests
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
    using Base::is_empty;
    using Base::linear_combo;
    using Base::make_batch;
    using Base::overlap_range;

    static HostVector reference_add(HostVector const& out,
                                    HostVector const& lhs,
                                    HostVector const& rhs,
                                    Basis const& basis,
                                    typename Base::DegreeRange out_range,
                                    typename Base::DegreeRange lhs_range,
                                    typename Base::DegreeRange rhs_range,
                                    Accum alpha,
                                    Accum beta) {
        auto result = out;
        auto const overlap = overlap_range(out_range, overlap_range(lhs_range, rhs_range));
        if (is_empty(overlap)) {
            return result;
        }

        for (auto idx = basis.start_of_degree(overlap.min);
             idx < basis.end_of_degree(overlap.max);
             ++idx) {
            auto const i = static_cast<std::size_t>(idx);
            auto const value = alpha * static_cast<Accum>(lhs[i]) +
                               beta * static_cast<Accum>(rhs[i]);
            result[i] = Base::scalar_from_accum(value);
        }
        return result;
    }

    static HostVector run_gpu_add(Basis const& basis,
                                  GpuStrategy const& gpu_strategy,
                                  HostVector const& out,
                                  HostVector const& lhs,
                                  HostVector const& rhs,
                                  typename Base::DegreeRange out_range,
                                  typename Base::DegreeRange lhs_range,
                                  typename Base::DegreeRange rhs_range,
                                  Accum alpha,
                                  Accum beta) {
        DeviceVector device_out(out);
        DeviceVector device_lhs(lhs);
        DeviceVector device_rhs(rhs);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::vector_add(
            gpu_strategy,
            std::move(launch_config),
            rpp::make_graded_vector_batch(Helper::device_data(device_out),
                                          basis.size(),
                                          basis,
                                          out_range.min,
                                          out_range.max),
            rpp::make_graded_vector_batch(Helper::device_data(device_lhs),
                                          basis.size(),
                                          basis,
                                          lhs_range.min,
                                          lhs_range.max),
            rpp::make_graded_vector_batch(Helper::device_data(device_rhs),
                                          basis.size(),
                                          basis,
                                          rhs_range.min,
                                          rhs_range.max),
            basis,
            Helper::tensor_count,
            alpha,
            beta);
        if (!static_cast<bool>(err)) {
            ADD_FAILURE() << err.message();
            return out;
        }
        auto const sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ADD_FAILURE() << "cudaDeviceSynchronize failed: "
                          << cudaGetErrorString(sync_err);
            return out;
        }

        return Helper::copy_to_host(device_out);
    }
};

TYPED_TEST_SUITE(GpuBlockVectorAddTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockVectorAddTypedTests, MatchesReferenceOnFullView) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const alpha = typename TestFixture::Accum{0.75};
    auto const beta = typename TestFixture::Accum{-1.25};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const out = TestFixture::make_batch(1, basis);
        auto const lhs = TestFixture::make_batch(2, basis);
        auto const rhs = TestFixture::make_batch(3, basis);

        auto const actual = TestFixture::run_gpu_add(
            basis,
            gpu_strategy,
            out,
            lhs,
            rhs,
            TestFixture::full_range(basis),
            TestFixture::full_range(basis),
            TestFixture::full_range(basis),
            alpha,
            beta);
        auto const expected = TestFixture::reference_add(
            out,
            lhs,
            rhs,
            basis,
            TestFixture::full_range(basis),
            TestFixture::full_range(basis),
            TestFixture::full_range(basis),
            alpha,
            beta);
        TestFixture::expect_tensor_near(actual, expected);
    }
}

TYPED_TEST(GpuBlockVectorAddTypedTests, EqualsLinearCombinationWhenOutputStartsZero) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const alpha = typename TestFixture::Accum{1.5};
    auto const beta = typename TestFixture::Accum{0.5};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const zero = TestFixture::make_zero_batch(basis);
        auto const lhs = TestFixture::make_batch(4, basis);
        auto const rhs = TestFixture::make_batch(5, basis);

        auto const actual = TestFixture::run_gpu_add(
            basis,
            gpu_strategy,
            zero,
            lhs,
            rhs,
            TestFixture::full_range(basis),
            TestFixture::full_range(basis),
            TestFixture::full_range(basis),
            alpha,
            beta);
        auto const expected = TestFixture::linear_combo(lhs, alpha, rhs, beta);
        TestFixture::expect_tensor_near(actual, expected);
    }
}

TYPED_TEST(GpuBlockVectorAddTypedTests, RespectsTruncatedIntersection) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const alpha = typename TestFixture::Accum{-0.5};
    auto const beta = typename TestFixture::Accum{2.0};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        if (basis.depth < 1) {
            continue;
        }
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const out = TestFixture::make_batch(6, basis);
        auto const lhs = TestFixture::make_batch(7, basis);
        auto const rhs = TestFixture::make_batch(8, basis);
        auto const out_range = typename TestFixture::DegreeRange{0, basis.depth - 1};
        auto const lhs_range = typename TestFixture::DegreeRange{1, basis.depth};
        auto const rhs_range = typename TestFixture::DegreeRange{0, basis.depth};

        auto const actual = TestFixture::run_gpu_add(
            basis,
            gpu_strategy,
            out,
            lhs,
            rhs,
            out_range,
            lhs_range,
            rhs_range,
            alpha,
            beta);
        auto const expected = TestFixture::reference_add(
            out, lhs, rhs, basis, out_range, lhs_range, rhs_range, alpha, beta);
        TestFixture::expect_tensor_near(actual, expected);
    }
}

} // namespace
