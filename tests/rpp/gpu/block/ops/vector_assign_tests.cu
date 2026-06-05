#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/linalg/vector_assign.hpp>

#include "gpu_typed_vector_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockVectorAssignTypedTests
    : public rpp::tests::TypedGpuVectorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuVectorOpTestBase<Config>;
    using typename Base::Basis;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::is_empty;
    using Base::make_batch;
    using Base::overlap_range;

    static HostVector reference_assign(HostVector const& out,
                                       HostVector const& arg,
                                       Basis const& basis,
                                       typename Base::DegreeRange out_range,
                                       typename Base::DegreeRange arg_range) {
        auto result = out;
        auto const overlap = overlap_range(out_range, arg_range);
        if (is_empty(overlap)) {
            return result;
        }

        for (auto idx = basis.start_of_degree(overlap.min);
             idx < basis.end_of_degree(overlap.max);
             ++idx) {
            auto const i = static_cast<std::size_t>(idx);
            result[i] = arg[i];
        }
        return result;
    }

    static HostVector run_gpu_assign(Basis const& basis,
                                     GpuStrategy const& gpu_strategy,
                                     HostVector const& out,
                                     HostVector const& arg,
                                     typename Base::DegreeRange out_range,
                                     typename Base::DegreeRange arg_range) {
        DeviceVector device_out(out);
        DeviceVector device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::vector_assign(
            gpu_strategy,
            std::move(launch_config),
            rpp::make_graded_vector_batch(Helper::device_data(device_out),
                                          basis.size(),
                                          basis,
                                          out_range.min,
                                          out_range.max),
            rpp::make_graded_vector_batch(Helper::device_data(device_arg),
                                          basis.size(),
                                          basis,
                                          arg_range.min,
                                          arg_range.max),
            basis,
            Helper::tensor_count);
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

TYPED_TEST_SUITE(GpuBlockVectorAssignTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockVectorAssignTypedTests, CopiesArgumentOnFullView) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const out = TestFixture::make_batch(1, basis);
        auto const arg = TestFixture::make_batch(2, basis);

        auto const actual = TestFixture::run_gpu_assign(
            basis,
            gpu_strategy,
            out,
            arg,
            TestFixture::full_range(basis),
            TestFixture::full_range(basis));
        TestFixture::expect_tensor_near(actual, arg);
    }
}

TYPED_TEST(GpuBlockVectorAssignTypedTests, RespectsTruncatedIntersection) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        if (basis.depth < 1) {
            continue;
        }
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const out = TestFixture::make_batch(3, basis);
        auto const arg = TestFixture::make_batch(4, basis);
        auto const out_range = typename TestFixture::DegreeRange{0, basis.depth - 1};
        auto const arg_range = typename TestFixture::DegreeRange{1, basis.depth};

        auto const actual = TestFixture::run_gpu_assign(
            basis, gpu_strategy, out, arg, out_range, arg_range);
        auto const expected = TestFixture::reference_assign(
            out, arg, basis, out_range, arg_range);
        TestFixture::expect_tensor_near(actual, expected);
    }
}

TYPED_TEST(GpuBlockVectorAssignTypedTests, NoOverlapLeavesOutputUnchanged) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        if (basis.depth < 1) {
            continue;
        }
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const out = TestFixture::make_batch(5, basis);
        auto const arg = TestFixture::make_batch(6, basis);
        auto const out_range = typename TestFixture::DegreeRange{0, 0};
        auto const arg_range = typename TestFixture::DegreeRange{1, basis.depth};

        auto const actual = TestFixture::run_gpu_assign(
            basis, gpu_strategy, out, arg, out_range, arg_range);
        TestFixture::expect_tensor_near(actual, out);
    }
}

} // namespace
