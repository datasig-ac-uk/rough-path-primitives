#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/linalg/vector_set_constant.hpp>
#include <rpp/gpu/block/operations/linalg/vector_scalar_multiply.hpp>
#include <rpp/gpu/block/operations/linalg/vector_inplace_add.hpp>

#include "../../../vector_bounds_test_helper.hpp"
#include "gpu_typed_vector_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockVectorBoundsTests : public rpp::tests::TypedGpuVectorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuVectorOpTestBase<Config>;
    using typename Base::Scalar;
    using typename Base::Accum;
    using typename Base::Degree;
    using typename Base::DegreeRange;
    using typename Base::Index;
    using typename Base::Helper;
    using typename Base::GpuStrategy;
    using typename Base::DeviceVector;

    template <rpp::tests::VectorBoundOperation Operation>
    static void check_bounds() {
        using Kind = rpp::tests::VectorBoundOperation;
        struct BasisConfig { Degree width, depth; };
        BasisConfig const configs[] = {{1, 4}, {4, 4}, {4, 0}, {127, 1}};
        for (auto const& config : configs) {
            SCOPED_TRACE(testing::Message() << "width=" << config.width
                                           << ", depth=" << config.depth);
            auto const basis_data = typename Helper::BasisData(config.width, config.depth);
            auto const& basis = basis_data.basis;
            auto const ranges = rpp::tests::all_tensor_degree_ranges<DegreeRange>(basis.depth);
            for (auto const out_range : ranges) {
                SCOPED_TRACE(testing::Message() << "out=[" << out_range.min << "," << out_range.max << "]");
                auto const rhs_ranges = Operation == Kind::InplaceAdd
                    ? ranges : std::vector<DegreeRange>{{0, basis.depth}};
                for (auto const rhs_range : rhs_ranges) {
                    SCOPED_TRACE(testing::Message() << "rhs=[" << rhs_range.min << "," << rhs_range.max << "]");
                    for (float scalar : {0.0f, 0.5f, -1.0f}) {
                        SCOPED_TRACE(scalar);
                        auto initial = rpp::tests::make_vector_bounds_storage<Scalar>(basis);
                        auto rhs = rpp::tests::make_vector_bounds_storage<Scalar>(basis);
                        for (Degree degree = 0; degree <= basis.depth; ++degree) {
                            for (auto idx = basis.start_of_degree(degree);
                                 idx < basis.end_of_degree(degree); ++idx) {
                                if constexpr (Operation == Kind::Set) {
                                    if (out_range.min <= degree && degree <= out_range.max) {
                                        initial[idx + 1] = static_cast<Scalar>(std::numeric_limits<float>::quiet_NaN());
                                    }
                                }
                                // An accidental read outside the RHS view must be observable.
                                if (degree < rhs_range.min || degree > rhs_range.max) {
                                    rhs[idx + 1] = static_cast<Scalar>(std::numeric_limits<float>::quiet_NaN());
                                }
                            }
                        }
                        auto const expected = rpp::tests::reference_vector_bounds(
                            initial, rhs, basis, out_range, rhs_range, scalar, Operation);
                        DeviceVector device_actual(initial), device_rhs(rhs);
                        auto const strategy = GpuStrategy{Helper::block_size};
                        rpp::gpu::DeviceLaunchConfig launch;
                        launch.stream = nullptr;
                        auto const out = rpp::make_graded_vector_batch(
                            Helper::device_data(device_actual) + 1, basis.size(), basis, out_range.min, out_range.max);
                        auto const arg = rpp::make_graded_vector_batch(
                            Helper::device_data(device_rhs) + 1, basis.size(), basis, rhs_range.min, rhs_range.max);
                        auto const err = [&] {
                            if constexpr (Operation == Kind::Set) {
                                return rpp::ops::vector_set_constant(strategy, launch, out, basis, Index{1}, static_cast<Accum>(scalar));
                            }
                            else if constexpr (Operation == Kind::Scale) {
                                return rpp::ops::vector_scalar_multiply(strategy, launch, out, basis, Index{1}, static_cast<Accum>(scalar));
                            }
                            else {
                                return rpp::ops::vector_inplace_add(strategy, launch, out, arg, basis, Index{1}, static_cast<Accum>(scalar));
                            }
                        }();
                        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
                        RPP_CUDA_ASSERT(cudaDeviceSynchronize());
                        auto const actual = Helper::copy_to_host(device_actual);
                        ASSERT_EQ(actual.size(), expected.size());
                        // All values and arithmetic here are exactly representable, including
                        // half/bfloat16. Check unchanged coefficients and guards exactly too.
                        for (std::size_t i = 0; i < actual.size(); ++i) {
                            EXPECT_EQ(static_cast<double>(actual[i]), static_cast<double>(expected[i]))
                                << "storage index " << i;
                        }
                    }
                }
            }
        }
    }
};

TYPED_TEST_SUITE(GpuBlockVectorBoundsTests, rpp::tests::TypedGpuAdjointTestTypes, rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockVectorBoundsTests, SetInitializesEveryViewAndPreservesOutside) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_bounds<rpp::tests::VectorBoundOperation::Set>();
}

TYPED_TEST(GpuBlockVectorBoundsTests, ScaleRespectsEveryView) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_bounds<rpp::tests::VectorBoundOperation::Scale>();
}

TYPED_TEST(GpuBlockVectorBoundsTests, InplaceAddRespectsEveryPairOfViews) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_bounds<rpp::tests::VectorBoundOperation::InplaceAdd>();
}

} // namespace

