#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/linalg/vector_set_constant.hpp>
#include <rpp/cpu/single_thread/operations/linalg/vector_scalar_multiply.hpp>
#include <rpp/cpu/single_thread/operations/linalg/vector_inplace_add.hpp>

#include "../../../vector_bounds_test_helper.hpp"
#include "cpu_typed_vector_ops_test_helper.hpp"

namespace {

template <typename Config>
class CpuVectorBoundsTests : public rpp::tests::TypedCpuVectorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuVectorOpTestBase<Config>;
    using typename Base::Scalar;
    using typename Base::Accum;
    using typename Base::Degree;
    using typename Base::DegreeRange;
    using typename Base::Strategy;

    template <rpp::tests::VectorBoundOperation Operation>
    static void check_bounds() {
        using Kind = rpp::tests::VectorBoundOperation;
        struct BasisConfig { Degree width, depth; };
        BasisConfig const configs[] = {{1, 4}, {4, 4}, {4, 0}, {127, 1}};
        for (auto const& config : configs) {
            SCOPED_TRACE(testing::Message() << "width=" << config.width
                                           << ", depth=" << config.depth);
            auto const basis_data = typename Base::BasisData(config.width, config.depth);
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
                        auto actual = initial;
                        typename Base::VectorView out(actual.data() + 1, basis, out_range.min, out_range.max);
                        typename Base::ConstVectorView arg(rhs.data() + 1, basis, rhs_range.min, rhs_range.max);
                        auto const ctx = Base::make_context();
                        if constexpr (Operation == Kind::Set) {
                            rpp::ops::VectorSetConstant<Strategy>{}(ctx, out, static_cast<Accum>(scalar));
                        }
                        else if constexpr (Operation == Kind::Scale) {
                            rpp::ops::VectorScalarMultiply<Strategy>{}(ctx, out, static_cast<Accum>(scalar));
                        }
                        else {
                            rpp::ops::VectorInplaceAdd<Strategy>{}(ctx, out, arg, static_cast<Accum>(scalar));
                        }
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

TYPED_TEST_SUITE(CpuVectorBoundsTests, rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(CpuVectorBoundsTests, SetInitializesEveryViewAndPreservesOutside) {
    TestFixture::template check_bounds<rpp::tests::VectorBoundOperation::Set>();
}

TYPED_TEST(CpuVectorBoundsTests, ScaleRespectsEveryView) {
    TestFixture::template check_bounds<rpp::tests::VectorBoundOperation::Scale>();
}

TYPED_TEST(CpuVectorBoundsTests, InplaceAddRespectsEveryPairOfViews) {
    TestFixture::template check_bounds<rpp::tests::VectorBoundOperation::InplaceAdd>();
}

} // namespace

