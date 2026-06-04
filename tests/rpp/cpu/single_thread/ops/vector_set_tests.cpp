#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/linalg/vector_set_constant.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_vector_ops_test_helper.hpp"

namespace {

TEST(VectorSetConstantWrapperTests, MatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto strategy = Wrapper::Strategy{};
    auto const value = Wrapper::make_scalar({{{{'v', 1}}, 5, 3}});

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;

    auto const error =
        rpp::ops::vector_set_constant(strategy,
                                      Wrapper::Strategy::LaunchConfig{},
                                      Wrapper::vector_batch(actual, basis),
                                      basis,
                                      Wrapper::tensor_count,
                                      value);
    EXPECT_TRUE(static_cast<bool>(error));
    Wrapper::apply_direct<rpp::ops::VectorSetConstant<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto vec = Wrapper::vector_view(expected, basis, tensor_idx);
            op(ctx, vec, value);
        });

    EXPECT_EQ(actual, expected);
}

template <typename Config>
class NumericVectorSetTests : public rpp::tests::TypedCpuVectorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuVectorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::DegreeRange;
    using typename Base::Strategy;
    using typename Base::VectorView;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::make_tensor;
    using Base::mutable_vector_view;

    [[nodiscard]] static std::vector<typename Base::Scalar>
    reference_set(std::vector<typename Base::Scalar> const& initial,
                  Basis const& basis,
                  DegreeRange range,
                  Accum value) {
        auto result = initial;
        for (auto idx = basis.start_of_degree(range.min);
             idx < basis.end_of_degree(range.max);
             ++idx) {
            result[static_cast<std::size_t>(idx)] =
                static_cast<typename Base::Scalar>(value);
        }
        return result;
    }

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_set(Basis const& basis,
            std::vector<typename Base::Scalar> const& initial,
            DegreeRange range,
            Accum value) {
        auto actual = initial;
        auto vec_view = mutable_vector_view(actual, basis, range);
        auto const ctx = Base::make_context();
        rpp::ops::VectorSetConstant<Strategy>{}(ctx, vec_view, value);
        return actual;
    }
};

TYPED_TEST_SUITE(NumericVectorSetTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericVectorSetTests, SetsFullViewToConstant) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const initial = TestFixture::make_tensor(1, basis);
    auto const value = typename TestFixture::Accum{3.25};

    auto const actual = TestFixture::run_set(
        basis, initial, TestFixture::full_range(basis), value);
    auto const expected =
        TestFixture::reference_set(initial, basis, TestFixture::full_range(basis), value);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericVectorSetTests, SetsOnlyActiveSliceForView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const initial = TestFixture::make_tensor(2, basis);
    auto const value = typename TestFixture::Accum{-1.5};
    typename TestFixture::DegreeRange const range{1, TestFixture::depth};

    auto const actual = TestFixture::run_set(basis, initial, range, value);
    auto const expected = TestFixture::reference_set(initial, basis, range, value);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericVectorSetTests, ZeroValueZerosActiveSlice) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const initial = TestFixture::make_tensor(3, basis);
    typename TestFixture::DegreeRange const range{0, 1};

    auto const actual = TestFixture::run_set(
        basis, initial, range, typename TestFixture::Accum{0});
    auto const expected = TestFixture::reference_set(
        initial, basis, range, typename TestFixture::Accum{0});
    TestFixture::expect_tensor_near(actual, expected);
}

} // namespace
