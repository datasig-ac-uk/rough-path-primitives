#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/tensor_add_identity.hpp>
#include <rpp/cpu/single_thread/operations/basic/tensor_set_identity.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_ft_ops_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

template <typename Config>
class NumericTensorIdentityTests
    : public rpp::tests::TypedCpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::DegreeRange;
    using typename Base::Strategy;
    using typename Base::TensorView;
    using Base::expect_tensor_near;
    using Base::make_tensor;
    using Base::mutable_tensor_view;
    using Base::zero_tensor;

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_add_identity(Basis const& basis,
                     std::vector<typename Base::Scalar> const& initial,
                     DegreeRange range,
                     Accum scalar) {
        auto tensor = initial;
        TensorView tensor_view(tensor.data(), basis, range.min, range.max);
        auto const ctx = Base::make_context();
        rpp::ops::TensorAddIdentity<Strategy>{}(ctx, tensor_view, scalar);
        return tensor;
    }

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_set_identity(Basis const& basis,
                     std::vector<typename Base::Scalar> const& initial,
                     DegreeRange range,
                     Accum scalar) {
        auto tensor = initial;
        TensorView tensor_view(tensor.data(), basis, range.min, range.max);
        auto const ctx = Base::make_context();
        rpp::ops::TensorSetIdentity<Strategy>{}(ctx, tensor_view, scalar);
        return tensor;
    }
};

TYPED_TEST_SUITE(NumericTensorIdentityTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericTensorIdentityTests, AddIdentityUpdatesUnitOnFullView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const scalar = typename TestFixture::Accum{0.875};

    auto const initial = TestFixture::make_tensor(1, basis);
    auto expected = initial;
    expected[0] = static_cast<typename TestFixture::Scalar>(
        static_cast<typename TestFixture::Accum>(expected[0]) + scalar);

    auto const actual = TestFixture::run_add_identity(
        basis, initial, TestFixture::full_range(basis), scalar);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericTensorIdentityTests, AddIdentityIsNoOpWhenViewExcludesUnit) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    if (basis.depth < 1) {
        return;
    }
    auto const scalar = typename TestFixture::Accum{1.25};
    typename TestFixture::DegreeRange const range{1, basis.depth};

    auto const initial = TestFixture::make_tensor(2, basis);
    auto const actual = TestFixture::run_add_identity(basis, initial, range, scalar);
    TestFixture::expect_tensor_near(actual, initial);
}

TYPED_TEST(NumericTensorIdentityTests, SetIdentitySetsUnitTensorOnFullView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const scalar = typename TestFixture::Accum{1.25};

    auto const initial = TestFixture::make_tensor(3, basis);
    auto expected = TestFixture::zero_tensor(basis);
    expected[0] = static_cast<typename TestFixture::Scalar>(scalar);

    auto const actual = TestFixture::run_set_identity(
        basis, initial, TestFixture::full_range(basis), scalar);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericTensorIdentityTests,
           SetIdentityZerosOnlyActiveSliceWhenViewExcludesUnit) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    if (basis.depth < 1) {
        return;
    }
    auto const scalar = typename TestFixture::Accum{2.0};
    typename TestFixture::DegreeRange const range{1, basis.depth};

    auto const initial = TestFixture::make_tensor(4, basis);
    auto expected = initial;
    for (auto idx = basis.start_of_degree(range.min);
         idx < basis.end_of_degree(range.max);
         ++idx) {
        expected[static_cast<std::size_t>(idx)] =
            static_cast<typename TestFixture::Scalar>(0);
    }

    auto const actual = TestFixture::run_set_identity(basis, initial, range, scalar);
    TestFixture::expect_tensor_near(actual, expected);
}

class TensorIdentityTests : public testing::Test,
                            public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;
};

TEST_F(TensorIdentityTests, AddIdentityAddsOneToUnitOnly) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto tensor = make_tensor('a', basis);
    auto expected = tensor;
    expected[0] += Scalar{1};

    TensorView<Scalar*> tensor_view(tensor.data(), basis);

    auto const ctx = make_context();
    rpp::ops::TensorAddIdentity<Strategy>{}(ctx, tensor_view);

    EXPECT_EQ(tensor, expected);
}

TEST_F(TensorIdentityTests, AddIdentitySupportsScaledIdentity) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto tensor = make_tensor('a', basis);
    auto expected = tensor;
    auto const scalar = make_scalar({{{{'s', 1}}, 5, 3}});
    expected[0] += scalar;

    TensorView<Scalar*> tensor_view(tensor.data(), basis);

    auto const ctx = make_context();
    rpp::ops::TensorAddIdentity<Strategy>{}(ctx, tensor_view, scalar);

    EXPECT_EQ(tensor, expected);
}

TEST_F(TensorIdentityTests, SetIdentitySetsUnitAndZerosPositiveDegrees) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto tensor = make_tensor('a', basis);
    std::vector<Scalar> expected(static_cast<std::size_t>(basis.size()));
    expected[0] = Scalar{1};

    TensorView<Scalar*> tensor_view(tensor.data(), basis);

    auto const ctx = make_context();
    rpp::ops::TensorSetIdentity<Strategy>{}(ctx, tensor_view);

    EXPECT_EQ(tensor, expected);
}

TEST_F(TensorIdentityTests, SetIdentitySupportsScaledIdentity) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto tensor = make_tensor('a', basis);
    std::vector<Scalar> expected(static_cast<std::size_t>(basis.size()));
    auto const scalar = make_scalar({{{{'s', 2}}, 7, 5}});
    expected[0] = scalar;

    TensorView<Scalar*> tensor_view(tensor.data(), basis);

    auto const ctx = make_context();
    rpp::ops::TensorSetIdentity<Strategy>{}(ctx, tensor_view, scalar);

    EXPECT_EQ(tensor, expected);
}

TEST_F(TensorIdentityTests, AddIdentityKernelWrapperMatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const scalar = Wrapper::make_scalar({{{{'i', 1}}, 2, 3}});

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;

    auto const err = rpp::ops::tensor_add_identity(
        strategy,
        typename Wrapper::Strategy::LaunchConfig{},
        Wrapper::tensor_batch(actual, basis),
        basis,
        Wrapper::tensor_count,
        scalar);
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::TensorAddIdentity<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto tensor = Wrapper::tensor_view(expected, basis, tensor_idx);
            op(ctx, tensor, scalar);
        });

    EXPECT_EQ(actual, expected);
}

TEST_F(TensorIdentityTests, SetIdentityKernelWrapperMatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const scalar = Wrapper::make_scalar({{{{'i', 2}}, 3, 4}});

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;

    auto const err = rpp::ops::tensor_set_identity(
        strategy,
        typename Wrapper::Strategy::LaunchConfig{},
        Wrapper::tensor_batch(actual, basis),
        basis,
        Wrapper::tensor_count,
        scalar);
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::TensorSetIdentity<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto tensor = Wrapper::tensor_view(expected, basis, tensor_idx);
            op(ctx, tensor, scalar);
        });

    EXPECT_EQ(actual, expected);
}

} // namespace
