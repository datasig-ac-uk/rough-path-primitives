#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/operations/single_thread/basic/tensor_add_identity.hpp>
#include <rpp/cpu/operations/single_thread/basic/tensor_set_identity.hpp>
#include <rpp/dense/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class TensorIdentityTests
    : public testing::Test,
      public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;
};

TEST_F(TensorIdentityTests, AddIdentityAddsOneToUnitOnly)
{
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

TEST_F(TensorIdentityTests, AddIdentitySupportsScaledIdentity)
{
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

TEST_F(TensorIdentityTests, SetIdentitySetsUnitAndZerosPositiveDegrees)
{
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

TEST_F(TensorIdentityTests, SetIdentitySupportsScaledIdentity)
{
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

TEST_F(TensorIdentityTests, AddIdentityKernelWrapperMatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const scalar = Wrapper::make_scalar({{{{'i', 1}}, 2, 3}});

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;

    rpp::cpu::single_thread::tensor_add_identity_kernel(
        Wrapper::tensor_batch(actual, basis),
        basis,
        strategy,
        Wrapper::tensor_count,
        scalar
    );
    Wrapper::apply_direct<rpp::ops::TensorAddIdentity<Wrapper::Strategy>>(
        basis,
        [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto tensor = Wrapper::tensor_view(expected, basis, tensor_idx);
            op(ctx, tensor, scalar);
        }
    );

    EXPECT_EQ(actual, expected);
}

TEST_F(TensorIdentityTests, SetIdentityKernelWrapperMatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const scalar = Wrapper::make_scalar({{{{'i', 2}}, 3, 4}});

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;

    rpp::cpu::single_thread::tensor_set_identity_kernel(
        Wrapper::tensor_batch(actual, basis),
        basis,
        strategy,
        Wrapper::tensor_count,
        scalar
    );
    Wrapper::apply_direct<rpp::ops::TensorSetIdentity<Wrapper::Strategy>>(
        basis,
        [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto tensor = Wrapper::tensor_view(expected, basis, tensor_idx);
            op(ctx, tensor, scalar);
        }
    );

    EXPECT_EQ(actual, expected);
}

} // namespace
