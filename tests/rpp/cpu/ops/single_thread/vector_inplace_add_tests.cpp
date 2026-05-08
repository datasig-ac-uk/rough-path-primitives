#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/operations/single_thread/basic/vector_inplace_add.hpp>
#include <rpp/dense/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class VectorInplaceAddTests
    : public testing::Test,
      public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 3;
};

TEST_F(VectorInplaceAddTests, AddsRhsCoefficientwise)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto lhs = make_tensor('a', basis);
    auto const rhs = make_tensor('b', basis);

    VectorView<Scalar*> lhs_view(lhs.data(), basis);
    VectorView<Scalar const*> rhs_view(rhs.data(), basis);

    auto const ctx = make_context();
    rpp::ops::VectorInplaceAdd<Strategy>{}(ctx, lhs_view, rhs_view);

    std::vector<Scalar> expected;
    expected.reserve(static_cast<std::size_t>(basis.size()));
    for_each_index(
        basis,
        [&expected, &basis](Degree degree, Index level_index) {
            expected.emplace_back(
                symbol('a', basis, degree, level_index)
                + symbol('b', basis, degree, level_index)
            );
        }
    );

    EXPECT_EQ(lhs, expected);
}

TEST_F(VectorInplaceAddTests, AddsScaledRhsCoefficientwise)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto lhs = make_tensor('a', basis);
    auto const rhs = make_tensor('b', basis);
    auto const multiplier = make_scalar({{{{'c', 0}}, 1, 1}});

    VectorView<Scalar*> lhs_view(lhs.data(), basis);
    VectorView<Scalar const*> rhs_view(rhs.data(), basis);

    auto const ctx = make_context();
    rpp::ops::VectorInplaceAdd<Strategy>{}(ctx, lhs_view, rhs_view, multiplier);

    std::vector<Scalar> expected;
    expected.reserve(static_cast<std::size_t>(basis.size()));
    for_each_index(
        basis,
        [&expected, &basis, &multiplier](Degree degree, Index level_index) {
            expected.emplace_back(
                symbol('a', basis, degree, level_index)
                + multiplier * symbol('b', basis, degree, level_index)
            );
        }
    );

    EXPECT_EQ(lhs, expected);
}

TEST_F(VectorInplaceAddTests, KernelWrapperMatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const alpha = Wrapper::make_scalar({{{{'p', 1}}, 7, 5}});

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;
    auto const rhs = Wrapper::make_batch('b', basis);

    rpp::cpu::single_thread::vector_inplace_add_kernel(
        Wrapper::vector_batch(actual, basis),
        Wrapper::vector_batch(rhs, basis),
        basis,
        strategy,
        Wrapper::tensor_count,
        alpha
    );
    Wrapper::apply_direct<rpp::ops::VectorInplaceAdd<Wrapper::Strategy>>(
        basis,
        [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto lhs = Wrapper::vector_view(expected, basis, tensor_idx);
            auto arg = Wrapper::vector_view(rhs, basis, tensor_idx);
            op(ctx, lhs, arg, alpha);
        }
    );

    EXPECT_EQ(actual, expected);
}

} // namespace
