#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/linalg/vector_assign.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class VectorAssignTests
    : public testing::Test,
      public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;
};

TEST_F(VectorAssignTests, CopiesSourceCoefficientwise)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto out = make_tensor('a', basis);
    auto const arg = make_tensor('b', basis);

    VectorView<Scalar*> out_view(out.data(), basis);
    VectorView<Scalar const*> arg_view(arg.data(), basis);

    auto const ctx = make_context();
    rpp::ops::VectorAssign<Strategy>{}(ctx, out_view, arg_view);

    EXPECT_EQ(out, arg);
}

TEST_F(VectorAssignTests, CopiesOnlyOverlappingDegreeRange)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto out = make_tensor('a', basis);
    auto const arg = make_tensor('b', basis);
    auto expected = out;

    for (Degree degree = 2; degree <= 3; ++degree) {
        auto const begin = basis.start_of_degree(degree);
        auto const end = basis.end_of_degree(degree);
        for (Index i = begin; i < end; ++i) {
            expected[static_cast<std::size_t>(i)] = arg[static_cast<std::size_t>(i)];
        }
    }

    VectorView<Scalar*> out_view(out.data(), basis, 1, 3);
    VectorView<Scalar const*> arg_view(arg.data(), basis, 2, 4);

    auto const ctx = make_context();
    rpp::ops::VectorAssign<Strategy>{}(ctx, out_view, arg_view);

    EXPECT_EQ(out, expected);
}

TEST_F(VectorAssignTests, KernelWrapperMatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;
    auto const arg = Wrapper::make_batch('b', basis);

    auto const err = rpp::ops::vector_assign(
        strategy,
        typename Wrapper::Strategy::LaunchConfig{},
        Wrapper::vector_batch(actual, basis),
        Wrapper::vector_batch(arg, basis),
        basis,
        Wrapper::tensor_count
    );
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::VectorAssign<Wrapper::Strategy>>(
        basis,
        [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::vector_view(expected, basis, tensor_idx);
            auto rhs = Wrapper::vector_view(arg, basis, tensor_idx);
            op(ctx, out, rhs);
        }
    );

    EXPECT_EQ(actual, expected);
}

} // namespace
