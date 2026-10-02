#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/tensor_reflect.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_ft_ops_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

template <typename Config>
class NumericTensorReflectTests
    : public rpp::tests::TypedCpuFreeTensorOpTestBase<Config> {};

TYPED_TEST_SUITE(NumericTensorReflectTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

class TensorReflectTests : public testing::Test,
                           public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;

    [[nodiscard]] static std::vector<Scalar>
    apply_reflect(Basis const& basis, std::vector<Scalar> const& arg) {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);

        auto const ctx = make_context();
        rpp::ops::TensorReflect<Strategy>{}(ctx, out_view, arg_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    linear_combo(std::vector<Scalar> const& lhs,
                 Scalar const& lhs_scale,
                 std::vector<Scalar> const& rhs,
                 Scalar const& rhs_scale) {
        std::vector<Scalar> result(lhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            result[i] = lhs_scale * lhs[i] + rhs_scale * rhs[i];
        }
        return result;
    }
};

TEST_F(TensorReflectTests, AppliesUnsignedWordReversal) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const arg = make_tensor('a', basis);
    auto const out = apply_reflect(basis, arg);

    std::vector<Scalar> expected(static_cast<std::size_t>(basis.size()));
    for_each_index(
        basis, [&expected, &arg, &basis](Degree degree, Index level_index) {
            auto const reversed = reverse_index(basis, degree, level_index);
            auto const source = basis.start_of_degree(degree) + level_index;
            auto const target = basis.start_of_degree(degree) + reversed;

            expected[static_cast<std::size_t>(target)] =
                arg[static_cast<std::size_t>(source)];
        });

    EXPECT_EQ(out, expected);
}

TEST_F(TensorReflectTests, IsLinear) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const lhs = make_tensor('a', basis);
    auto const rhs = make_tensor('b', basis);
    auto const alpha = make_scalar({{{{'p', 1}}, 2, 1}});
    auto const beta = make_scalar({{{{'q', 2}}, 3, 1}});

    auto const arg = linear_combo(lhs, alpha, rhs, beta);
    auto const reflected_arg = apply_reflect(basis, arg);
    auto const reflected_lhs = apply_reflect(basis, lhs);
    auto const reflected_rhs = apply_reflect(basis, rhs);

    EXPECT_EQ(reflected_arg,
              linear_combo(reflected_lhs, alpha, reflected_rhs, beta));
}

TEST_F(TensorReflectTests, IsInvolution) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const arg = make_tensor('a', basis);

    EXPECT_EQ(apply_reflect(basis, apply_reflect(basis, arg)), arg);
}

TEST_F(TensorReflectTests, AppliesOnlyToOverlappingArgumentDegreeRange) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const arg = make_tensor('a', basis);
    std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));
    std::vector<Scalar> expected(static_cast<std::size_t>(basis.size()));

    TensorView<Scalar*> out_view(out.data(), basis);
    TensorView<Scalar const*> arg_view(arg.data(), basis, 2, 3);

    auto const ctx = make_context();
    rpp::ops::TensorReflect<Strategy>{}(ctx, out_view, arg_view);

    for_each_index(
        basis, [&expected, &arg, &basis](Degree degree, Index level_index) {
            if (degree < 2 || degree > 3) {
                return;
            }
            auto const reversed = reverse_index(basis, degree, level_index);
            auto const source = basis.start_of_degree(degree) + level_index;
            auto const target = basis.start_of_degree(degree) + reversed;

            expected[static_cast<std::size_t>(target)] =
                arg[static_cast<std::size_t>(source)];
        });

    EXPECT_EQ(out, expected);
}

TEST_F(TensorReflectTests, KernelWrapperMatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto const arg = Wrapper::make_batch('a', basis);

    auto const err =
        rpp::ops::tensor_reflect(strategy,
                                 typename Wrapper::Strategy::LaunchConfig{},
                                 Wrapper::tensor_batch(actual, basis),
                                 Wrapper::tensor_batch(arg, basis),
                                 basis,
                                 Wrapper::tensor_count);
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::TensorReflect<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto in = Wrapper::tensor_view(arg, basis, tensor_idx);
            op(ctx, out, in);
        });

    EXPECT_EQ(actual, expected);
}

TYPED_TEST(NumericTensorReflectTests, ZeroExtendsArgumentAndPreservesOutsideOutputView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    using Scalar = typename TestFixture::Scalar;
    using Degree = typename TestFixture::Degree;
    using Index = typename TestFixture::Index;
    using DegreeRange = typename TestFixture::DegreeRange;
    auto const& basis = basis_data.basis;
    std::vector<Scalar> arg(static_cast<std::size_t>(basis.size()));
    for (std::size_t i = 0; i < arg.size(); ++i) {
        arg[i] = static_cast<Scalar>(i + 1);
    }
    std::vector<Scalar> initial_out(arg.size(), Scalar{7});

    struct Case {
        const char* name;
        DegreeRange out;
        DegreeRange arg;
    };
    const Case cases[] = {
        {"missing lower and upper degrees", {0, 4}, {2, 3}},
        {"single-degree overlap in truncated output", {1, 3}, {2, 2}},
        {"input wider than output", {1, 2}, {0, 4}},
        {"argument entirely above unit-only output", {0, 0}, {1, 4}},
        {"argument entirely below output", {2, 3}, {0, 1}},
        {"unit-only argument", {0, 4}, {0, 0}},
        {"unit-only input and output", {0, 0}, {0, 0}},
        {"matching truncated ranges", {1, 3}, {1, 3}},
    };
    for (auto const& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        auto actual = initial_out;
        auto expected = initial_out;
        for (Degree degree = test_case.out.min; degree <= test_case.out.max;
             ++degree) {
            const auto begin = basis.start_of_degree(degree);
            for (Index i = 0; i < basis.size_of_degree(degree); ++i) {
                const auto target = static_cast<std::size_t>(begin + i);
                if (degree < test_case.arg.min || degree > test_case.arg.max) {
                    expected[target] = Scalar{0};
                    continue;
                }

                // Reverse the base-width digits independently of reverse_index.
                Index source_index = i;
                Index reversed{0};
                for (Degree letter = 0; letter < degree; ++letter) {
                    reversed = reversed * basis.width + source_index % basis.width;
                    source_index /= basis.width;
                }
                const auto value = arg[static_cast<std::size_t>(begin + reversed)];
                expected[target] = value;
            }
        }

        typename TestFixture::TensorView out_view(
            actual.data(), basis, test_case.out.min, test_case.out.max);
        typename TestFixture::ConstTensorView arg_view(
            arg.data(), basis, test_case.arg.min, test_case.arg.max);
        rpp::ops::TensorReflect<typename TestFixture::Strategy>{}(
            TestFixture::make_context(), out_view, arg_view);
        TestFixture::expect_tensor_near(actual, expected);
    }
}

} // namespace
