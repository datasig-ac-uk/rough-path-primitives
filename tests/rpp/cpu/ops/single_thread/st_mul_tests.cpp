#include <cstdint>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/operations/single_thread/basic/st_mul.hpp>
#include <rpp/dense/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

using Helper = rpp::tests::PolynomialTensorHelper;
using Degree = Helper::Degree;
using Index = Helper::Index;
using Scalar = Helper::Scalar;

struct DegreeRange {
    Degree min;
    Degree max;
};

struct MulViewCase {
    char const* name;
    DegreeRange out;
    DegreeRange lhs;
    DegreeRange rhs;
};

[[nodiscard]] bool contains(DegreeRange range, Degree degree) noexcept
{
    return range.min <= degree && degree <= range.max;
}

class ShuffleTensorMulTests
    : public testing::Test,
      public Helper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;

    [[nodiscard]] static TensorView<Scalar*> mutable_tensor_view(
        std::vector<Scalar>& data,
        Basis const& basis,
        DegreeRange range
    )
    {
        return {data.data(), basis, range.min, range.max};
    }

    [[nodiscard]] static TensorView<Scalar const*> const_tensor_view(
        std::vector<Scalar> const& data,
        Basis const& basis,
        DegreeRange range
    )
    {
        return {data.data(), basis, range.min, range.max};
    }

    [[nodiscard]] static Scalar shuffle_product_coefficient(
        Basis const& basis,
        std::vector<Scalar> const& lhs,
        std::vector<Scalar> const& rhs,
        Degree degree,
        Index level_index,
        DegreeRange lhs_range,
        DegreeRange rhs_range
    )
    {
        auto const word = unpack_level_index(basis, degree, level_index);
        auto const mask_count = std::uint32_t{1} << degree;

        Scalar entry{0};
        for (std::uint32_t mask = 0; mask < mask_count; ++mask) {
            std::vector<std::size_t> lhs_word;
            std::vector<std::size_t> rhs_word;
            lhs_word.reserve(word.size());
            rhs_word.reserve(word.size());

            for (Degree i = 0; i < degree; ++i) {
                if (((mask >> i) & std::uint32_t{1}) != 0) {
                    lhs_word.push_back(word[static_cast<std::size_t>(i)]);
                } else {
                    rhs_word.push_back(word[static_cast<std::size_t>(i)]);
                }
            }

            auto const lhs_degree = static_cast<Degree>(lhs_word.size());
            auto const rhs_degree = static_cast<Degree>(rhs_word.size());

            if (!contains(lhs_range, lhs_degree) || !contains(rhs_range, rhs_degree)) {
                continue;
            }

            auto const lhs_index = pack_word(basis, lhs_word.begin(), lhs_word.end());
            auto const rhs_index = pack_word(basis, rhs_word.begin(), rhs_word.end());

            entry += lhs[static_cast<std::size_t>(basis.start_of_degree(lhs_degree) + lhs_index)]
                   * rhs[static_cast<std::size_t>(basis.start_of_degree(rhs_degree) + rhs_index)];
        }

        return entry;
    }

    [[nodiscard]] static std::vector<Scalar> expected_mul(
        Basis const& basis,
        std::vector<Scalar> const& initial_out,
        std::vector<Scalar> const& lhs,
        std::vector<Scalar> const& rhs,
        DegreeRange out_range,
        DegreeRange lhs_range,
        DegreeRange rhs_range,
        Scalar const& beta = Scalar{1}
    )
    {
        auto expected = initial_out;

        for_each_index(
            basis,
            [&](Degree degree, Index level_index) {
                if (!contains(out_range, degree)) {
                    return;
                }

                expected[static_cast<std::size_t>(basis.start_of_degree(degree) + level_index)] =
                    beta * shuffle_product_coefficient(
                        basis,
                        lhs,
                        rhs,
                        degree,
                        level_index,
                        lhs_range,
                        rhs_range
                    );
            }
        );

        return expected;
    }

    static void expect_mul_matches_reference(
        DegreeRange out_range,
        DegreeRange lhs_range,
        DegreeRange rhs_range,
        Scalar const& beta = Scalar{1}
    )
    {
        auto const basis_data = BasisData(width, depth);
        auto const& basis = basis_data.basis;

        auto const initial_out = make_tensor('o', basis);
        auto out = initial_out;
        auto const lhs = make_tensor('a', basis);
        auto const rhs = make_tensor('b', basis);

        auto out_view = mutable_tensor_view(out, basis, out_range);
        auto const lhs_view = const_tensor_view(lhs, basis, lhs_range);
        auto const rhs_view = const_tensor_view(rhs, basis, rhs_range);

        auto const ctx = make_context();
        rpp::ops::STMul<Strategy>{}(ctx, out_view, lhs_view, rhs_view, beta);

        EXPECT_EQ(
            out,
            expected_mul(
                basis,
                initial_out,
                lhs,
                rhs,
                out_range,
                lhs_range,
                rhs_range,
                beta
            )
        );
    }

    [[nodiscard]] static std::vector<Scalar> apply_mul(
        Basis const& basis,
        std::vector<Scalar> const& lhs,
        std::vector<Scalar> const& rhs,
        DegreeRange out_range,
        DegreeRange lhs_range,
        DegreeRange rhs_range
    )
    {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        auto out_view = mutable_tensor_view(out, basis, out_range);
        auto const lhs_view = const_tensor_view(lhs, basis, lhs_range);
        auto const rhs_view = const_tensor_view(rhs, basis, rhs_range);

        auto const ctx = make_context();
        rpp::ops::STMul<Strategy>{}(ctx, out_view, lhs_view, rhs_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar> linear_combo(
        std::vector<Scalar> const& lhs,
        Scalar const& lhs_scale,
        std::vector<Scalar> const& rhs,
        Scalar const& rhs_scale
    )
    {
        std::vector<Scalar> result(lhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            result[i] = lhs_scale * lhs[i] + rhs_scale * rhs[i];
        }
        return result;
    }
};

TEST_F(ShuffleTensorMulTests, ComputesShuffleProduct)
{
    expect_mul_matches_reference(
        {0, depth},
        {0, depth},
        {0, depth}
    );
}

TEST_F(ShuffleTensorMulTests, HandlesTruncatedOperandDegreeRanges)
{
    MulViewCase const cases[] = {
        {"lhs has positive min degree", {0, depth}, {1, depth}, {0, depth}},
        {"rhs has positive min degree", {0, depth}, {0, depth}, {1, depth}},
        {"lhs has truncated max degree", {0, depth}, {0, 2}, {0, depth}},
        {"rhs has truncated max degree", {0, depth}, {0, depth}, {0, 2}},
        {"both operands are interior ranges", {0, depth}, {1, 3}, {1, 2}},
    };

    for (auto const& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        expect_mul_matches_reference(
            test_case.out,
            test_case.lhs,
            test_case.rhs
        );
    }
}

TEST_F(ShuffleTensorMulTests, RespectsTruncatedOutputDegreeRange)
{
    MulViewCase const cases[] = {
        {"output begins above zero", {2, depth}, {0, depth}, {0, depth}},
        {"output has truncated max degree", {0, 2}, {0, depth}, {0, depth}},
        {"output is an interior range", {1, 3}, {0, 2}, {1, depth}},
    };

    for (auto const& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        expect_mul_matches_reference(
            test_case.out,
            test_case.lhs,
            test_case.rhs
        );
    }
}

TEST_F(ShuffleTensorMulTests, HandlesScaledProduct)
{
    auto const beta = make_scalar({{{{'p', 1}}, 5, 2}});

    expect_mul_matches_reference(
        {1, depth},
        {0, 2},
        {1, depth},
        beta
    );
}

TEST_F(ShuffleTensorMulTests, IsBilinearInOperands)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    DegreeRange const out_range{1, 3};
    DegreeRange const lhs_range{1, depth};
    DegreeRange const rhs_range{0, 2};

    auto const lhs1 = make_tensor('a', basis);
    auto const lhs2 = make_tensor('b', basis);
    auto const rhs1 = make_tensor('x', basis);
    auto const rhs2 = make_tensor('y', basis);

    auto const alpha = make_scalar({{{{'p', 1}}, 2, 1}});
    auto const beta = make_scalar({{{{'q', 2}}, 3, 1}});
    auto const gamma = make_scalar({{{{'r', 3}}, 5, 1}});
    auto const delta = make_scalar({{{{'s', 4}}, 7, 1}});

    auto const lhs = linear_combo(lhs1, alpha, lhs2, beta);
    auto const rhs = linear_combo(rhs1, gamma, rhs2, delta);

    auto const result = apply_mul(basis, lhs, rhs, out_range, lhs_range, rhs_range);

    auto const m11 = apply_mul(basis, lhs1, rhs1, out_range, lhs_range, rhs_range);
    auto const m12 = apply_mul(basis, lhs1, rhs2, out_range, lhs_range, rhs_range);
    auto const m21 = apply_mul(basis, lhs2, rhs1, out_range, lhs_range, rhs_range);
    auto const m22 = apply_mul(basis, lhs2, rhs2, out_range, lhs_range, rhs_range);

    std::vector<Scalar> expected(static_cast<std::size_t>(basis.size()));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] = (alpha * gamma) * m11[i]
                    + (alpha * delta) * m12[i]
                    + (beta * gamma) * m21[i]
                    + (beta * delta) * m22[i];
    }

    EXPECT_EQ(result, expected);
}

TEST_F(ShuffleTensorMulTests, KernelWrapperMatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const beta = Wrapper::make_scalar({{{{'q', 2}}, 5, 3}});

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto const lhs = Wrapper::make_batch('a', basis);
    auto const rhs = Wrapper::make_batch('b', basis);

    rpp::cpu::single_thread::st_mul_kernel(
        Wrapper::tensor_batch(actual, basis),
        Wrapper::tensor_batch(lhs, basis),
        Wrapper::tensor_batch(rhs, basis),
        basis,
        strategy,
        Wrapper::tensor_count,
        beta
    );
    Wrapper::apply_direct<rpp::ops::STMul<Wrapper::Strategy>>(
        basis,
        [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto left = Wrapper::tensor_view(lhs, basis, tensor_idx);
            auto right = Wrapper::tensor_view(rhs, basis, tensor_idx);
            op(ctx, out, left, right, beta);
        }
    );

    EXPECT_EQ(actual, expected);
}

} // namespace
