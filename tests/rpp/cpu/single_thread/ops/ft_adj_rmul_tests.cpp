#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/ft_adj_rmul.hpp>
#include <rpp/cpu/single_thread/operations/basic/ft_mul.hpp>
#include <rpp/cpu/single_thread/operations/basic/tensor_pairing.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

using TensorBasis = rpp::basis::StandardTensorBasis;
using Degree = typename TensorBasis::Degree;
using Index = typename TensorBasis::Index;

struct NumericTestArchitecture {
    using Degree = ::Degree;
    using Index = ::Index;
    using Letter = std::uint8_t;
    using Bitmask = std::uint32_t;

    static constexpr unsigned max_depth = 16;
};

template <typename Scalar_, typename Accum_>
struct NumericAdjointRightMulConfig {
    using Scalar = Scalar_;
    using Accum = Accum_;
};

template <typename Scalar>
Scalar cast_scalar(float value) {
    return static_cast<Scalar>(value);
}

template <typename Scalar>
double scalar_to_double(Scalar value) {
    return static_cast<double>(value);
}

template <typename Scalar, typename Accum>
struct NumericTolerance {
    static constexpr double value = std::is_same_v<Scalar, double> ? 1e-12 : 1e-5;
};

template <typename Config>
class NumericFreeTensorAdjointRightMulTests : public testing::Test {
protected:
    using Scalar = typename Config::Scalar;
    using Accum = typename Config::Accum;
    using Basis = TensorBasis;
    using Strategy =
        rpp::cpu::strategies::SingleThreadStrategy<Accum, NumericTestArchitecture>;
    using TensorView = rpp::DenseTensorView<Scalar*, Basis>;
    using ConstTensorView = rpp::DenseTensorView<Scalar const*, Basis>;

    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;

    struct BasisData {
        std::vector<Index> degree_begin;
        Basis basis;

        BasisData(Degree width_, Degree depth_)
            : degree_begin(make_degree_begin(width_, depth_)),
              basis(width_, depth_, degree_begin.data()) {}
    };

    [[nodiscard]] static std::vector<Index> make_degree_begin(Degree width_,
                                                              Degree depth_) {
        std::vector<Index> result(static_cast<std::size_t>(depth_ + 2));
        for (Degree degree = 1; degree <= depth_ + 1; ++degree) {
            result[static_cast<std::size_t>(degree)] =
                1 + width_ * result[static_cast<std::size_t>(degree - 1)];
        }
        return result;
    }

    [[nodiscard]] static Scalar one() { return cast_scalar<Scalar>(1.0f); }

    [[nodiscard]] static std::vector<Scalar> zero_tensor(Basis const& basis) {
        return std::vector<Scalar>(static_cast<std::size_t>(basis.size()),
                                   cast_scalar<Scalar>(0.0f));
    }

    [[nodiscard]] static std::vector<Scalar>
    make_tensor(unsigned seed, Basis const& basis) {
        std::vector<Scalar> result(static_cast<std::size_t>(basis.size()));
        for (std::size_t i = 0; i < result.size(); ++i) {
            auto const centered =
                static_cast<int>((seed * 17u + static_cast<unsigned>(i) * 5u) % 9u) -
                4;
            auto const magnitude =
                static_cast<float>(centered) * 0.03125f +
                static_cast<float>((i + seed) % 3u) * 0.0078125f;
            result[i] = cast_scalar<Scalar>(magnitude);
        }
        return result;
    }

    [[nodiscard]] static Accum pairing(Basis const& basis,
                                       std::vector<Scalar> const& lhs,
                                       std::vector<Scalar> const& rhs) {
        Accum result{0};
        ConstTensorView lhs_view(lhs.data(), basis);
        ConstTensorView rhs_view(rhs.data(), basis);

        auto const ctx = Strategy::make_context(nullptr);
        rpp::ops::TensorPairing<Strategy>{}(ctx, result, lhs_view, rhs_view);
        return result;
    }

    [[nodiscard]] static std::vector<Scalar>
    apply_right_product(Basis const& basis,
                        std::vector<Scalar> const& arg,
                        std::vector<Scalar> const& op) {
        auto out = zero_tensor(basis);

        for (Degree degree = 0; degree <= basis.depth; ++degree) {
            const auto level_size = basis.size_of_degree(degree);
            const auto dst_begin = basis.start_of_degree(degree);

            for (Index level_index = 0; level_index < level_size; ++level_index) {
                Scalar value = cast_scalar<Scalar>(0.0f);
                auto word = rpp::tests::PolynomialTensorHelper::unpack_level_index(
                    basis, degree, level_index);

                for (Degree lhs_degree = 0; lhs_degree <= degree; ++lhs_degree) {
                    auto const rhs_degree = degree - lhs_degree;
                    auto const split =
                        word.begin() + static_cast<std::ptrdiff_t>(lhs_degree);
                    auto const lhs_index = rpp::tests::PolynomialTensorHelper::pack_word(
                        basis, word.begin(), split);
                    auto const rhs_index = rpp::tests::PolynomialTensorHelper::pack_word(
                        basis, split, word.end());

                    value +=
                        arg[static_cast<std::size_t>(
                            basis.start_of_degree(lhs_degree) + lhs_index)] *
                        op[static_cast<std::size_t>(
                            basis.start_of_degree(rhs_degree) + rhs_index)];
                }

                out[static_cast<std::size_t>(dst_begin + level_index)] = value;
            }
        }

        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    apply_adj_mul(Basis const& basis,
                  std::vector<Scalar> const& op,
                  std::vector<Scalar> const& arg) {
        using AdjMul = rpp::ops::FTAdjRMul<Strategy>;

        auto out = zero_tensor(basis);
        auto const scratch_bytes = AdjMul::scratch_space_size(Strategy{}, basis);
        std::vector<std::byte> scratch(scratch_bytes);

        TensorView out_view(out.data(), basis);
        ConstTensorView op_view(op.data(), basis);
        ConstTensorView arg_view(arg.data(), basis);

        auto const ctx = Strategy::make_context(scratch.data());
        AdjMul::init_scratch_space(ctx, basis);
        AdjMul{}(ctx, out_view, op_view, arg_view);
        AdjMul::destroy_scratch_space(ctx, basis);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    make_identity_operator(Basis const& basis) {
        auto result = zero_tensor(basis);
        result[0] = one();
        return result;
    }

    [[nodiscard]] static std::vector<Scalar>
    make_letter_operator(Basis const& basis, Index letter_index) {
        auto result = zero_tensor(basis);
        result[static_cast<std::size_t>(basis.start_of_degree(1) + letter_index)] =
            one();
        return result;
    }

    [[nodiscard]] static std::vector<Scalar>
    expected_right_shift(Basis const& basis,
                         std::vector<Scalar> const& arg,
                         Index letter_index) {
        auto result = zero_tensor(basis);
        for (Degree degree = 0; degree < basis.depth; ++degree) {
            const auto level_size = basis.size_of_degree(degree);
            const auto src_begin = basis.start_of_degree(degree + 1);
            const auto dst_begin = basis.start_of_degree(degree);
            for (Index idx = 0; idx < level_size; ++idx) {
                result[static_cast<std::size_t>(dst_begin + idx)] =
                    arg[static_cast<std::size_t>(
                        src_begin + idx * basis.width + letter_index)];
            }
        }
        return result;
    }

    static void expect_scalar_near(Accum actual, Accum expected) {
        auto const actual_d = scalar_to_double(actual);
        auto const expected_d = scalar_to_double(expected);
        auto const scale =
            std::max({1.0, std::abs(actual_d), std::abs(expected_d)});
        auto const tolerance = NumericTolerance<Scalar, Accum>::value * scale;
        EXPECT_NEAR(actual_d, expected_d, tolerance);
    }

    static void expect_tensor_near(std::vector<Scalar> const& actual,
                                   std::vector<Scalar> const& expected) {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            auto const actual_d = scalar_to_double(actual[i]);
            auto const expected_d = scalar_to_double(expected[i]);
            auto const scale =
                std::max({1.0, std::abs(actual_d), std::abs(expected_d)});
            auto const tolerance = NumericTolerance<Scalar, Accum>::value * scale;
            EXPECT_NEAR(actual_d, expected_d, tolerance)
                << "at coefficient " << i;
        }
    }
};

using NumericAdjointRightMulTestTypes = testing::Types<
    NumericAdjointRightMulConfig<float, float>,
    NumericAdjointRightMulConfig<double, double>>;

TYPED_TEST_SUITE(NumericFreeTensorAdjointRightMulTests,
                 NumericAdjointRightMulTestTypes);

TYPED_TEST(NumericFreeTensorAdjointRightMulTests,
           SatisfiesAdjointPairingCriterion) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;

    auto const op = TestFixture::make_tensor(1, basis);
    auto const t = TestFixture::make_tensor(2, basis);
    auto const arg = TestFixture::make_tensor(3, basis);

    auto const product = TestFixture::apply_right_product(basis, t, op);
    auto const adjoint = TestFixture::apply_adj_mul(basis, op, arg);

    TestFixture::expect_scalar_near(TestFixture::pairing(basis, adjoint, t),
                                    TestFixture::pairing(basis, arg, product));
}

TYPED_TEST(NumericFreeTensorAdjointRightMulTests,
           IdentityOperatorReturnsArgumentForTruncatedView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;

    auto const op = TestFixture::make_identity_operator(basis);
    auto const arg = TestFixture::make_tensor(7, basis);

    using AdjMul = rpp::ops::FTAdjRMul<typename TestFixture::Strategy>;
    auto out = TestFixture::zero_tensor(basis);
    auto const scratch_bytes =
        AdjMul::scratch_space_size(typename TestFixture::Strategy{}, basis);
    std::vector<std::byte> scratch(scratch_bytes);

    typename TestFixture::TensorView out_view(out.data(), basis);
    typename TestFixture::ConstTensorView op_view(
        op.data(), basis, Degree{0}, Degree{0});
    typename TestFixture::ConstTensorView arg_view(arg.data(), basis);

    auto const ctx = TestFixture::Strategy::make_context(scratch.data());
    AdjMul::init_scratch_space(ctx, basis);
    AdjMul{}(ctx, out_view, op_view, arg_view);
    AdjMul::destroy_scratch_space(ctx, basis);

    TestFixture::expect_tensor_near(out, arg);
}

TYPED_TEST(NumericFreeTensorAdjointRightMulTests,
           LetterOperatorShiftsCoefficientsRight) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;

    constexpr Index letter_index = 1;
    auto const op = TestFixture::make_letter_operator(basis, letter_index);
    auto const arg = TestFixture::make_tensor(11, basis);

    using AdjMul = rpp::ops::FTAdjRMul<typename TestFixture::Strategy>;
    auto out = TestFixture::zero_tensor(basis);
    auto const scratch_bytes =
        AdjMul::scratch_space_size(typename TestFixture::Strategy{}, basis);
    std::vector<std::byte> scratch(scratch_bytes);

    typename TestFixture::TensorView out_view(out.data(), basis);
    typename TestFixture::ConstTensorView op_view(
        op.data(), basis, Degree{1}, Degree{1});
    typename TestFixture::ConstTensorView arg_view(arg.data(), basis);

    auto const ctx = TestFixture::Strategy::make_context(scratch.data());
    AdjMul::init_scratch_space(ctx, basis);
    AdjMul{}(ctx, out_view, op_view, arg_view);
    AdjMul::destroy_scratch_space(ctx, basis);

    TestFixture::expect_tensor_near(
        out, TestFixture::expected_right_shift(basis, arg, letter_index));
}

class FreeTensorAdjointRightMulTests
    : public testing::Test,
      public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr ::Degree width = 3;
    static constexpr ::Degree depth = 4;

    [[nodiscard]] static Scalar one() { return make_scalar({{{}, 1, 1}}); }

    [[nodiscard]] static std::vector<Scalar> zero_tensor(Basis const& basis) {
        return std::vector<Scalar>(static_cast<std::size_t>(basis.size()));
    }

    [[nodiscard]] static Scalar pairing(Basis const& basis,
                                        std::vector<Scalar> const& lhs,
                                        std::vector<Scalar> const& rhs) {
        Scalar result;
        TensorView<Scalar const*> lhs_view(lhs.data(), basis);
        TensorView<Scalar const*> rhs_view(rhs.data(), basis);

        auto const ctx = make_context();
        rpp::ops::TensorPairing<Strategy>{}(ctx, result, lhs_view, rhs_view);
        return result;
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

    [[nodiscard]] static std::vector<Scalar>
    apply_right_product(Basis const& basis,
                        std::vector<Scalar> const& arg,
                        std::vector<Scalar> const& op) {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);
        TensorView<Scalar const*> op_view(op.data(), basis);

        auto const ctx = make_context();
        rpp::ops::FTMul<Strategy>{}(ctx, out_view, arg_view, op_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    apply_adj_mul(Basis const& basis,
                  std::vector<Scalar> const& op,
                  std::vector<Scalar> const& arg) {
        using AdjMul = rpp::ops::FTAdjRMul<Strategy>;

        auto out = zero_tensor(basis);
        auto const scratch_bytes =
            AdjMul::scratch_space_size(Strategy{}, basis);
        std::vector<std::byte> scratch(scratch_bytes);

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> op_view(op.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);

        auto const ctx = Strategy::make_context(scratch.data());
        AdjMul::init_scratch_space(ctx, basis);
        AdjMul{}(ctx, out_view, op_view, arg_view);
        AdjMul::destroy_scratch_space(ctx, basis);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    make_identity_operator(Basis const& basis) {
        auto result = zero_tensor(basis);
        result[0] = one();
        return result;
    }

    [[nodiscard]] static std::vector<Scalar>
    make_letter_operator(Basis const& basis, Index letter_index) {
        auto result = zero_tensor(basis);
        result[static_cast<std::size_t>(basis.start_of_degree(1) + letter_index)] =
            one();
        return result;
    }

    [[nodiscard]] static std::vector<Scalar>
    expected_right_shift(Basis const& basis,
                         std::vector<Scalar> const& arg,
                         Index letter_index) {
        auto result = zero_tensor(basis);
        for (Degree degree = 0; degree < basis.depth; ++degree) {
            const auto level_size = basis.size_of_degree(degree);
            const auto src_begin = basis.start_of_degree(degree + 1);
            const auto dst_begin = basis.start_of_degree(degree);
            for (Index idx = 0; idx < level_size; ++idx) {
                result[static_cast<std::size_t>(dst_begin + idx)] =
                    arg[static_cast<std::size_t>(
                        src_begin + idx * basis.width + letter_index)];
            }
        }
        return result;
    }
};

TEST_F(FreeTensorAdjointRightMulTests, SatisfiesAdjointPairingCriterion) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const op = make_tensor('a', basis);
    auto const t = make_tensor('t', basis);
    auto const arg = make_tensor('x', basis);

    auto const product = apply_right_product(basis, t, op);
    auto const adjoint = apply_adj_mul(basis, op, arg);

    EXPECT_EQ(pairing(basis, adjoint, t), pairing(basis, arg, product));
}

TEST_F(FreeTensorAdjointRightMulTests, IsBilinearInOperatorAndArgument) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const op1 = make_tensor('a', basis);
    auto const op2 = make_tensor('b', basis);
    auto const arg1 = make_tensor('x', basis);
    auto const arg2 = make_tensor('y', basis);

    auto const alpha = make_scalar({{{{'p', 1}}, 2, 1}});
    auto const beta = make_scalar({{{{'q', 2}}, 3, 1}});
    auto const gamma = make_scalar({{{{'r', 3}}, 5, 1}});
    auto const delta = make_scalar({{{{'s', 4}}, 7, 1}});

    auto const op = linear_combo(op1, alpha, op2, beta);
    auto const arg = linear_combo(arg1, gamma, arg2, delta);

    auto const lhs = apply_adj_mul(basis, op, arg);

    auto const ax11 = apply_adj_mul(basis, op1, arg1);
    auto const ax12 = apply_adj_mul(basis, op1, arg2);
    auto const ax21 = apply_adj_mul(basis, op2, arg1);
    auto const ax22 = apply_adj_mul(basis, op2, arg2);

    std::vector<Scalar> rhs(static_cast<std::size_t>(basis.size()));
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] = (alpha * gamma) * ax11[i] + (alpha * delta) * ax12[i] +
            (beta * gamma) * ax21[i] + (beta * delta) * ax22[i];
    }

    EXPECT_EQ(lhs, rhs);
}

TEST_F(FreeTensorAdjointRightMulTests,
       IdentityOperatorReturnsArgumentForTruncatedView) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const op = make_identity_operator(basis);
    auto const arg = make_tensor('x', basis);

    using AdjMul = rpp::ops::FTAdjRMul<Strategy>;
    auto out = zero_tensor(basis);
    auto const scratch_bytes = AdjMul::scratch_space_size(Strategy{}, basis);
    std::vector<std::byte> scratch(scratch_bytes);

    TensorView<Scalar*> out_view(out.data(), basis);
    TensorView<Scalar const*> op_view(op.data(), basis, Degree{0}, Degree{0});
    TensorView<Scalar const*> arg_view(arg.data(), basis);

    auto const ctx = Strategy::make_context(scratch.data());
    AdjMul::init_scratch_space(ctx, basis);
    AdjMul{}(ctx, out_view, op_view, arg_view);
    AdjMul::destroy_scratch_space(ctx, basis);

    EXPECT_EQ(out, arg);
}

TEST_F(FreeTensorAdjointRightMulTests, LetterOperatorShiftsCoefficientsRight) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    constexpr Index letter_index = 1;
    auto const op = make_letter_operator(basis, letter_index);
    auto const arg = make_tensor('x', basis);

    using AdjMul = rpp::ops::FTAdjRMul<Strategy>;
    auto out = zero_tensor(basis);
    auto const scratch_bytes = AdjMul::scratch_space_size(Strategy{}, basis);
    std::vector<std::byte> scratch(scratch_bytes);

    TensorView<Scalar*> out_view(out.data(), basis);
    TensorView<Scalar const*> op_view(op.data(), basis, Degree{1}, Degree{1});
    TensorView<Scalar const*> arg_view(arg.data(), basis);

    auto const ctx = Strategy::make_context(scratch.data());
    AdjMul::init_scratch_space(ctx, basis);
    AdjMul{}(ctx, out_view, op_view, arg_view);
    AdjMul::destroy_scratch_space(ctx, basis);

    EXPECT_EQ(out, expected_right_shift(basis, arg, letter_index));
}

TEST_F(FreeTensorAdjointRightMulTests,
       KernelWrapperMatchesDirectOperationForIdentityOperator) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto op = Wrapper::make_batch('a', basis);
    auto const arg = Wrapper::make_batch('x', basis);

    for (Wrapper::Index tensor_idx = 0; tensor_idx < Wrapper::tensor_count;
         ++tensor_idx) {
        auto const offset =
            static_cast<std::size_t>(tensor_idx * basis.size());
        std::fill(op.begin() + offset, op.begin() + offset + basis.size(), Scalar{});
        op[offset] = one();
    }

    const auto err = rpp::ops::ft_adj_rmul(
        strategy,
        typename Strategy::LaunchConfig{},
        Wrapper::tensor_batch(actual, basis),
        rpp::make_tensor_batch(op.data(), basis.size(), Degree{0}, Degree{0}),
        Wrapper::tensor_batch(arg, basis),
        basis,
        Wrapper::tensor_count);

    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::FTAdjRMul<Wrapper::Strategy>>(
        basis, [&](auto const& op_impl, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto operator_arg =
                Wrapper::tensor_view(op, basis, tensor_idx).truncate(0, 0);
            auto operand = Wrapper::tensor_view(arg, basis, tensor_idx);
            op_impl(ctx, out, operator_arg, operand);
        });

    EXPECT_EQ(actual, expected);
}


} // namespace
