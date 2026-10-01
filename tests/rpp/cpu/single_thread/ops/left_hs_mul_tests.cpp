#include <array>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/left_hs_mul.hpp>
#include <rpp/cpu/single_thread/operations/basic/st_mul.hpp>
#include <rpp/cpu/single_thread/operations/intermediate/ft_exp.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class LeftHalfShuffleTests : public testing::Test,
                             public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;
    BasisData basis_data{width, depth};
    Basis const& basis = basis_data.basis;
    using Tensor = std::vector<Scalar>;

    Tensor zero() const { return Tensor(basis.size(), Scalar{0}); }

    Tensor word(std::initializer_list<Index> letters) const {
        auto result = zero();
        result[basis.start_of_degree(letters.size()) +
               pack_word(basis, letters.begin(), letters.end())] = Scalar{1};
        return result;
    }

    Tensor half(Tensor const& integrator, Tensor const& integrand) const {
        // Nonzero initial storage detects coefficients that are never assigned.
        Tensor out(basis.size(), Scalar{17});
        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> lhs(integrator.data(), basis);
        TensorView<Scalar const*> rhs(integrand.data(), basis);
        rpp::ops::LeftHSMul<Strategy>{}(make_context(), out_view, lhs, rhs);
        return out;
    }

    Tensor shuffle(Tensor const& lhs, Tensor const& rhs) const {
        auto out = zero();
        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> lhs_view(lhs.data(), basis);
        TensorView<Scalar const*> rhs_view(rhs.data(), basis);
        rpp::ops::STMul<Strategy>{}(
            make_context(), out_view, lhs_view, rhs_view);
        return out;
    }

    static Tensor add(Tensor lhs, Tensor const& rhs) {
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            lhs[i] += rhs[i];
        }
        return lhs;
    }

    static Tensor scale(Tensor value, Scalar const& factor) {
        for (auto& entry : value) {
            entry = factor * entry;
        }
        return value;
    }
};

TEST_F(LeftHalfShuffleTests, EmptyWordRules) {
    auto const unit = word({});
    auto const f = make_tensor('f', basis);
    auto positive_f = f;
    positive_f[0] = Scalar{0};
    // e prec f = 0; f prec e = f - f_e e, including e prec e = 0.
    EXPECT_EQ(half(unit, f), zero());
    EXPECT_EQ(half(f, unit), positive_f);
    EXPECT_EQ(half(unit, unit), zero());
    EXPECT_EQ(half(zero(), f), zero());
    EXPECT_EQ(half(f, zero()), zero());
}

TEST_F(LeftHalfShuffleTests,
       ExplicitWordProductsFixOrientationAndMultiplicity) {
    // a prec b = ab, while b prec a = ba.
    EXPECT_EQ(half(word({0}), word({1})), word({0, 1}));
    EXPECT_EQ(half(word({1}), word({0})), word({1, 0}));
    // ab prec c = a(b shuffle c) = abc + acb.
    EXPECT_EQ(half(word({0, 1}), word({2})),
              add(word({0, 1, 2}), word({0, 2, 1})));
    // a^2 prec a^2 = binomial(3, 2) a^4.
    EXPECT_EQ(half(word({0, 0}), word({0, 0})),
              scale(word({0, 0, 0, 0}), Scalar{3}));
}

TEST_F(LeftHalfShuffleTests, ShuffleSplitsIntoTwoHalfShufflesAndConstantTerm) {
    auto const f = make_tensor('f', basis);
    auto const g = make_tensor('g', basis);
    auto expected = add(half(f, g), half(g, f));
    expected[0] += f[0] * g[0];
    EXPECT_EQ(shuffle(f, g), expected);
}

TEST_F(LeftHalfShuffleTests, SatisfiesZinbielIdentityIncludingConstants) {
    auto const f = make_tensor('f', basis);
    auto const g = make_tensor('g', basis);
    auto const h = make_tensor('h', basis);
    // (f prec g) prec h = f prec (g shuffle h).
    // All products are in the quotient by degrees greater than depth.
    EXPECT_EQ(half(half(f, g), h), half(f, shuffle(g, h)));
}

TEST_F(LeftHalfShuffleTests, IsBilinearAndAppliesLeftScaling) {
    auto const f = make_tensor('f', basis);
    auto const g = make_tensor('g', basis);
    auto const h = make_tensor('h', basis);
    auto const k = make_tensor('k', basis);
    auto const p = symbol('p', basis, 0, 0);
    auto const q = symbol('q', basis, 0, 0);
    EXPECT_EQ(half(add(scale(f, p), g), add(scale(h, q), k)),
              add(add(scale(half(f, h), p * q), scale(half(f, k), p)),
                  add(scale(half(g, h), q), half(g, k))));
}

TEST_F(LeftHalfShuffleTests,
       DegreeViewsAgreeWithProjectionAndPreserveOutsideStorage) {
    auto const f = make_tensor('f', basis);
    auto const g = make_tensor('g', basis);
    struct Range {
        Degree min, max;
    };
    Range const ranges[] = {{0, 0}, {0, 1}, {1, 2}, {2, 4}, {0, 4}};
    auto const beta = Scalar{3};
    for (auto lhs_range : ranges) {
        for (auto rhs_range : ranges) {
            auto projected_f = f;
            auto projected_g = g;
            for_each_index(basis, [&](Degree d, Index i) {
                auto const idx = basis.start_of_degree(d) + i;
                if (d < lhs_range.min || d > lhs_range.max) {
                    projected_f[idx] = Scalar{0};
                }
                if (d < rhs_range.min || d > rhs_range.max) {
                    projected_g[idx] = Scalar{0};
                }
            });
            auto const product = scale(half(projected_f, projected_g), beta);
            for (auto out_range : ranges) {
                SCOPED_TRACE(testing::Message()
                             << lhs_range.min << ':' << lhs_range.max << ' '
                             << rhs_range.min << ':' << rhs_range.max << ' '
                             << out_range.min << ':' << out_range.max);
                auto actual = make_tensor('o', basis);
                auto expected = actual;
                for_each_index(basis, [&](Degree d, Index i) {
                    auto const idx = basis.start_of_degree(d) + i;
                    if (out_range.min <= d && d <= out_range.max) {
                        expected[idx] = product[idx];
                    }
                });
                TensorView<Scalar*> out(
                    actual.data(), basis, out_range.min, out_range.max);
                TensorView<Scalar const*> lhs(
                    f.data(), basis, lhs_range.min, lhs_range.max);
                TensorView<Scalar const*> rhs(
                    g.data(), basis, rhs_range.min, rhs_range.max);
                rpp::ops::LeftHSMul<Strategy>{}(
                    make_context(), out, lhs, rhs, beta);
                EXPECT_EQ(actual, expected);
            }
        }
    }
}

TEST_F(LeftHalfShuffleTests,
       PairingWithStraightLineSignatureMatchesIntegration) {
    auto const t = symbol('t', basis, 0, 0);
    auto const increment = scale(word({0}), t);
    auto signature = zero();
    TensorView<Scalar*> out(signature.data(), basis);
    TensorView<Scalar const*> arg(increment.data(), basis);
    rpp::ops::FTExp<Strategy>{}(make_context(), out, arg);
    // Along x(s)=t*s, <a^m,S(s)>=(t*s)^m/m!.
    // Integral <a^n,S(s)> d<a^m,S(s)> =
    // t^(m+n) / ((m+n)*(m-1)!*n!). Choose m=2, n=1.
    auto const product = half(word({0, 0}), word({0}));
    Scalar pairing{0};
    for (std::size_t i = 0; i < product.size(); ++i) {
        pairing += product[i] * signature[i];
    }
    EXPECT_EQ(pairing, (t * t * t) * make_scalar({{{}, 1, 3}}));
    // Reversing the roles gives integral x(s)^2/2 dx(s) = t^3/6.
    auto const reversed = half(word({0}), word({0, 0}));
    pairing = Scalar{0};
    for (std::size_t i = 0; i < reversed.size(); ++i) {
        pairing += reversed[i] * signature[i];
    }
    EXPECT_EQ(pairing, (t * t * t) * make_scalar({{{}, 1, 6}}));
}

TEST_F(LeftHalfShuffleTests, KernelWrapperUsesIntegratorThenIntegrand) {
    using Helper = rpp::tests::CpuKernelWrapperTestHelper;
    auto const data = Helper::BasisData(Helper::width, Helper::depth);
    auto const& b = data.basis;
    auto actual = Helper::make_batch('o', b);
    auto expected = actual;
    auto const lhs = Helper::make_batch('a', b);
    auto const rhs = Helper::make_batch('b', b);
    auto const err = rpp::ops::left_hs_mul(Helper::Strategy{},
                                           Helper::Strategy::LaunchConfig{},
                                           Helper::tensor_batch(actual, b),
                                           Helper::tensor_batch(lhs, b),
                                           Helper::tensor_batch(rhs, b),
                                           b,
                                           Helper::tensor_count);
    ASSERT_TRUE(static_cast<bool>(err));
    Helper::apply_direct<rpp::ops::LeftHSMul<Helper::Strategy>>(
        b, [&](auto const& op, auto const& ctx, Helper::Index tensor_idx) {
            auto out = Helper::tensor_view(expected, b, tensor_idx);
            auto left = Helper::tensor_view(lhs, b, tensor_idx);
            auto right = Helper::tensor_view(rhs, b, tensor_idx);
            op(ctx, out, left, right);
        });
    EXPECT_EQ(actual, expected);
}

// Matrix coefficients exercise ordering without imposing commutative
// identities.
struct Matrix {
    std::array<int, 4> entries{};
    Matrix() = default;
    Matrix(int scalar) : entries{scalar, 0, 0, scalar} {}
    explicit Matrix(std::array<int, 4> value) : entries(value) {}
    Matrix& operator+=(Matrix const& rhs) {
        for (std::size_t i = 0; i < 4; ++i) {
            entries[i] += rhs.entries[i];
        }
        return *this;
    }
    friend Matrix operator*(Matrix const& lhs, Matrix const& rhs) {
        Matrix result;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    result.entries[2 * i + j] +=
                        lhs.entries[2 * i + k] * rhs.entries[2 * k + j];
                }
            }
        }
        return result;
    }
    friend bool operator==(Matrix const& lhs, Matrix const& rhs) {
        return lhs.entries == rhs.entries;
    }
};

TEST(LeftHalfShuffleOrderingTests,
     PreservesNoncommutingCoefficientAndScaleOrder) {
    using Helper = rpp::tests::PolynomialTensorHelper;
    using Strategy =
        rpp::cpu::strategies::SingleThreadStrategy<Matrix,
                                                   Helper::TestArchitecture>;
    auto const data = Helper::BasisData(2, 3);
    auto const& basis = data.basis;
    Matrix const a(std::array<int, 4>{0, 1, 0, 0});
    Matrix const b(std::array<int, 4>{0, 0, 1, 0});
    Matrix const beta(std::array<int, 4>{1, 2, 3, 4});
    ASSERT_FALSE(a * b == b * a);
    std::vector<Matrix> lhs(basis.size()), rhs(basis.size()),
        out(basis.size(), Matrix{17});
    // ab prec a = a(ba + ab); the same coefficient order applies to both words.
    lhs[basis.start_of_degree(2) + 1] = a;
    rhs[basis.start_of_degree(1)] = b;
    rpp::DenseTensorView<Matrix*, Helper::Basis> out_view(out.data(), basis);
    rpp::DenseTensorView<Matrix const*, Helper::Basis> lhs_view(lhs.data(),
                                                                basis),
        rhs_view(rhs.data(), basis);
    rpp::ops::LeftHSMul<Strategy>{}(
        Strategy::make_context(nullptr), out_view, lhs_view, rhs_view, beta);
    std::vector<Matrix> expected(basis.size());
    expected[basis.start_of_degree(3) + 2] = beta * (a * b); // aba
    expected[basis.start_of_degree(3) + 1] = beta * (a * b); // aab
    EXPECT_EQ(out, expected);
}

} // namespace
