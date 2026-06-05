#ifndef RPP_TESTS_CPU_SINGLE_THREAD_OPS_CPU_TYPED_FT_OPS_TEST_HELPER_HPP
#define RPP_TESTS_CPU_SINGLE_THREAD_OPS_CPU_TYPED_FT_OPS_TEST_HELPER_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/basis/tensor_basis.hpp>
#include <rpp/cpu/strategies.hpp>
#include <rpp/views/views.hpp>

#include "polynomial_tensor_helper.hpp"

namespace rpp::tests {

using CpuTypedTensorBasis = rpp::basis::StandardTensorBasis;
using CpuTypedDegree = typename CpuTypedTensorBasis::Degree;
using CpuTypedIndex = typename CpuTypedTensorBasis::Index;

struct CpuTypedNumericTestArchitecture {
    using Degree = CpuTypedDegree;
    using Index = CpuTypedIndex;
    using Letter = std::uint8_t;
    using Bitmask = std::uint32_t;

    static constexpr unsigned max_depth = 16;
};

template <typename Scalar_, typename Accum_>
struct TypedScalarAccumConfig {
    using Scalar = Scalar_;
    using Accum = Accum_;
};

template <typename Scalar, typename Accum>
struct NumericTolerance {
    static constexpr double value = std::is_same_v<Scalar, double> ? 1e-12 : 1e-5;
};

template <typename Config>
class TypedCpuFreeTensorOpTestBase : public testing::Test {
protected:
    using Scalar = typename Config::Scalar;
    using Accum = typename Config::Accum;
    using Basis = CpuTypedTensorBasis;
    using Degree = CpuTypedDegree;
    using Index = CpuTypedIndex;
    using Strategy = rpp::cpu::strategies::SingleThreadStrategy<
        Accum,
        CpuTypedNumericTestArchitecture>;
    using TensorView = rpp::DenseTensorView<Scalar*, Basis>;
    using ConstTensorView = rpp::DenseTensorView<Scalar const*, Basis>;

    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;

    struct DegreeRange {
        Degree min;
        Degree max;
    };

    struct BasisData {
        std::vector<Index> degree_begin;
        Basis basis;

        BasisData(Degree width_, Degree depth_)
            : degree_begin(make_degree_begin(width_, depth_)),
              basis(width_, depth_, degree_begin.data()) {}
    };

    [[nodiscard]] static std::vector<Index> make_degree_begin(Degree width_,
                                                              Degree depth_) {
        return PolynomialTensorHelper::make_degree_begin(width_, depth_);
    }

    [[nodiscard]] static auto make_context() noexcept {
        return Strategy::make_context(nullptr);
    }

    [[nodiscard]] static Scalar cast_scalar(float value) {
        return static_cast<Scalar>(value);
    }

    [[nodiscard]] static double scalar_to_double(Scalar value) {
        return static_cast<double>(value);
    }

    [[nodiscard]] static std::vector<Scalar> zero_tensor(Basis const& basis) {
        return std::vector<Scalar>(static_cast<std::size_t>(basis.size()),
                                   cast_scalar(0.0f));
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
            result[i] = cast_scalar(magnitude);
        }
        return result;
    }

    [[nodiscard]] static std::vector<Scalar>
    make_unit_tensor(Basis const& basis) {
        auto result = zero_tensor(basis);
        result[0] = cast_scalar(1.0f);
        return result;
    }

    [[nodiscard]] static TensorView mutable_tensor_view(std::vector<Scalar>& data,
                                                        Basis const& basis,
                                                        DegreeRange range) {
        return {data.data(), basis, range.min, range.max};
    }

    [[nodiscard]] static ConstTensorView
    const_tensor_view(std::vector<Scalar> const& data,
                      Basis const& basis,
                      DegreeRange range) {
        return {data.data(), basis, range.min, range.max};
    }

    [[nodiscard]] static bool contains(DegreeRange range, Degree degree) noexcept {
        return range.min <= degree && degree <= range.max;
    }

    [[nodiscard]] static DegreeRange full_range(Basis const& basis) {
        return DegreeRange{0, basis.depth};
    }

    [[nodiscard]] static DegreeRange overlap_range(DegreeRange lhs,
                                                   DegreeRange rhs) {
        return DegreeRange{std::max(lhs.min, rhs.min), std::min(lhs.max, rhs.max)};
    }

    [[nodiscard]] static bool is_empty(DegreeRange range) noexcept {
        return range.max < range.min;
    }

    [[nodiscard]] static Scalar scalar_from_accum(Accum value) {
        return static_cast<Scalar>(value);
    }

    [[nodiscard]] static std::vector<Scalar>
    linear_combo(std::vector<Scalar> const& lhs,
                 Accum lhs_scale,
                 std::vector<Scalar> const& rhs,
                 Accum rhs_scale) {
        std::vector<Scalar> result(lhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            result[i] = scalar_from_accum(lhs_scale * static_cast<Accum>(lhs[i]) +
                                          rhs_scale * static_cast<Accum>(rhs[i]));
        }
        return result;
    }

    [[nodiscard]] static std::vector<Scalar>
    reference_mul(Basis const& basis,
                  std::vector<Scalar> const& initial_out,
                  std::vector<Scalar> const& lhs,
                  std::vector<Scalar> const& rhs,
                  DegreeRange out_range,
                  DegreeRange lhs_range,
                  DegreeRange rhs_range,
                  Accum beta = Accum{1}) {
        auto expected = initial_out;

        PolynomialTensorHelper::for_each_index(
            basis, [&](Degree degree, Index level_index) {
                if (!contains(out_range, degree)) {
                    return;
                }

                Accum entry{0};
                auto const word = PolynomialTensorHelper::unpack_level_index(
                    basis, degree, level_index);
                for (Degree mid = 0; mid <= degree; ++mid) {
                    auto const split =
                        word.begin() + static_cast<std::ptrdiff_t>(mid);
                    auto const lhs_index = PolynomialTensorHelper::pack_word(
                        basis, word.begin(), split);
                    auto const rhs_index = PolynomialTensorHelper::pack_word(
                        basis, split, word.end());
                    auto const rhs_degree = degree - mid;

                    if (contains(lhs_range, mid) &&
                        contains(rhs_range, rhs_degree)) {
                        entry += beta *
                            static_cast<Accum>(lhs[static_cast<std::size_t>(
                                basis.start_of_degree(mid) + lhs_index)]) *
                            static_cast<Accum>(rhs[static_cast<std::size_t>(
                                basis.start_of_degree(rhs_degree) + rhs_index)]);
                    }
                }

                expected[static_cast<std::size_t>(basis.start_of_degree(degree) +
                                                  level_index)] =
                    scalar_from_accum(entry);
            });

        return expected;
    }

    [[nodiscard]] static std::vector<Scalar>
    reference_fma(Basis const& basis,
                  std::vector<Scalar> const& initial_out,
                  std::vector<Scalar> const& a,
                  std::vector<Scalar> const& b,
                  std::vector<Scalar> const& c,
                  DegreeRange out_range,
                  DegreeRange a_range,
                  DegreeRange b_range,
                  DegreeRange c_range,
                  Accum alpha = Accum{1},
                  Accum beta = Accum{1}) {
        auto expected = initial_out;

        PolynomialTensorHelper::for_each_index(
            basis, [&](Degree degree, Index level_index) {
                if (!contains(out_range, degree)) {
                    return;
                }

                Accum entry{0};
                auto const global_index = static_cast<std::size_t>(
                    basis.start_of_degree(degree) + level_index);

                if (contains(a_range, degree)) {
                    entry += alpha * static_cast<Accum>(a[global_index]);
                }

                auto const word = PolynomialTensorHelper::unpack_level_index(
                    basis, degree, level_index);
                for (Degree mid = 0; mid <= degree; ++mid) {
                    auto const split =
                        word.begin() + static_cast<std::ptrdiff_t>(mid);
                    auto const lhs_index = PolynomialTensorHelper::pack_word(
                        basis, word.begin(), split);
                    auto const rhs_index = PolynomialTensorHelper::pack_word(
                        basis, split, word.end());
                    auto const rhs_degree = degree - mid;

                    if (contains(b_range, mid) && contains(c_range, rhs_degree)) {
                        entry += beta *
                            static_cast<Accum>(b[static_cast<std::size_t>(
                                basis.start_of_degree(mid) + lhs_index)]) *
                            static_cast<Accum>(c[static_cast<std::size_t>(
                                basis.start_of_degree(rhs_degree) + rhs_index)]);
                    }
                }

                expected[global_index] = scalar_from_accum(entry);
            });

        return expected;
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

using TypedCpuFreeTensorTestTypes = testing::Types<
    TypedScalarAccumConfig<float, float>,
    TypedScalarAccumConfig<double, double>>;

} // namespace rpp::tests

#endif // RPP_TESTS_CPU_SINGLE_THREAD_OPS_CPU_TYPED_FT_OPS_TEST_HELPER_HPP
