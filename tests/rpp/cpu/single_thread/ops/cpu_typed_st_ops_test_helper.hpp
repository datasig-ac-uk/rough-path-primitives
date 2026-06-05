#ifndef RPP_TESTS_CPU_SINGLE_THREAD_OPS_CPU_TYPED_ST_OPS_TEST_HELPER_HPP
#define RPP_TESTS_CPU_SINGLE_THREAD_OPS_CPU_TYPED_ST_OPS_TEST_HELPER_HPP

#include <cstdint>
#include <vector>

#include "cpu_typed_ft_ops_test_helper.hpp"

namespace rpp::tests {

template <typename Config>
class TypedCpuShuffleTensorOpTestBase : public TypedCpuFreeTensorOpTestBase<Config> {
protected:
    using Base = TypedCpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Degree;
    using typename Base::DegreeRange;
    using typename Base::Index;
    using typename Base::Scalar;

    [[nodiscard]] static Accum shuffle_product_coefficient(
        Basis const& basis,
        std::vector<Scalar> const& lhs,
        std::vector<Scalar> const& rhs,
        Degree degree,
        Index level_index,
        DegreeRange lhs_range,
        DegreeRange rhs_range) {
        auto const word =
            PolynomialTensorHelper::unpack_level_index(basis, degree, level_index);
        auto const mask_count = std::uint32_t{1} << degree;

        Accum entry{0};
        for (std::uint32_t mask = 0; mask < mask_count; ++mask) {
            std::vector<std::size_t> lhs_word;
            std::vector<std::size_t> rhs_word;
            lhs_word.reserve(word.size());
            rhs_word.reserve(word.size());

            for (Degree i = 0; i < degree; ++i) {
                if (((mask >> i) & std::uint32_t{1}) != 0) {
                    lhs_word.push_back(word[static_cast<std::size_t>(i)]);
                }
                else {
                    rhs_word.push_back(word[static_cast<std::size_t>(i)]);
                }
            }

            auto const lhs_degree = static_cast<Degree>(lhs_word.size());
            auto const rhs_degree = static_cast<Degree>(rhs_word.size());

            if (!Base::contains(lhs_range, lhs_degree) ||
                !Base::contains(rhs_range, rhs_degree)) {
                continue;
            }

            auto const lhs_index = PolynomialTensorHelper::pack_word(
                basis, lhs_word.begin(), lhs_word.end());
            auto const rhs_index = PolynomialTensorHelper::pack_word(
                basis, rhs_word.begin(), rhs_word.end());

            entry += static_cast<Accum>(lhs[static_cast<std::size_t>(
                         basis.start_of_degree(lhs_degree) + lhs_index)]) *
                static_cast<Accum>(rhs[static_cast<std::size_t>(
                    basis.start_of_degree(rhs_degree) + rhs_index)]);
        }

        return entry;
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
                if (!Base::contains(out_range, degree)) {
                    return;
                }

                expected[static_cast<std::size_t>(
                    basis.start_of_degree(degree) + level_index)] =
                    Base::scalar_from_accum(beta * shuffle_product_coefficient(
                                                       basis,
                                                       lhs,
                                                       rhs,
                                                       degree,
                                                       level_index,
                                                       lhs_range,
                                                       rhs_range));
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
                if (!Base::contains(out_range, degree)) {
                    return;
                }

                Accum value{0};
                auto const global_index = static_cast<std::size_t>(
                    basis.start_of_degree(degree) + level_index);
                if (Base::contains(a_range, degree)) {
                    value += alpha * static_cast<Accum>(a[global_index]);
                }
                value += beta * shuffle_product_coefficient(
                                    basis,
                                    b,
                                    c,
                                    degree,
                                    level_index,
                                    b_range,
                                    c_range);
                expected[global_index] = Base::scalar_from_accum(value);
            });

        return expected;
    }
};

} // namespace rpp::tests

#endif // RPP_TESTS_CPU_SINGLE_THREAD_OPS_CPU_TYPED_ST_OPS_TEST_HELPER_HPP
