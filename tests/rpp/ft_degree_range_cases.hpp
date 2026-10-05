#ifndef RPP_TESTS_FT_DEGREE_RANGE_CASES_HPP
#define RPP_TESTS_FT_DEGREE_RANGE_CASES_HPP

#include <array>

namespace rpp::tests {

// All cases use depth four. Storage outside each operand view is deliberately
// populated too, so accidental reads beyond a view affect the reference check.

template <typename Range>
struct FtMulDegreeRangeCase {
    char const* name;
    Range out;
    Range lhs;
    Range rhs;
};

template <typename Range>
constexpr auto ft_mul_degree_range_cases() {
    return std::array{
        FtMulDegreeRangeCase<Range>{"positive_minima", {0, 4}, {1, 2}, {1, 2}},
        FtMulDegreeRangeCase<Range>{"disjoint_operands", {0, 4}, {1, 1}, {2, 2}},
        FtMulDegreeRangeCase<Range>{"disjoint_operands_reversed", {0, 4}, {2, 2}, {1, 1}},
        FtMulDegreeRangeCase<Range>{"lhs_unit_only", {0, 4}, {0, 0}, {2, 3}},
        FtMulDegreeRangeCase<Range>{"rhs_unit_only", {0, 4}, {2, 3}, {0, 0}},
        FtMulDegreeRangeCase<Range>{"both_unit_only", {0, 4}, {0, 0}, {0, 0}},
        FtMulDegreeRangeCase<Range>{"below_product", {0, 1}, {1, 2}, {1, 2}},
        FtMulDegreeRangeCase<Range>{"above_product", {3, 4}, {1, 1}, {1, 1}},
        FtMulDegreeRangeCase<Range>{"product_beyond_depth", {0, 4}, {3, 4}, {2, 3}},
        FtMulDegreeRangeCase<Range>{"unit_output_lhs_absent", {0, 0}, {1, 2}, {0, 2}},
        FtMulDegreeRangeCase<Range>{"unit_output_rhs_absent", {0, 0}, {0, 2}, {1, 2}},
        FtMulDegreeRangeCase<Range>{"interior_output", {1, 3}, {1, 1}, {1, 1}},
    };
}

template <typename Range>
struct FtFmaDegreeRangeCase {
    char const* name;
    Range out;
    Range a;
    Range b;
    Range c;
};

template <typename Range>
constexpr auto ft_fma_degree_range_cases() {
    return std::array{
        FtFmaDegreeRangeCase<Range>{"addend_below_product", {0, 4}, {0, 1}, {1, 2}, {1, 2}},
        FtFmaDegreeRangeCase<Range>{"addend_above_product", {0, 4}, {3, 4}, {1, 1}, {1, 1}},
        FtFmaDegreeRangeCase<Range>{"gap_between_addend_and_product", {0, 4}, {0, 0}, {2, 2}, {2, 2}},
        FtFmaDegreeRangeCase<Range>{"disjoint_product_operands", {0, 4}, {1, 1}, {1, 1}, {2, 2}},
        FtFmaDegreeRangeCase<Range>{"disjoint_product_operands_reversed", {0, 4}, {1, 1}, {2, 2}, {1, 1}},
        FtFmaDegreeRangeCase<Range>{"lhs_unit_only", {0, 4}, {1, 1}, {0, 0}, {2, 3}},
        FtFmaDegreeRangeCase<Range>{"rhs_unit_only", {0, 4}, {1, 1}, {2, 3}, {0, 0}},
        FtFmaDegreeRangeCase<Range>{"both_unit_only", {0, 4}, {0, 0}, {0, 0}, {0, 0}},
        FtFmaDegreeRangeCase<Range>{"below_all_contributions", {0, 1}, {2, 3}, {1, 2}, {1, 2}},
        FtFmaDegreeRangeCase<Range>{"above_all_contributions", {3, 4}, {1, 2}, {1, 1}, {1, 1}},
        FtFmaDegreeRangeCase<Range>{"only_addend_contributes", {0, 4}, {1, 2}, {3, 4}, {2, 3}},
        FtFmaDegreeRangeCase<Range>{"unit_output_lhs_absent", {0, 0}, {1, 2}, {1, 2}, {0, 2}},
        FtFmaDegreeRangeCase<Range>{"unit_output_rhs_absent", {0, 0}, {1, 2}, {0, 2}, {1, 2}},
        FtFmaDegreeRangeCase<Range>{"unit_output_addend_only", {0, 0}, {0, 0}, {1, 2}, {0, 2}},
        FtFmaDegreeRangeCase<Range>{"interior_output", {1, 3}, {3, 3}, {1, 1}, {1, 1}},
    };
}

template <typename Range>
struct FtInplaceDegreeRangeCase {
    char const* name;
    Range a;
    Range b;
    Range c;
};

template <typename Range>
constexpr auto ft_inplace_degree_range_cases() {
    return std::array{
        FtInplaceDegreeRangeCase<Range>{"full_views", {0, 4}, {0, 4}, {0, 4}},
        FtInplaceDegreeRangeCase<Range>{"positive_minima", {1, 4}, {1, 2}, {1, 2}},
        FtInplaceDegreeRangeCase<Range>{"disjoint_ranges", {2, 4}, {1, 1}, {0, 0}},
        FtInplaceDegreeRangeCase<Range>{"unit_multiplier", {1, 4}, {0, 0}, {2, 2}},
        FtInplaceDegreeRangeCase<Range>{"unit_destination", {0, 0}, {0, 0}, {0, 0}},
        FtInplaceDegreeRangeCase<Range>{"unit_destination_missing_operands", {0, 0}, {1, 2}, {1, 2}},
        FtInplaceDegreeRangeCase<Range>{"low_output_missing_addend", {0, 2}, {0, 1}, {3, 4}},
        FtInplaceDegreeRangeCase<Range>{"high_output_missing_addend", {3, 4}, {0, 1}, {0, 2}},
        FtInplaceDegreeRangeCase<Range>{"product_above_output", {0, 2}, {3, 4}, {3, 4}},
        FtInplaceDegreeRangeCase<Range>{"product_below_output", {3, 4}, {0, 1}, {0, 1}},
        FtInplaceDegreeRangeCase<Range>{"interior_output", {1, 3}, {1, 1}, {3, 3}},
        FtInplaceDegreeRangeCase<Range>{"single_high_degree", {4, 4}, {0, 1}, {1, 2}},
    };
}

} // namespace rpp::tests

#endif // RPP_TESTS_FT_DEGREE_RANGE_CASES_HPP
