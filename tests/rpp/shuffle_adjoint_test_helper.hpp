#ifndef RPP_TESTS_SHUFFLE_ADJOINT_TEST_HELPER_HPP
#define RPP_TESTS_SHUFFLE_ADJOINT_TEST_HELPER_HPP

#include <array>
#include <cstdint>
#include <vector>

namespace rpp::tests {

template <typename Range>
struct ShuffleAdjointRangeCase {
    char const* name;
    Range out;
    Range op;
    Range arg;
};

template <typename Range>
constexpr auto shuffle_adjoint_range_cases() {
    return std::array{
        ShuffleAdjointRangeCase<Range>{"full_views", {0, 4}, {0, 4}, {0, 4}},
        ShuffleAdjointRangeCase<Range>{"unit_operator", {0, 4}, {0, 0}, {1, 2}},
        ShuffleAdjointRangeCase<Range>{"positive_operator_minimum", {0, 4}, {1, 2}, {2, 3}},
        ShuffleAdjointRangeCase<Range>{"single_degree_operands", {0, 4}, {1, 1}, {3, 3}},
        ShuffleAdjointRangeCase<Range>{"operator_above_argument", {0, 4}, {3, 4}, {0, 2}},
        ShuffleAdjointRangeCase<Range>{"output_below_support", {0, 1}, {0, 1}, {3, 4}},
        ShuffleAdjointRangeCase<Range>{"output_above_support", {3, 4}, {1, 2}, {1, 2}},
        ShuffleAdjointRangeCase<Range>{"unit_argument", {0, 4}, {0, 0}, {0, 0}},
        ShuffleAdjointRangeCase<Range>{"unit_argument_missing_operator", {0, 4}, {1, 2}, {0, 0}},
        ShuffleAdjointRangeCase<Range>{"unit_output_present", {0, 0}, {2, 2}, {2, 2}},
        ShuffleAdjointRangeCase<Range>{"unit_output_absent", {0, 0}, {1, 1}, {2, 2}},
        ShuffleAdjointRangeCase<Range>{"interior_output", {1, 3}, {0, 1}, {2, 3}},
        ShuffleAdjointRangeCase<Range>{"single_high_output", {4, 4}, {0, 0}, {4, 4}},
    };
}

// Two repeated-letter words per layer exercise shuffle multiplicities. Their
// dyadic coefficients keep even low-precision atomic sums exactly representable.
template <typename Vector, typename Basis>
auto make_sparse_shuffle_adjoint_operand(Vector result, Basis const& basis) {
    using Scalar = typename Vector::value_type;
    for (auto& value : result) {
        value = static_cast<Scalar>(0.0f);
    }
    for (typename Basis::Degree degree = 0; degree <= basis.depth; ++degree) {
        result[basis.start_of_degree(degree)] = static_cast<Scalar>(1.0f);
        result[basis.end_of_degree(degree) - 1] = static_cast<Scalar>(1.0f);
    }
    return result;
}

// Transpose the shuffle product by independently enumerating every partition
// of each argument word. Do not use the operation's word-packing helpers.
template <typename Accum, typename Basis, typename Vector, typename Range>
auto reference_shuffle_adjoint(Basis const& basis,
                               Vector const& op,
                               Vector const& arg,
                               Range out_range,
                               Range op_range,
                               Range arg_range) {
    using Degree = typename Basis::Degree;
    using Index = typename Basis::Index;
    std::vector<Accum> result(static_cast<std::size_t>(basis.size()), Accum{0});
    for (Degree degree = arg_range.min; degree <= arg_range.max; ++degree) {
        for (Index idx = 0; idx < basis.size_of_degree(degree); ++idx) {
            std::vector<Index> letters(static_cast<std::size_t>(degree));
            auto remainder = idx;
            for (Degree pos = degree; pos > 0; --pos) {
                letters[pos - 1] = remainder % basis.width;
                remainder /= basis.width;
            }
            for (std::uint32_t mask = 0; mask < (std::uint32_t{1} << degree); ++mask) {
                Degree op_degree{0}, out_degree{0};
                Index op_idx{0}, out_idx{0};
                for (Degree pos = 0; pos < degree; ++pos) {
                    if (mask & (std::uint32_t{1} << pos)) {
                        ++op_degree;
                        op_idx = op_idx * basis.width + letters[pos];
                    }
                    else {
                        ++out_degree;
                        out_idx = out_idx * basis.width + letters[pos];
                    }
                }
                if (op_degree < op_range.min || op_degree > op_range.max ||
                    out_degree < out_range.min || out_degree > out_range.max) {
                    continue;
                }
                result[basis.start_of_degree(out_degree) + out_idx] +=
                    static_cast<Accum>(op[basis.start_of_degree(op_degree) + op_idx]) *
                    static_cast<Accum>(arg[basis.start_of_degree(degree) + idx]);
            }
        }
    }
    return result;
}

} // namespace rpp::tests

#endif // RPP_TESTS_SHUFFLE_ADJOINT_TEST_HELPER_HPP
