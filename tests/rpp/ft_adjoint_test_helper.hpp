#ifndef RPP_TESTS_FT_ADJOINT_TEST_HELPER_HPP
#define RPP_TESTS_FT_ADJOINT_TEST_HELPER_HPP

#include "shuffle_adjoint_test_helper.hpp"

namespace rpp::tests {

template <typename Range, typename Degree>
auto ft_adjoint_range_cases(Degree depth) {
    auto const common = shuffle_adjoint_range_cases<Range>();
    std::vector<ShuffleAdjointRangeCase<Range>> result(common.begin(), common.end());
    result.front() = {"full_views", {0, depth}, {0, depth}, {0, depth}};
    // Retain the exact regressions from the earlier zero-extension tests.
    result.push_back({"clear_only_output_view", {1, 2}, {3, 3}, {0, 0}});
    result.push_back({"disjoint_inputs_with_positive_output", {0, 1}, {1, 1}, {2, 2}});
    result.push_back({"disjoint_inputs_without_contributions", {0, 1}, {2, 2}, {1, 1}});
    result.push_back({"unit_only_views", {0, 0}, {0, 0}, {0, 0}});
    return result;
}

template <typename Vector, typename Basis>
auto make_ft_adjoint_range_operator(Vector result, Basis const& basis) {
    using Scalar = typename Vector::value_type;
    for (auto& value : result) {
        value = static_cast<Scalar>(0.0f);
    }
    for (typename Basis::Degree degree = 0; degree <= basis.depth; ++degree) {
        auto const begin = basis.start_of_degree(degree);
        auto const end = basis.end_of_degree(degree);
        result[begin] = static_cast<Scalar>(0.5f);
        if (end - begin > 1) {
            result[end - 1] = static_cast<Scalar>(-0.25f);
        }
    }
    return result;
}

template <typename Vector>
auto make_ft_adjoint_range_argument(Vector result) {
    using Scalar = typename Vector::value_type;
    for (std::size_t i = 0; i < result.size(); ++i) {
        auto value = static_cast<int>((i * 5) % 7) - 3;
        if (value == 0) {
            value = 4;
        }
        result[i] = static_cast<Scalar>(static_cast<float>(value) / 8.0f);
    }
    return result;
}

// Transpose concatenation by splitting each argument word into prefix/suffix.
// The operator is the prefix for the left adjoint and suffix for the right.
// Sparse dyadic operator values keep every accumulation exactly representable
// in the GPU scalar/accumulator types, independently of reduction order.
template <typename Accum, typename Basis, typename Vector, typename Range>
auto reference_ft_adjoint(Basis const& basis,
                          Vector const& op,
                          Vector const& arg,
                          Range out_range,
                          Range op_range,
                          Range arg_range,
                          bool left_adjoint) {
    using Degree = typename Basis::Degree;
    using Index = typename Basis::Index;
    std::vector<Accum> result(static_cast<std::size_t>(basis.size()), Accum{0});
    for (Degree degree = arg_range.min; degree <= arg_range.max; ++degree) {
        for (Index idx = 0; idx < basis.size_of_degree(degree); ++idx) {
            for (Degree op_degree = op_range.min;
                 op_degree <= op_range.max && op_degree <= degree; ++op_degree) {
                auto const out_degree = static_cast<Degree>(degree - op_degree);
                if (out_degree < out_range.min || out_degree > out_range.max) {
                    continue;
                }
                auto const suffix_size = basis.size_of_degree(
                    left_adjoint ? out_degree : op_degree);
                auto const prefix_idx = idx / suffix_size;
                auto const suffix_idx = idx % suffix_size;
                auto const op_idx = left_adjoint ? prefix_idx : suffix_idx;
                auto const out_idx = left_adjoint ? suffix_idx : prefix_idx;
                result[basis.start_of_degree(out_degree) + out_idx] +=
                    static_cast<Accum>(op[basis.start_of_degree(op_degree) + op_idx]) *
                    static_cast<Accum>(arg[basis.start_of_degree(degree) + idx]);
            }
        }
    }
    return result;
}

} // namespace rpp::tests

#endif // RPP_TESTS_FT_ADJOINT_TEST_HELPER_HPP
