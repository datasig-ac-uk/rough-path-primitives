#ifndef RPP_TESTS_FT_INTERMEDIATE_TEST_HELPER_HPP
#define RPP_TESTS_FT_INTERMEDIATE_TEST_HELPER_HPP

#include <algorithm>
#include <vector>

namespace rpp::tests {

enum class IntermediateOperation { Exp, Log, FMExp };

template <typename Range>
struct IntermediateRangeCase {
    char const* name;
    Range multiplier;
    Range arg;
    bool zero_multiplier;
    bool zero_arg;
};

template <typename Range, typename Degree>
auto intermediate_range_cases(IntermediateOperation operation, Degree depth) {
    using Case = IntermediateRangeCase<Range>;
    if (operation == IntermediateOperation::FMExp) {
        return std::vector<Case>{
            {"full_views", {0, depth}, {0, depth}, false, false},
            {"unit_only_views", {0, 0}, {0, 0}, false, false},
            {"positive_minima", {1, 2}, {1, 2}, false, false},
            {"high_multiplier", {3, 4}, {1, 1}, false, false},
            {"unit_multiplier_high_exponent", {0, 0}, {3, 4}, false, false},
            {"unit_only_exponent", {3, 4}, {0, 0}, false, false},
            {"product_beyond_depth", {2, 3}, {3, 4}, false, false},
            {"zero_multiplier", {0, depth}, {0, depth}, true, false},
            {"zero_exponent", {0, depth}, {0, depth}, false, true},
            {"zero_exponent_truncated_multiplier", {1, 2}, {0, depth}, false, true},
        };
    }
    return std::vector<Case>{
        {"full_view", {0, depth}, {0, depth}, false, false},
        {"unit_only", {0, depth}, {0, 0}, false, false},
        {"positive_degrees_only", {0, depth}, {1, depth}, false, false},
        {"single_degree", {0, depth}, {1, 1}, false, false},
        {"truncated_maximum", {0, depth}, {1, 2}, false, false},
        {"interior_degrees", {0, depth}, {2, 3}, false, false},
        {"high_degrees", {0, depth}, {3, 4}, false, false},
        {"highest_degree_only", {0, depth}, {depth, depth}, false, false},
        {"zero_input", {0, depth}, {0, depth}, false, true},
    };
}

template <typename Vector, typename Basis>
auto make_intermediate_input(Vector result, Basis const& basis, bool multiplier, bool zero) {
    using Scalar = typename Vector::value_type;
    for (auto& value : result) {
        value = static_cast<Scalar>(0.0f);
    }
    if (zero) {
        return result;
    }
    result[0] = static_cast<Scalar>(multiplier ? 0.75f : 0.0f);
    for (typename Basis::Degree degree = 1; degree <= basis.depth; ++degree) {
        auto const begin = basis.start_of_degree(degree);
        auto const end = basis.end_of_degree(degree);
        result[begin] = static_cast<Scalar>(multiplier ? 0.5f : 0.0625f);
        if (end - begin > 1) {
            result[end - 1] = static_cast<Scalar>(multiplier ? -0.25f : -0.03125f);
        }
    }
    return result;
}

// Direct coefficient convolution, independent of the operation kernels.
template <typename Basis>
auto intermediate_reference_product(Basis const& basis,
                                    std::vector<double> const& lhs,
                                    std::vector<double> const& rhs) {
    using Degree = typename Basis::Degree;
    using Index = typename Basis::Index;
    std::vector<double> result(static_cast<std::size_t>(basis.size()), 0.0);
    for (Degree degree = 0; degree <= basis.depth; ++degree) {
        for (Index idx = 0; idx < basis.size_of_degree(degree); ++idx) {
            for (Degree lhs_degree = 0; lhs_degree <= degree; ++lhs_degree) {
                auto const rhs_degree = degree - lhs_degree;
                auto const splitter = basis.size_of_degree(rhs_degree);
                result[basis.start_of_degree(degree) + idx] +=
                    lhs[basis.start_of_degree(lhs_degree) + idx / splitter] *
                    rhs[basis.start_of_degree(rhs_degree) + idx % splitter];
            }
        }
    }
    return result;
}

template <typename Basis, typename Vector, typename Range>
auto reference_intermediate(Basis const& basis,
                            Vector const& multiplier,
                            Vector const& arg,
                            Range multiplier_range,
                            Range arg_range,
                            IntermediateOperation operation) {
    std::vector<double> x(static_cast<std::size_t>(basis.size()), 0.0);
    std::vector<double> m(x.size(), 0.0);
    for (typename Basis::Degree degree = 0; degree <= basis.depth; ++degree) {
        for (auto idx = basis.start_of_degree(degree); idx < basis.end_of_degree(degree); ++idx) {
            if (degree > 0 && degree >= arg_range.min && degree <= arg_range.max) {
                x[idx] = static_cast<double>(arg[idx]);
            }
            if (degree >= multiplier_range.min && degree <= multiplier_range.max) {
                m[idx] = static_cast<double>(multiplier[idx]);
            }
        }
    }
    std::vector<double> result(x.size(), 0.0);
    std::vector<double> power(x.size(), 0.0);
    power[0] = 1.0;
    if (operation != IntermediateOperation::Log) {
        result[0] = 1.0;
    }
    double factorial = 1.0;
    for (typename Basis::Degree order = 1; order <= basis.depth; ++order) {
        power = intermediate_reference_product(basis, power, x);
        factorial *= order;
        auto const coefficient = operation == IntermediateOperation::Log
            ? (order % 2 == 0 ? -1.0 : 1.0) / order : 1.0 / factorial;
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] += coefficient * power[i];
        }
    }
    if (operation == IntermediateOperation::FMExp) {
        result = intermediate_reference_product(basis, m, result);
    }
    return result;
}

} // namespace rpp::tests

#endif // RPP_TESTS_FT_INTERMEDIATE_TEST_HELPER_HPP
