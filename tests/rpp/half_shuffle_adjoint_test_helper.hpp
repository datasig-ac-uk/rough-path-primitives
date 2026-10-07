#ifndef RPP_TESTS_HALF_SHUFFLE_ADJOINT_TEST_HELPER_HPP
#define RPP_TESTS_HALF_SHUFFLE_ADJOINT_TEST_HELPER_HPP

#include <cstdint>
#include <vector>

#include "shuffle_adjoint_test_helper.hpp"
#include "tensor_antipode_test_helper.hpp"

namespace rpp::tests {

// Enumerate the forward product independently of the kernel's masked packing.
// Position zero is the first letter and always comes from the left operand.
template <typename Basis, typename Range, typename Callback>
void for_each_half_shuffle_term(Basis const& basis,
                                Range range,
                                Callback callback) {
    using Degree = typename Basis::Degree;
    using Index = typename Basis::Index;
    for (Degree degree = std::max<Degree>(1, range.min); degree <= range.max;
         ++degree) {
        for (Index idx = 0; idx < basis.size_of_degree(degree); ++idx) {
            std::vector<Index> letters(static_cast<std::size_t>(degree));
            auto remaining = idx;
            for (Degree pos = degree; pos > 0; --pos) {
                letters[pos - 1] = remaining % basis.width;
                remaining /= basis.width;
            }
            for (std::uint32_t mask = 0;
                 mask < (std::uint32_t{1} << (degree - 1));
                 ++mask) {
                Degree left_degree{1}, right_degree{0};
                Index left_idx = letters[0], right_idx{0};
                for (Degree pos = 1; pos < degree; ++pos) {
                    if (mask & (std::uint32_t{1} << (pos - 1))) {
                        ++left_degree;
                        left_idx = left_idx * basis.width + letters[pos];
                    }
                    else {
                        ++right_degree;
                        right_idx = right_idx * basis.width + letters[pos];
                    }
                }
                callback(basis.start_of_degree(degree) + idx,
                         left_degree,
                         basis.start_of_degree(left_degree) + left_idx,
                         right_degree,
                         basis.start_of_degree(right_degree) + right_idx);
            }
        }
    }
}

template <typename Basis, typename Vector, typename Range>
auto reference_half_shuffle_adjoint(Basis const& basis,
                                    Vector const& op,
                                    Vector const& arg,
                                    Range out_range,
                                    Range op_range,
                                    Range arg_range,
                                    bool fixed_left,
                                    double beta) {
    std::vector<double> result(static_cast<std::size_t>(basis.size()), 0.0);
    for_each_half_shuffle_term(
        basis,
        arg_range,
        [&](auto arg_idx,
            auto left_degree,
            auto left_idx,
            auto right_degree,
            auto right_idx) {
            auto const op_degree = fixed_left ? left_degree : right_degree;
            auto const out_degree = fixed_left ? right_degree : left_degree;
            if (op_degree < op_range.min || op_degree > op_range.max ||
                out_degree < out_range.min || out_degree > out_range.max) {
                return;
            }
            auto const op_idx = fixed_left ? left_idx : right_idx;
            auto const out_idx = fixed_left ? right_idx : left_idx;
            result[out_idx] += beta * static_cast<double>(op[op_idx]) *
                static_cast<double>(arg[arg_idx]);
        });
    return result;
}

template <typename Range, typename Degree>
auto half_shuffle_adjoint_range_cases(Degree depth) {
    using Case = ShuffleAdjointRangeCase<Range>;
    std::vector<Case> result;
    if (depth <= 2) {
        auto const ranges = all_tensor_degree_ranges<Range>(depth);
        for (auto const out : ranges) {
            for (auto const op : ranges) {
                for (auto const arg : ranges) {
                    result.push_back({"all_small_ranges", out, op, arg});
                }
            }
        }
    }
    else {
        auto const cases = shuffle_adjoint_range_cases<Range>();
        result.assign(cases.begin(), cases.end());
        result.push_back({"mixed_words", {1, 2}, {1, 2}, {2, 4}});
        result.push_back(
            {"unit_cotangent_only", {0, depth}, {0, depth}, {0, 0}});
    }
    return result;
}

template <typename Scalar, typename Basis>
auto make_half_shuffle_adjoint_operator(Basis const& basis) {
    std::vector<Scalar> result(static_cast<std::size_t>(basis.size()),
                               static_cast<Scalar>(0.0f));
    for (typename Basis::Degree degree = 0; degree <= basis.depth; ++degree) {
        auto const begin = basis.start_of_degree(degree);
        auto const size = basis.size_of_degree(degree);
        result[begin] = static_cast<Scalar>(0.5f);
        if (size > 1) {
            result[begin + size - 1] = static_cast<Scalar>(-0.5f);
        }
        if (size > 2) {
            result[begin + size / 2] = static_cast<Scalar>(0.25f);
        }
    }
    return result;
}

template <typename Scalar, typename Basis>
auto make_half_shuffle_cotangent(Basis const& basis) {
    std::vector<Scalar> result(static_cast<std::size_t>(basis.size()));
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<Scalar>(
            static_cast<float>(int((i * 13) % 5) - 2) / 8.0f);
    }
    return result;
}

} // namespace rpp::tests

#endif // RPP_TESTS_HALF_SHUFFLE_ADJOINT_TEST_HELPER_HPP
