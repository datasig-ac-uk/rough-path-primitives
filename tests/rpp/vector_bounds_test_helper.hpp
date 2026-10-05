#ifndef RPP_TESTS_VECTOR_BOUNDS_TEST_HELPER_HPP
#define RPP_TESTS_VECTOR_BOUNDS_TEST_HELPER_HPP

#include <limits>

#include "tensor_antipode_test_helper.hpp"

namespace rpp::tests {

enum class VectorBoundOperation { Set, Scale, InplaceAdd };

template <typename Scalar, typename Basis>
auto make_vector_bounds_storage(Basis const& basis) {
    std::vector<Scalar> result(static_cast<std::size_t>(basis.size()) + 2);
    result.front() = static_cast<Scalar>(7.0f);
    result.back() = static_cast<Scalar>(-7.0f);
    for (std::size_t i = 1; i + 1 < result.size(); ++i) {
        result[i] = static_cast<Scalar>(static_cast<float>(int(i % 5) - 2) / 4.0f);
    }
    return result;
}

template <typename Vector, typename Basis, typename Range>
auto reference_vector_bounds(Vector const& initial, Vector const& rhs,
                             Basis const& basis, Range out_range, Range rhs_range,
                             float scalar, VectorBoundOperation operation) {
    using Scalar = typename Vector::value_type;
    auto expected = initial;
    for (auto degree = out_range.min; degree <= out_range.max; ++degree) {
        for (auto idx = basis.start_of_degree(degree);
             idx < basis.end_of_degree(degree); ++idx) {
            double value = scalar;
            if (operation == VectorBoundOperation::Scale) {
                value *= static_cast<double>(initial[idx + 1]);
            }
            else if (operation == VectorBoundOperation::InplaceAdd) {
                value = static_cast<double>(initial[idx + 1]);
                if (rhs_range.min <= degree && degree <= rhs_range.max) {
                    value += scalar * static_cast<double>(rhs[idx + 1]);
                }
            }
            expected[idx + 1] = static_cast<Scalar>(value);
        }
    }
    return expected;
}

} // namespace rpp::tests

#endif // RPP_TESTS_VECTOR_BOUNDS_TEST_HELPER_HPP
