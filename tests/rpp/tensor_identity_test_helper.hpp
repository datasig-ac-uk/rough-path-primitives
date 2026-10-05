#ifndef RPP_TESTS_TENSOR_IDENTITY_TEST_HELPER_HPP
#define RPP_TESTS_TENSOR_IDENTITY_TEST_HELPER_HPP

#include "tensor_antipode_test_helper.hpp"

namespace rpp::tests {

template <typename Accum, typename Vector, typename Basis, typename Range>
auto reference_tensor_identity(Vector const& initial,
                               Basis const& basis,
                               Range range,
                               Accum scalar,
                               bool set_identity,
                               typename Basis::Index storage_offset) {
    using Scalar = typename Vector::value_type;
    auto expected = initial;
    if (set_identity) {
        for (auto idx = basis.start_of_degree(range.min);
             idx < basis.end_of_degree(range.max); ++idx) {
            expected[storage_offset + idx] = static_cast<Scalar>(0.0f);
        }
    }
    if (range.min == 0) {
        expected[storage_offset] = static_cast<Scalar>(set_identity ? scalar :
            static_cast<Accum>(initial[storage_offset]) + scalar);
    }
    return expected;
}

} // namespace rpp::tests

#endif // RPP_TESTS_TENSOR_IDENTITY_TEST_HELPER_HPP
