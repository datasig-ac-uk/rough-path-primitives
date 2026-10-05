#ifndef RPP_TESTS_TENSOR_PAIRING_TEST_HELPER_HPP
#define RPP_TESTS_TENSOR_PAIRING_TEST_HELPER_HPP

#include <utility>

#include "tensor_antipode_test_helper.hpp"

namespace rpp::tests {

enum class PairingInput { Nonzero, ZeroFunctional, ZeroArgument, Cancelling };

template <typename Vector, typename Basis>
auto pairing_range_inputs(Vector storage, Basis const& basis, PairingInput scenario) {
    using Scalar = typename Vector::value_type;
    auto functional = storage;
    auto arg = storage;
    for (std::size_t i = 0; i < storage.size(); ++i) {
        functional[i] = static_cast<Scalar>(0.0f);
        arg[i] = static_cast<Scalar>(static_cast<float>((i * 5) % 7 + 1) / 16.0f);
    }
    for (typename Basis::Degree degree = 0; degree <= basis.depth; ++degree) {
        auto const begin = basis.start_of_degree(degree);
        auto const end = basis.end_of_degree(degree);
        if (scenario == PairingInput::Cancelling) {
            // Each whole layer cancels, including when the unit is in range.
            if (end - begin > 1) {
                functional[begin] = static_cast<Scalar>(0.5f);
                functional[begin + 1] = static_cast<Scalar>(-0.5f);
            }
        }
        else if (scenario != PairingInput::ZeroFunctional) {
            functional[begin] = static_cast<Scalar>(0.5f);
            if (end - begin > 1) {
                functional[end - 1] = static_cast<Scalar>(-0.25f);
            }
        }
    }
    if (scenario == PairingInput::ZeroArgument || scenario == PairingInput::Cancelling) {
        for (auto& value : arg) {
            value = static_cast<Scalar>(scenario == PairingInput::ZeroArgument ? 0.0f : 0.25f);
        }
    }
    // The sparse functional keeps sums exactly representable, independently
    // of CPU traversal or GPU reduction order. Nonzero storage outside operand
    // views makes accidental contributions from those degrees observable.
    return std::make_pair(std::move(functional), std::move(arg));
}

} // namespace rpp::tests

#endif // RPP_TESTS_TENSOR_PAIRING_TEST_HELPER_HPP
