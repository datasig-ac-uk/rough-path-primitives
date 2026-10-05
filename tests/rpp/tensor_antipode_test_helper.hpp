#ifndef RPP_TESTS_TENSOR_ANTIPODE_TEST_HELPER_HPP
#define RPP_TESTS_TENSOR_ANTIPODE_TEST_HELPER_HPP

#include <vector>

namespace rpp::tests {

template <typename Range, typename Degree>
auto all_tensor_degree_ranges(Degree depth) {
    std::vector<Range> result;
    for (Degree min = 0; min <= depth; ++min) {
        for (Degree max = min; max <= depth; ++max) {
            result.push_back(Range{min, max});
        }
    }
    return result;
}

template <typename Vector>
auto make_antipode_range_argument(Vector result) {
    using Scalar = typename Vector::value_type;
    for (std::size_t i = 0; i < result.size(); ++i) {
        // Dyadic values are exact even in the low-precision GPU scalar types.
        // Vary within each layer so an incorrect word reversal is observable.
        result[i] = static_cast<Scalar>(
            static_cast<float>((i * 37) % 127 + 1) / 128.0f);
    }
    return result;
}

template <typename Accum, typename Vector, typename Basis, typename Range>
auto reference_tensor_antipode(Vector const& initial_out,
                              Vector const& arg,
                              Basis const& basis,
                              Range out_range,
                              Range arg_range,
                              bool sign_by_degree) {
    using Scalar = typename Vector::value_type;
    using Degree = typename Basis::Degree;
    using Index = typename Basis::Index;
    auto expected = initial_out;
    for (Degree degree = out_range.min; degree <= out_range.max; ++degree) {
        auto const begin = basis.start_of_degree(degree);
        for (Index i = 0; i < basis.size_of_degree(degree); ++i) {
            if (degree < arg_range.min || degree > arg_range.max) {
                expected[begin + i] = static_cast<Scalar>(0.0f);
                continue;
            }
            // Reverse the base-width digits independently of reverse_index.
            Index source = i;
            Index reversed{0};
            for (Degree letter = 0; letter < degree; ++letter) {
                reversed = reversed * basis.width + source % basis.width;
                source /= basis.width;
            }
            auto value = static_cast<Accum>(arg[begin + reversed]);
            if (sign_by_degree && degree % 2 != 0) {
                value = -value;
            }
            expected[begin + i] = static_cast<Scalar>(value);
        }
    }
    return expected;
}

} // namespace rpp::tests

#endif // RPP_TESTS_TENSOR_ANTIPODE_TEST_HELPER_HPP
