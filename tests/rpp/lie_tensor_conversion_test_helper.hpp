#ifndef RPP_TESTS_LIE_TENSOR_CONVERSION_TEST_HELPER_HPP
#define RPP_TESTS_LIE_TENSOR_CONVERSION_TEST_HELPER_HPP

#include <vector>

#include <rpp/sparse/matrix.hpp>

namespace rpp::tests {

enum class ConversionInput { Nonzero, Zero, EmptyMatrix, Cancelling };

template <typename Scalar, typename Index>
struct ConversionMatrixData {
    std::vector<Scalar> values;
    std::vector<Index> indices;
    std::vector<Index> offsets;
};

// A partial conversion map for x, y, and [x,y]. Other basis elements have
// empty rows/columns, deliberately exercising output initialization.
template <rpp::sparse::MatrixFormat Format, typename Scalar, typename Index>
auto conversion_matrix_data(bool lie_to_tensor,
                            Index tensor_size,
                            Index lie_size,
                            Index bracket,
                            Index xy,
                            Index yx,
                            bool empty) {
    struct Entry { Index row, col; Scalar value; };
    std::vector<Entry> entries;
    if (!empty) {
        entries = {{1, 1, Scalar{1}}, {2, 2, Scalar{1}}};
        if (lie_to_tensor) {
            entries.push_back({xy, bracket, Scalar{1}});
            entries.push_back({yx, bracket, Scalar{-1}});
        }
        else {
            entries.push_back({bracket, xy, Scalar{0.5}});
            entries.push_back({bracket, yx, Scalar{-0.5}});
        }
    }
    auto const rows = lie_to_tensor ? tensor_size : lie_size;
    auto const cols = lie_to_tensor ? lie_size : tensor_size;
    auto const outer_size = Format == rpp::sparse::CSRMatrix ? rows : cols;
    ConversionMatrixData<Scalar, Index> result;
    result.offsets.push_back(0);
    for (Index outer = 0; outer < outer_size; ++outer) {
        for (auto const& entry : entries) {
            if ((Format == rpp::sparse::CSRMatrix ? entry.row : entry.col) == outer) {
                result.values.push_back(entry.value);
                result.indices.push_back(Format == rpp::sparse::CSRMatrix ? entry.col : entry.row);
            }
        }
        result.offsets.push_back(static_cast<Index>(result.values.size()));
    }
    return result;
}

// Compute the expected partial conversion independently of sparse storage.
template <typename Scalar, typename Index>
auto conversion_expected(bool lie_to_tensor,
                         std::vector<Scalar> const& arg,
                         Index output_size,
                         Index bracket,
                         Index xy,
                         Index yx,
                         bool empty) {
    std::vector<Scalar> result(static_cast<std::size_t>(output_size), Scalar{0});
    if (!empty) {
        result[1] = arg[1];
        result[2] = arg[2];
        if (lie_to_tensor) {
            result[xy] = arg[bracket];
            result[yx] = -arg[bracket];
        }
        else {
            result[bracket] = Scalar{0.5} * (arg[xy] - arg[yx]);
        }
    }
    return result;
}

} // namespace rpp::tests

#endif // RPP_TESTS_LIE_TENSOR_CONVERSION_TEST_HELPER_HPP
