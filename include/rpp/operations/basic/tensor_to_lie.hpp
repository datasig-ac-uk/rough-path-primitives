#ifndef RPP_OPERATIONS_BASIC_TENSOR_TO_LIE_HPP
#define RPP_OPERATIONS_BASIC_TENSOR_TO_LIE_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/sparse_matrix_vector.hpp>

namespace rpp::ops {


enum class T2LImplementationType {
    CSRSparseMatrix,
    CSCSparseMatrix,
};

template <typename Strategy, T2LImplementationType Implementation, typename=void>
class TensorToLie : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;

    static constexpr auto matrix_format = (Implementation == T2LImplementationType::CSRSparseMatrix)
        ? sparse::MatrixFormat::CSR : sparse::MatrixFormat::CSC;

    template <typename... Args>
    using Matrix = sparse::GradedMatrixView<matrix_format, Args...>;

    using Impl = SparseMatrixVectorProduct<Strategy, matrix_format>;

    Impl impl;

public:
    static constexpr bool is_implemented = Impl::is_implemented;

    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        return Impl::scratch_space_size(strategy, basis);
    }

    template <typename LieOut, typename TensorIn, typename... MatrixArgs>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, LieOut& out, TensorIn const& arg, Matrix<MatrixArgs...> const& matrix) const noexcept {
        impl(ctx, out, arg, matrix);
    }
};


} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_TENSOR_TO_LIE_HPP
