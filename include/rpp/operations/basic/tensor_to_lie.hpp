#ifndef RPP_OPERATIONS_BASIC_TENSOR_TO_LIE_HPP
#define RPP_OPERATIONS_BASIC_TENSOR_TO_LIE_HPP

#include <cstddef>
#include <tuple>
#include <utility>

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

template <typename Strategy, typename LieBatchOut, typename TensorBatchIn, typename Matrix, typename Bases>
auto tensor_to_lie(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    LieBatchOut const& out,
    TensorBatchIn const& arg,
    Matrix const& matrix,
    Bases const& bases,
    typename Strategy::Index batch_size
    ) noexcept {
    static constexpr auto implementation = sparse::matrix_format_v<Matrix> == sparse::MatrixFormat::CSR
        ? T2LImplementationType::CSRSparseMatrix
        : T2LImplementationType::CSCSparseMatrix;
    using Op = TensorToLie<Strategy, implementation>;

    static_assert(
        Op::is_implemented,
        "The operation object \"TensorToLie\" that implements \"tensor_to_lie\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/basic/tensor_to_lie.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, arg),
        bases,
        batch_size,
        matrix
        );
}


} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_TENSOR_TO_LIE_HPP
