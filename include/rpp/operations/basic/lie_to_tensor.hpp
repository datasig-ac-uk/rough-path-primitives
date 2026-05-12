#ifndef RPP_OPERATIONS_BASIC_LIE_TO_TENSOR_HPP
#define RPP_OPERATIONS_BASIC_LIE_TO_TENSOR_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/sparse_matrix_vector.hpp>

namespace rpp::ops {

enum class L2TImplementationType {
    // CommutatorExpansion,
    CSRSparseMatrix,
    CSCSparseMatrix,
};

template <typename Strategy, L2TImplementationType Implementation, typename=void>
class LieToTensor : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;
    static_assert((Implementation == L2TImplementationType::CSRSparseMatrix
        || Implementation == L2TImplementationType::CSCSparseMatrix),
        "Commutator expansion does not have a default implementation");

    static constexpr auto matrix_format = (Implementation == L2TImplementationType::CSRSparseMatrix)
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

    template <typename TensorOut, typename LieIn, typename... MatrixArgs>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, TensorOut& out, LieIn const& arg, Matrix<MatrixArgs...> const& matrix) const noexcept {
        impl(ctx, out, arg, matrix);
    }
};

template <typename Strategy, typename TensorBatchOut, typename LieBatchIn, typename Matrix, typename Bases>
auto lie_to_tensor(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    TensorBatchOut const& out,
    LieBatchIn const& arg,
    Matrix const& matrix,
    Bases const& bases,
    typename Strategy::Index batch_size
    ) noexcept {
    static constexpr auto implementation = sparse::matrix_format_v<Matrix> == sparse::MatrixFormat::CSR
        ? L2TImplementationType::CSRSparseMatrix
        : L2TImplementationType::CSCSparseMatrix;
    using Op = LieToTensor<Strategy, implementation>;

    static_assert(
        Op::is_implemented,
        "The operation object \"LieToTensor\" that implements \"lie_to_tensor\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/basic/lie_to_tensor.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, arg),
        bases,
        batch_size,
        matrix
        );
}

// template <typename Strategy> class LieToTensor<Strategy, L2TImplementationType::CommutatorExpansion> {
//     using Context = typename Strategy::Context;
// public:
//
//     template <typename Basis>
//     static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
//         ignore_unused(strategy, basis);
//         return 0;
//     }
//
//     template <typename TensorOut, typename LieIn>
//     RPP_HOST_DEVICE
//     void operator()(Context const& ctx, TensorOut& out, LieIn const& arg) const noexcept {
//         static_assert(
//             static_assert_fail<Strategy, Context, TensorOut, LieIn>,
//             "rpp::ops::LieToTensor has no implementation for this Strategy. "
//             "Use an operation specialization for the selected strategy and include its header."
//         );
//     }
//
// };


} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_LIE_TO_TENSOR_HPP
