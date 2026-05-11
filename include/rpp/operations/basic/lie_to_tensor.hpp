#ifndef RPP_OPERATIONS_BASIC_LIE_TO_TENSOR_HPP
#define RPP_OPERATIONS_BASIC_LIE_TO_TENSOR_HPP

#include <cstddef>

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
