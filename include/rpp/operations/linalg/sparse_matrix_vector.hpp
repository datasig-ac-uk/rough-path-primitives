#ifndef RPP_OPERATIONS_BASIC_SPARSE_MATRIX_VECTOR_HPP
#define RPP_OPERATIONS_BASIC_SPARSE_MATRIX_VECTOR_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/basis/basis_pack.hpp>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/sparse/matrix.hpp>

namespace rpp::ops {

template <typename Strategy, sparse::MatrixFormat Format, typename=void>
class SparseMatrixVectorProduct : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    template <typename... Args>
    using MatrixView = typename sparse::MatrixView<Format, Args...>::type;

public:
    static constexpr bool is_implemented = false;

    template <typename VectorOut, typename DataIter, typename IndexIter, typename OffsetsIter, typename VectorArg>
    RPP_HOST_DEVICE
    void operator()(
        Context const& ctx,
        VectorOut& out,
        VectorArg const& arg,
        MatrixView<DataIter, IndexIter, OffsetsIter> const& matrix,
        Accum alpha = Accum{1}
    ) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, VectorOut, DataIter, IndexIter, OffsetsIter, VectorArg, Accum>,
            "rpp::ops::SparseMatrixVectorProduct has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

template <typename Strategy,
          typename BatchOut,
          typename BatchArg,
          typename Matrix,
          typename OutBasis,
          typename ArgBasis
          >
auto sparse_matrix_vector_product(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchOut const& out,
    BatchArg const& arg,
    OutBasis const& out_basis,
    ArgBasis const& arg_basis,
    typename Strategy::Index num_batches,
    Matrix const& matrix,
    typename Strategy::Accum alpha = typename Strategy::Accum{1}
    ) noexcept {
    static constexpr auto format = sparse::matrix_format_v<Matrix>;
    using Op = SparseMatrixVectorProduct<Strategy, format>;

    static_assert(
        Op::is_implemented,
        "The operation object \"SparseMatrixVectorProduct\" that implements "
        "\"sparse_matrix_vector_product\" is not implemented. This either means "
        "that the Strategy object is invalid, or that the necessary specialisation "
        "headers have not been included. For example, you may need to add the "
        "following include directive to bring in the single-threaded CPU "
        "implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/linalg/sparse_matrix_vector.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(tag_batch(out, OutputBasisTagger{}), tag_batch(arg, InputBasisTagger{})),
        make_basis_pack(basis::out(out_basis), basis::in(arg_basis)),
        num_batches,
        matrix,
        alpha
        );
}


} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_SPARSE_MATRIX_VECTOR_HPP
