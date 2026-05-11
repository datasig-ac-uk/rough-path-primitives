#ifndef RPP_OPERATIONS_BASIC_SPARSE_MATRIX_VECTOR_HPP
#define RPP_OPERATIONS_BASIC_SPARSE_MATRIX_VECTOR_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

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


} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_SPARSE_MATRIX_VECTOR_HPP
