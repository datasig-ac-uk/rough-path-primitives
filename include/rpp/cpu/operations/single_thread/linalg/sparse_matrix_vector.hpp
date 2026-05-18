#ifndef RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_SPARSE_MATRIX_VECTOR_HPP
#define RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_SPARSE_MATRIX_VECTOR_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/dense/batch.hpp>
#include <rpp/sparse/matrix.hpp>

#include <rpp/operations/linalg/sparse_matrix_vector.hpp>

#include <rpp/cpu/operations/single_thread/linalg/vector_set_constant.hpp>
#include <rpp/cpu/operations/single_thread/detail/batch_wrapper.hpp>
#include <rpp/cpu/operations/single_thread/strategy.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture_>
class SparseMatrixVectorProduct<
    cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>,
    sparse::MatrixFormat::CSR>
    : public BaseOperation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

    template <typename D, typename I, typename O>
    using Matrix = sparse::MatrixView<sparse::MatrixFormat::CSR, D, I, O>;

    template <typename D, typename I, typename O>
    using GradedMatrix =
        sparse::GradedMatrixView<sparse::MatrixFormat::CSR, D, I, O>;

public:
    static constexpr bool is_implemented = true;

    template <typename VectorOut,
              typename DataIter,
              typename IndexIter,
              typename OffsetsIter,
              typename VectorArg>
    void operator()(Context const& ctx,
                    VectorOut& out,
                    VectorArg const& arg,
                    Matrix<DataIter, IndexIter, OffsetsIter> const& matrix,
                    Accum alpha = Accum{1}) const noexcept {
        using Scalar = typename VectorOut::value_type;
        ignore_unused(ctx);

        auto out_it = out.begin();
        auto arg_it = arg.begin();

        for (Index row = 0; row < static_cast<Index>(matrix.rows()); ++row) {
            Accum acc{0};
            const auto begin = matrix.offset(row);
            const auto end = matrix.offset(row + 1);

            for (auto entry = begin; entry < end; ++entry) {
                acc += Accum{matrix.value(entry)} *
                    Accum{arg_it[matrix.inner_index(entry)]};
            }

            out_it[row] = static_cast<Scalar>(alpha * acc);
        }
    }

    template <typename VectorOut,
              typename DataIter,
              typename IndexIter,
              typename OffsetsIter,
              typename VectorArg>
    void
    operator()(Context const& ctx,
               VectorOut& out,
               VectorArg const& arg,
               GradedMatrix<DataIter, IndexIter, OffsetsIter> const& matrix,
               Accum alpha = Accum{1}) const noexcept {
        return operator()(
            ctx,
            out,
            arg,
            static_cast<Matrix<DataIter, IndexIter, OffsetsIter> const&>(
                matrix),
            alpha);
    }
};


template <typename Accum_, typename Architecture_>
class SparseMatrixVectorProduct<
    cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>,
    sparse::MatrixFormat::CSC>
    : public BaseOperation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

    VectorSetConstant<Strategy> set_constant;

    template <typename D, typename I, typename O>
    using Matrix = sparse::MatrixView<sparse::MatrixFormat::CSC, D, I, O>;

    template <typename D, typename I, typename O>
    using GradedMatrix =
        sparse::GradedMatrixView<sparse::MatrixFormat::CSC, D, I, O>;

public:
    static constexpr bool is_implemented = true;

    template <typename Basis>
    static constexpr std::size_t
    scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        return VectorSetConstant<Strategy>::scratch_space_size(strategy, basis);
    }

    template <typename VectorOut,
              typename DataIter,
              typename IndexIter,
              typename OffsetsIter,
              typename VectorArg>
    void operator()(Context const& ctx,
                    VectorOut& out,
                    VectorArg const& arg,
                    Matrix<DataIter, IndexIter, OffsetsIter> const& matrix,
                    Accum alpha = Accum{1}) const noexcept {
        using Scalar = typename VectorOut::value_type;
        ignore_unused(ctx);

        set_constant(ctx, out, Accum{0});

        auto out_it = out.begin();
        auto arg_it = arg.begin();

        for (Index col = 0; col < static_cast<Index>(matrix.cols()); ++col) {
            const auto begin = matrix.offset(col);
            const auto end = matrix.offset(col + 1);
            const Accum arg_value = alpha * Accum{arg_it[col]};

            for (auto entry = begin; entry < end; ++entry) {
                const auto row = matrix.inner_index(entry);
                const Accum result =
                    Accum{out_it[row]} + Accum{matrix.value(entry)} * arg_value;
                out_it[row] = static_cast<Scalar>(result);
            }
        }
    }

    template <typename VectorOut,
              typename DataIter,
              typename IndexIter,
              typename OffsetsIter,
              typename VectorArg>
    void
    operator()(Context const& ctx,
               VectorOut& out,
               VectorArg const& arg,
               GradedMatrix<DataIter, IndexIter, OffsetsIter> const& matrix,
               Accum alpha = Accum{1}) const noexcept {
        return operator()(
            ctx,
            out,
            arg,
            static_cast<Matrix<DataIter, IndexIter, OffsetsIter> const&>(
                matrix),
            alpha);
    }
};


} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchOut,
          typename Matrix,
          typename BatchArg,
          typename OutBasis,
          typename ArgBasis,
          typename Accum_,
          typename Architecture>
void sparse_matrix_vector_product_kernel(
    const BatchOut batch_out,
    const BatchArg batch_arg,
    const Matrix matrix,
    const OutBasis out_basis,
    const ArgBasis arg_basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ alpha = Accum_{1}) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::SparseMatrixVectorProduct<Strategy,
                                              sparse::matrix_format_v<Matrix>>;

    detail::apply_batch<Op>(out_basis,
                            strategy,
                            n_tensors,
                            [&](Op const& op,
                                typename Strategy::Context const& ctx,
                                typename Strategy::Index idx) {
                                auto out = batch_out.view(idx, out_basis);
                                auto arg = batch_arg.view(idx, arg_basis);
                                op(ctx, out, arg, matrix, alpha);
                            });
}

} // namespace rpp::cpu::single_thread

#endif // RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_SPARSE_MATRIX_VECTOR_HPP
