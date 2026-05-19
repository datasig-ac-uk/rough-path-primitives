#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_SPARSE_MATRIX_VECTOR_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_SPARSE_MATRIX_VECTOR_HPP

#include <cstddef>


#include <cuda/atomic>

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/views/batch.hpp>
#include <rpp/sparse/matrix.hpp>
#include <rpp/support/algorithm.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/linalg/sparse_matrix_vector.hpp>

#include <rpp/gpu/operations/block/kernel.hpp>
#include <rpp/gpu/operations/block/strategy.hpp>
#include <rpp/gpu/operations/block/linalg/vector_set_constant.hpp>

namespace rpp::ops {
template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class SparseMatrixVectorProduct<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>,
            sparse::MatrixFormat::CSR> : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize,
            MaxBlockSize, Architecture> > {
public:
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;


    template<typename D, typename I, typename O>
    using Matrix = sparse::MatrixView<sparse::MatrixFormat::CSR, D, I, O>;

    template<typename D, typename I, typename O>
    using GradedMatrix = sparse::GradedMatrixView<sparse::MatrixFormat::CSR, D, I, O>;

public:
    static constexpr bool is_implemented = true;

    template<typename VectorOut, typename DataIter, typename IndexIter, typename OffsetsIter, typename VectorArg>
    RPP_DEVICE void operator()(Context const &ctx, VectorOut &out,
                               VectorArg const &arg,
                               Matrix<DataIter, IndexIter, OffsetsIter> const &matrix,
                               Accum alpha = Accum{1}) const noexcept {
        using Scalar = typename VectorOut::value_type;

        auto out_it = out.begin();
        auto arg_it = arg.begin();

        const auto warp_lane = ctx.warp_lane();
        const auto warp_idx = ctx.warp_idx();

        for (Index row = warp_idx; row < static_cast<Index>(matrix.rows()); row += ctx.num_warps()) {
            Accum acc{0};
            const auto begin = matrix.offset(row);
            const auto end = matrix.offset(row + 1);

            for (auto entry = begin + warp_lane; entry < end; entry += Strategy::warp_size) {
                Accum coeff{matrix.value(entry)};
                Accum arg_val{arg_it[matrix.inner_index(entry)]};
                acc += coeff * arg_val;
            }

            const auto result = ctx.warp_reduce(acc, std::plus<Accum>{});

            if (warp_lane == 0) {
                out_it[row] = static_cast<Scalar>(alpha * result);
            }
        }
    }
};

template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class SparseMatrixVectorProduct<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>,
            sparse::MatrixFormat::CSC> : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize,
            MaxBlockSize, Architecture> > {
public:
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

    template<typename D, typename I, typename O>
    using Matrix = sparse::MatrixView<sparse::MatrixFormat::CSC, D, I, O>;

    template<typename D, typename I, typename O>
    using GradedMatrix = sparse::GradedMatrixView<sparse::MatrixFormat::CSC, D, I, O>;

private:
    VectorSetConstant<Strategy> set_constant;
public:
    static constexpr bool is_implemented = true;
    template<typename Basis>
    static constexpr size_t scratch_space_size(Strategy const &strategy, Basis const &basis) noexcept {
        ignore_unused(basis);
        return strategy.block_size * sizeof(Accum_);
    }


private:

    using BlockAtomic = cuda::atomic_ref<Accum, cuda::thread_scope_block>;

    template <typename VectorOut, typename VectorArg, typename DataIter, typename IndexIter, typename OffsetIter>
    RPP_DEVICE void atomic_out(Context const& ctx, VectorOut& out, VectorArg const& arg, Matrix<DataIter, IndexIter, OffsetIter> const &matrix, Accum alpha) const noexcept {

        set_constant(ctx, out, Accum{0});
        ctx.sync();

        const auto col_begin = arg.begin_index();
        const auto col_end = arg.end_index();

        const auto row_begin = out.begin_index();
        const auto row_end = out.end_index();

        for (Index idx=ctx.thread_rank(), end=matrix.nnz(); idx<end; idx += ctx.num_threads()) {
            auto my_row = matrix.inner_index(idx);
            auto my_col = algo::index_upper_bound(matrix.offsets(), Index(0), matrix.outer_dim()+1, idx)-1;

            const Accum mat_val { matrix.value(idx) };
            Accum arg_val { 0 };
            if (in_range(my_col, col_begin, col_end)) {
                arg_val = Accum { arg[my_col] };
            }
            arg_val *= mat_val;

            if (in_range(my_row, row_begin, row_end)) {
                BlockAtomic out_ref { out[my_row] };
                out_ref.fetch_add(alpha * arg_val, cuda::memory_order_relaxed);
            }
        }


    }

    template <typename VectorOut, typename VectorArg, typename DataIter, typename IndexIter, typename OffsetIter>
    RPP_DEVICE static void block_wide(Context const& ctx, VectorOut& out, VectorArg const& arg, Matrix<DataIter, IndexIter, OffsetIter> const& matrix, Accum alpha) noexcept {


        auto* scratch = ctx.template shared_memory<Accum*>();

        const auto col_begin = arg.begin_index();
        const auto col_end = arg.end_index();

        const auto row_begin = out.begin_index();
        const auto row_end = out.end_index();


        for (Index row_block_begin=row_begin; row_block_begin < row_end; row_block_begin += ctx.num_threads()) {
            scratch[ctx.thread_rank()] = Accum{0};
            ctx.sync();

            for (Index col=col_begin; col < col_end; ++col) {
                auto [lower, upper] = matrix.offsets(col);
                auto block_col_begin = algo::index_lower_bound(matrix.indices(), lower, upper, row_block_begin);
                auto block_col_end = algo::index_lower_bound(matrix.indices(), block_col_begin, upper, row_block_begin + ctx.num_threads());

                auto my_offset = block_col_begin + ctx.thread_rank();
                if (my_offset < block_col_end) {
                    auto my_row = matrix.inner_index(my_offset);
                    const Accum mat_val { matrix.value(my_offset) };
                    const Accum vec_val { out[my_row] };
                    BlockAtomic out_ref { scratch[my_row - matrix.offset(block_col_begin)] };
                    out_ref.fetch_add(mat_val * vec_val, cuda::memory_order_relaxed);
                }
            }

            ctx.sync();
            auto my_row = row_block_begin + ctx.thread_rank();
            if (my_row < row_end) {
                out[my_row] = alpha*scratch[ctx.thread_rank()];
            }

        }
    }

public:
    template<typename VectorOut, typename DataIter, typename IndexIter, typename OffsetsIter, typename VectorArg>
    RPP_DEVICE void operator()(Context const &ctx, VectorOut &out,
                               VectorArg const &arg,
                               Matrix<DataIter, IndexIter, OffsetsIter> const &matrix,
                               Accum alpha = Accum{1}) const noexcept {
        atomic_out(ctx, out, arg, matrix, alpha);
        // block_wide(ctx, out, arg, matrix, alpha);
    }
};
} // namespace rpp::ops

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_SPARSE_MATRIX_VECTOR_HPP
