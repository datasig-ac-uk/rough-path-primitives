#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_SPARSE_MATRIX_VECTOR_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_SPARSE_MATRIX_VECTOR_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/dense/batch.hpp>
#include <rpp/sparse/matrix.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/sparse_matrix_vector.hpp>

#include <rpp/gpu/operations/block/strategy.hpp>
#include <rpp/gpu/operations/block/basic/vector_set_constant.hpp>

namespace rpp::ops {
template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class SparseMatrixVectorProduct<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>,
            sparse::MatrixFormat::CSR> : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize,
            MaxBlockSize, Architecture> > {
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
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

    template<typename D, typename I, typename O>
    using Matrix = sparse::MatrixView<sparse::MatrixFormat::CSC, D, I, O>;

    template<typename D, typename I, typename O>
    using GradedMatrix = sparse::GradedMatrixView<sparse::MatrixFormat::CSC, D, I, O>;

    VectorSetConstant<Strategy> set_constant;

public:
    template<typename Basis>
    static constexpr size_t scratch_space_size(Strategy const &strategy, Basis const &basis) noexcept {
        ignore_unused(basis);
        return strategy.block_size * sizeof(Accum_);
    }

    template<typename VectorOut, typename DataIter, typename IndexIter, typename OffsetsIter, typename VectorArg>
    RPP_DEVICE void operator()(Context const &ctx, VectorOut &out,
                               VectorArg const &arg,
                               Matrix<DataIter, IndexIter, OffsetsIter> const &matrix,
                               Accum alpha = Accum{1}) const noexcept {
        set_constant(ctx, out, Accum{0});
        ctx.sync();

        auto out_it = out.begin();
        auto arg_it = arg.begin();

        const auto warp_lane = ctx.warp_lane();
        const auto warp_idx = ctx.warp_idx();

        auto *scratch = ctx.template shared_memory<Accum *>() + warp_idx * Strategy::warp_size;

        for (Index block_start = 0; block_start < matrix.inner_dim(); block_start += Strategy::warp_size) {
            scratch[warp_lane] = Accum{0};
            ctx.sync_warp();

            for (Index col = warp_idx; col < matrix.cols(); col += ctx.num_warps()) {
                const auto begin = matrix.offset(col);
                const auto end = matrix.offset(col + 1);

                for (Index idx = begin + warp_lane; idx < end; idx += Strategy::warp_size) {
                    const auto row = matrix.inner_index(idx);
                    const Accum coeff{matrix.value(idx)};
                    if (block_start <= row && row < block_start + Strategy::warp_size) {
                        const auto index = row - block_start;
                        const Accum arg_val{arg_it[row]};
                        scratch[index] += coeff * arg_val;
                    }
                }
            }

            ctx.sync();

            auto *smem = ctx.template shared_memory<Accum *>();
            for (unsigned step = ctx.num_warps(); step > 2; step /= 2) {
                if (warp_idx < step / 2) {
                    const unsigned left_idx = 2 * warp_idx;
                    const unsigned right_idx = left_idx + 1;

                    Accum left_val{smem[left_idx * Strategy::warp_size + warp_lane]};
                    Accum right_val{0};
                    if (right_idx < step) {
                        right_val = Accum{smem[right_idx * Strategy::warp_size + warp_lane]};
                    }

                    smem[warp_idx * Strategy::warp_size + warp_lane] = left_val + right_val;
                }
                ctx.sync();
            }

            if (warp_idx == 0) {
                Accum val{smem[warp_lane]};
                if (ctx.num_warps() > 1) {
                    val += Accum{smem[Strategy::warp_size + warp_lane]};
                }
                out_it[block_start + warp_lane] = val;
            }
        }
    }
};
} // namespace rpp::ops

namespace rpp::gpu::block {
template<typename BatchOut, typename Matrix, typename BatchArg, typename OutBasis, typename ArgBasis, typename Accum_,
    unsigned BlockSize,
    unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void sparse_matrix_vector_product_kernel(
    const BatchOut batch_out,
    const BatchArg batch_arg,
    const Matrix matrix,
    const OutBasis out_basis,
    const ArgBasis arg_basis,
    const strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ alpha = Accum_{1}
) {
    using Strategy = strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::SparseMatrixVectorProduct<Strategy, sparse::matrix_format_v<Matrix> > op;

    auto out = batch_out.view(my_index, out_basis);
    auto arg = batch_arg.view(my_index, arg_basis);
    op(ctx, out, arg, matrix, alpha);
}
} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_SPARSE_MATRIX_VECTOR_HPP
