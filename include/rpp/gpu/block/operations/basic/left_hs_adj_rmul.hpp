#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP

#include <rpp/gpu/block/strategy.hpp>
#include <rpp/operations/basic/left_hs_adj_rmul.hpp>

namespace rpp::ops {

template <typename Accum_,
          unsigned BlockSize,
          unsigned MaxBlockSize,
          typename Architecture>
class LeftHSAdjRMul<
    gpu::strategies::
        BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>>
    : public BaseOperation<
          gpu::strategies::
              BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>> {};

} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP
