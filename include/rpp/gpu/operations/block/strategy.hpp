#ifndef RPP_GPU_OPERATIONS_BLOCK_STRATEGY_HPP
#define RPP_GPU_OPERATIONS_BLOCK_STRATEGY_HPP

#include <cstddef>

#include <cuda_runtime.h>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/gpu/architecture.hpp>

namespace rpp::gpu::strategies {

namespace detail {

template <typename Strategy_>
class SmallBlockContext {
public:
    using Strategy = Strategy_;
    using BlockReduceArray = typename Strategy::BlockReduceArray;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Accum = typename Strategy::Accum;
    using Letter = typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;

    static constexpr unsigned block_size = Strategy_::block_size;
private:

    std::byte* smem_ptr_;

public:

    explicit constexpr SmallBlockContext(std::byte* smem_ptr) noexcept
        : smem_ptr_(smem_ptr)
    {}

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned thread_rank() noexcept {
        return threadIdx.x % block_size;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned num_threads() noexcept {
        return block_size;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned group_idx() noexcept {
        return threadIdx.x / block_size;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned group_mask() noexcept {
        return ((1u << block_size) - 1) << group_idx();
    }

    RPP_DEVICE static constexpr Index warp_idx() noexcept {
        return threadIdx.x / Strategy::warp_size;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned num_warps() noexcept {
        return Strategy::max_warp_count;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned warp_lane() noexcept {
        return threadIdx.x % Strategy::warp_size;
    }

    RPP_DEVICE static void sync_warp() noexcept {
        __syncwarp();
    }

    RPP_DEVICE static void sync() noexcept {
        return __syncwarp();
    }

    template <typename SharedMemory>
    RPP_DEVICE constexpr decltype(auto) shared_memory() const noexcept {
        if constexpr (std::is_pointer_v<SharedMemory>) {
            return reinterpret_cast<SharedMemory>(smem_ptr_);
        } else {
            return *reinterpret_cast<SharedMemory*>(smem_ptr_);
        }
    }

    template <typename T, typename Fn>
    RPP_DEVICE static T warp_reduce(T val, Fn&& fn) noexcept {
        for (unsigned i = Strategy::warp_size / 2; i > 0; i /= 2) {
            val = fn(val, __shfl_down_sync(0xFFFFFFFF, val, i));
        }
        return val;
    }

    template <typename T, typename Fn>
    RPP_DEVICE static T reduce(T val, Fn&& fn) noexcept {
        for (unsigned i = Strategy::block_size / 2; i > 0; i /= 2) {
            val = fn(val, __shfl_down_sync(0xFFFFFFFF, val, i, block_size));
        }
        return val;
    }

    template <typename Basis>
    RPP_DEVICE Degree low_range_degree(Basis const& basis) const noexcept {
        Degree result = 0;
        const auto threads = static_cast<Index>(num_threads());
        while (result <= basis.depth && basis.start_of_degree(result) < threads) {
            ++result;
        }
        return result;
    }

    
};

}



}


#endif //RPP_GPU_OPERATIONS_BLOCK_STRATEGY_HPP
