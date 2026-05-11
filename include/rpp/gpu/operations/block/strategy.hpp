#ifndef RPP_GPU_OPERATIONS_BLOCK_STRATEGY_HPP
#define RPP_GPU_OPERATIONS_BLOCK_STRATEGY_HPP

#include <cstddef>
#include <limits>

#include <cuda_runtime.h>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/gpu/architecture.hpp>

namespace rpp::gpu::strategies {

inline constexpr unsigned dynamic_block_size = std::numeric_limits<unsigned>::max();

template <typename Strategy>
inline constexpr bool is_block_strategy_v = false;

namespace detail {

template <typename Strategy_>
class BlockContext {
public:
    using Strategy = Strategy_;
    using BlockReduceArray = typename Strategy::BlockReduceArray;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Accum = typename Strategy::Accum;
    using Letter = typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;

    static constexpr unsigned static_block_size = Strategy_::static_block_size;

    static constexpr bool static_small_block = static_block_size < Strategy::warp_size;
private:

    std::byte* smem_ptr_;

public:

    explicit constexpr BlockContext(std::byte* smem_ptr) noexcept
        : smem_ptr_(smem_ptr)
    {}

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned thread_rank() noexcept {
        if constexpr (static_small_block) {
            return threadIdx.x % static_block_size;
        } else {
            return threadIdx.x;
        }
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned num_threads() noexcept {
        if constexpr (static_small_block) {
            return static_block_size;
        } else {
            return static_block_size;
        }
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned group_idx() noexcept {
        if constexpr (static_small_block) {
            return threadIdx.x / static_block_size;
        } else {
            return 0;
        }
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned group_mask() noexcept {
        if constexpr (static_small_block) {
            return ((1u << static_block_size) - 1) << group_idx();
        } else {
            return 0XFFFFFFFFU;
        }
    }

    RPP_DEVICE static constexpr Index warp_idx() noexcept {
        return threadIdx.x / Strategy::warp_size;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned num_warps() noexcept {
        if constexpr (static_small_block) {
            return Strategy::warp_count;
        } else {
            return (blockDim.x + Strategy::warp_size - 1) / Strategy::warp_size;
        }
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned warp_lane() noexcept {
        return threadIdx.x % Strategy::warp_size;
    }

    RPP_DEVICE static void sync_warp() noexcept {
        __syncwarp();
    }

    RPP_DEVICE static void sync() noexcept {
        if constexpr (static_small_block) {
            __syncwarp();
        } else {
            __syncthreads();
        }
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
    RPP_DEVICE T reduce(T val, Fn&& fn) const noexcept {
        if constexpr (static_block_size == dynamic_block_size || static_block_size >= Strategy::warp_size) {
            auto& smem = shared_memory<BlockReduceArray>();
            val = warp_reduce(val, fn);

            const auto widx = warp_idx();
            const auto wlane = warp_lane();
            if (wlane == 0) {
                smem[widx] = val;
            }
            sync();

            Accum block_sum { 0 };
            if (widx == 0) {
                if (wlane < num_warps()) {
                    block_sum = smem[wlane];
                }
                block_sum = warp_reduce(block_sum, fn);
            }

            return block_sum;
        } else {
            for (unsigned i = Strategy::block_size / 2; i > 0; i /= 2) {
                val = fn(val, __shfl_down_sync(0xFFFFFFFF, val, i, num_threads()));
            }
            return val;
        }
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


template <unsigned BlockSize>
struct BlockSizeHolder {
    static constexpr unsigned static_block_size = BlockSize;
    static constexpr unsigned block_size = BlockSize;

    constexpr BlockSizeHolder(unsigned size) noexcept
    {}
};

template <>
struct BlockSizeHolder<dynamic_block_size> {
    static constexpr unsigned static_block_size = dynamic_block_size;
    unsigned block_size;

    constexpr BlockSizeHolder(unsigned size) noexcept
        : block_size(size)
    {}
};



} // namespace detail



template <typename Accum_, unsigned BlockSize=dynamic_block_size, unsigned MaxBlockSize=256, typename Architecture_=arch::DefaultArchitecture>
struct BlockStrategy : public detail::BlockSizeHolder<BlockSize>{
    using Accum = Accum_;
    using Architecture = Architecture_;
    using Size = typename Architecture::Size;
    using Index = typename Architecture::Index;
    using Letter = typename Architecture::Letter;
    using Bitmask = typename Architecture::Bitmask;
    using Degree = typename Architecture::Degree;

    using Context = detail::BlockContext<BlockStrategy>;

    using detail::BlockSizeHolder<BlockSize>::block_size;
    using detail::BlockSizeHolder<BlockSize>::static_block_size;
    using detail::BlockSizeHolder<BlockSize>::BlockSizeHolder;

    static constexpr unsigned max_block_size = MaxBlockSize;
    static constexpr unsigned warp_size = Architecture::warp_size;
    static constexpr unsigned warp_count = (block_size + warp_size - 1) / warp_size;
    static constexpr unsigned max_warp_count = (max_block_size + warp_size - 1) / warp_size;
    static constexpr unsigned sector_alignment = Architecture::sector_alignment;
    static constexpr bool static_small_block = static_block_size < warp_size;

    static_assert(
        is_pow_2(block_size)
        && block_size <= max_block_size
        && (block_size % warp_size) == 0,
        "invalid block configuration: block size must be a power of 2, within max block size, and divisible by warp size"
        );

    using BlockReduceArray = Accum[max_warp_count];

    RPP_HOST_DEVICE RPP_NODISCARD
    static constexpr Index objects_per_block() noexcept {
        if constexpr (BlockSize < MaxBlockSize) {
            return MaxBlockSize / block_size;
        } else {
            return 1;
        }
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    static constexpr Index threads_per_object() noexcept {
        return block_size;
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    static constexpr Index object_index(unsigned block_index, unsigned thread_index) noexcept {
        if constexpr (BlockSize < MaxBlockSize) {
            return static_cast<Index>(block_index) * objects_per_block() + thread_index / block_size;
        } else {
            return static_cast<Index>(block_index);
        }
    }

    RPP_DEVICE static constexpr Context make_context(std::byte* smem_bytes) noexcept {
        if constexpr (BlockSize < MaxBlockSize) {
            return Context { smem_bytes + threadIdx.x / block_size };
        } else {
            return Context { smem_bytes };
        }
    }

};


template <typename Accum, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
inline constexpr bool is_block_strategy_v<BlockStrategy<Accum, BlockSize, MaxBlockSize, Architecture>> = true;


}


#endif //RPP_GPU_OPERATIONS_BLOCK_STRATEGY_HPP
