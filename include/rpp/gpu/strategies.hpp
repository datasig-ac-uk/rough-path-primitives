#ifndef RPP_GPU_STRATEGIES_HPP
#define RPP_GPU_STRATEGIES_HPP

#include <cstdint>
#include <cstddef>
#include <type_traits>


#include <cuda_runtime.h>

#include <rpp/gpu/architecture.hpp>


namespace rpp::gpu::strategies {

namespace detail {

template <typename Strategy_>
class BlockContext {
public:
    using Strategy = Strategy_;
    using Accum = typename Strategy::Accum;
    using BlockReduceArray = typename Strategy::BlockReduceArray;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Letter = typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;

private:
    std::byte* smem_ptr_;

public:
    explicit constexpr BlockContext(std::byte* smem_ptr) noexcept
        : smem_ptr_(smem_ptr)
    {}

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned thread_rank() noexcept {
        return threadIdx.x;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned warp_lane() noexcept {
        return threadIdx.x % Strategy::warp_size;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned warp_idx() noexcept {
        return threadIdx.x / Strategy::warp_size;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned num_threads() noexcept {
        return blockDim.x;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned num_warps() noexcept {
        return blockDim.x / Strategy::warp_size;
    }

    RPP_DEVICE static void sync_warp() noexcept {
        return __syncwarp();
    }

    RPP_DEVICE static void sync() noexcept {
        __syncthreads();
    }

    template <typename SharedMemory>
    RPP_DEVICE constexpr decltype(auto) shared_memory() const noexcept {
        if constexpr (std::is_pointer_v<SharedMemory>) {
            return reinterpret_cast<SharedMemory>(smem_ptr_);
        } else {
            return *reinterpret_cast<SharedMemory*>(smem_ptr_);
        }
    }

    template <typename Fn>
    RPP_DEVICE static Accum warp_reduce(Accum val, Fn&& fn) noexcept {
        for (unsigned i = Strategy_::warp_size / 2; i > 0; i /= 2) {
            val = fn(val, __shfl_down_sync(0xffffffffu, val, i));
        }
        return val;
    }

    template <typename Fn>
    RPP_DEVICE Accum reduce(Accum val, Fn&& fn) const noexcept {
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

struct GroupConfig {
    unsigned thread_rank_: 4; // 4 (0 <= x <= 31)
    unsigned num_threads_: 5; // 9 (0 <= x <= 32, power of 2)
    unsigned num_threads_pow_: 3; // 12 (0 <= x <= 5)
    unsigned group_lane_: 7; // 19 (0 <= x < 256)
};

template <typename Strategy_>
class SmallCoopGroupContext {
    using Strategy = Strategy_;

    uint8_t* smem_ptr;
    GroupConfig config;
    unsigned group_mask;

public:
    RPP_DEVICE explicit constexpr SmallCoopGroupContext(unsigned group_size_pow)
        : smem_ptr(nullptr),
          config{
              static_cast<unsigned>(threadIdx.x & ((1u << group_size_pow) - 1u)),
              static_cast<unsigned>(1u << group_size_pow),
              group_size_pow,
              static_cast<unsigned>(threadIdx.x)
          },
          group_mask(0xffffffffu)
    {}

    RPP_DEVICE constexpr unsigned thread_rank() const noexcept {
        return config.thread_rank_;
    }

    RPP_DEVICE constexpr unsigned num_threads() const noexcept {
        return config.num_threads_;
    }

    RPP_DEVICE void sync() const noexcept {
        __syncwarp(group_mask);
    }

    RPP_DEVICE static unsigned warp_lane() noexcept {
        return threadIdx.x % Strategy::warp_size;
    }

    RPP_DEVICE static unsigned warp_idx() noexcept {
        return threadIdx.x / Strategy::warp_size;
    }

    RPP_DEVICE static unsigned num_warps() noexcept {
        return blockDim.x / Strategy::warp_size;
    }


    template <typename SharedMemory>
    RPP_DEVICE decltype(auto) shared_memory() const noexcept {
        if constexpr (std::is_pointer_v<SharedMemory>) {
            return reinterpret_cast<SharedMemory>(smem_ptr);
        } else {
            return *reinterpret_cast<SharedMemory*>(smem_ptr);
        }
    }

};


} // namespace detail

template <typename Accum_, unsigned MaxBlockSize = 256, typename Architecture_=arch::DefaultArchitecture>
struct BlockStrategy {
    using Accum = Accum_;
    using Architecture = Architecture_;
    using Size = typename Architecture::Size;
    using Index = typename Architecture::Index;
    using Degree = typename Architecture::Degree;
    using Letter = typename Architecture::Letter;
    using Bitmask = typename Architecture::Bitmask;

    using Context = detail::BlockContext<BlockStrategy>;

    static constexpr unsigned max_block_size = MaxBlockSize;
    static constexpr unsigned warp_size = Architecture::warp_size;
    static constexpr unsigned max_warp_count = (max_block_size + warp_size - 1) / warp_size;
    static constexpr unsigned sector_alignment = Architecture::sector_alignment;

    using BlockReduceArray = Accum[max_warp_count];

    unsigned block_size;

    RPP_HOST_DEVICE
    static constexpr Index objects_per_block() noexcept {
        return 1;
    }

    RPP_HOST_DEVICE
    constexpr Index threads_per_object() noexcept {
        return block_size;
    }

    RPP_HOST_DEVICE
    static constexpr Index object_index(unsigned block_index, unsigned thread_index RPP_MAYBE_UNUSED) noexcept {
        return static_cast<Index>(block_index);
    }

    RPP_DEVICE static constexpr Context make_context(std::byte* smem_bytes) noexcept {
        return Context { smem_bytes };
    }


};






} // namespace rpp::gpu::strategies

#endif // RPP_GPU_STRATEGIES_HPP
