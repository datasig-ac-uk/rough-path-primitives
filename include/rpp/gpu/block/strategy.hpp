#ifndef RPP_GPU_BLOCK_STRATEGY_HPP
#define RPP_GPU_BLOCK_STRATEGY_HPP

#include <cstddef>
#include <limits>
#include <type_traits>

#include <cuda_runtime.h>

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/support/error.hpp>
#include <rpp/support/data_mapping.hpp>

#include <rpp/gpu/architecture.hpp>
#include <rpp/gpu/device.hpp>
#include <rpp/gpu/block/kernel.hpp>

namespace rpp::gpu::strategies {
inline constexpr unsigned dynamic_block_size = std::numeric_limits<unsigned>::max();

template<typename Strategy>
inline constexpr bool is_block_strategy_v = false;

namespace detail {
template<typename Strategy_>
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

    static constexpr bool is_nothrow = std::is_integral_v<Accum> || std::is_floating_point_v<Accum>;
private:
    std::byte *smem_ptr_;

public:
    explicit constexpr BlockContext(std::byte *smem_ptr) noexcept
        : smem_ptr_(smem_ptr) {
    }

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
            return blockDim.x;
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
        if constexpr (static_small_block) {
            return 0;
        } else  {
            return threadIdx.x / Strategy::warp_size;
        }
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

    template<typename SharedMemory>
    RPP_DEVICE constexpr decltype(auto) shared_memory() const noexcept {
        if constexpr (std::is_pointer_v<SharedMemory>) {
            return reinterpret_cast<SharedMemory>(smem_ptr_);
        } else {
            return *reinterpret_cast<SharedMemory *>(smem_ptr_);
        }
    }

    template<typename T, typename Fn>
    RPP_DEVICE static T warp_reduce(T val, Fn &&fn) noexcept {
        for (unsigned i = Strategy::warp_size / 2; i > 0; i /= 2) {
            val = fn(val, __shfl_down_sync(0xFFFFFFFF, val, i));
        }
        return val;
    }

    template<typename T, typename Fn>
    RPP_DEVICE T reduce(T val, Fn &&fn) const noexcept {
        if constexpr (static_block_size == dynamic_block_size || static_block_size >= Strategy::warp_size) {
            auto &smem = shared_memory<BlockReduceArray>();
            val = warp_reduce(val, fn);

            const auto widx = warp_idx();
            const auto wlane = warp_lane();
            if (wlane == 0) {
                smem[widx] = val;
            }
            sync();

            Accum block_sum{0};
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

    template<typename Basis>
    RPP_DEVICE Degree low_range_degree(Basis const &basis) const noexcept {
        Degree result = 0;
        const auto threads = static_cast<Index>(num_threads());
        while (result <= basis.depth && basis.start_of_degree(result) < threads) {
            ++result;
        }
        return result;
    }
};


template<unsigned BlockSize>
struct BlockSizeHolder {
    static constexpr unsigned static_block_size = BlockSize;
    static constexpr unsigned block_size = BlockSize;

    constexpr BlockSizeHolder(unsigned size) noexcept {
    }
};

template<>
struct BlockSizeHolder<dynamic_block_size> {
    static constexpr unsigned static_block_size = dynamic_block_size;
    unsigned block_size;

    constexpr BlockSizeHolder(unsigned size) noexcept
        : block_size(size) {
    }
};
} // namespace detail


template<typename Accum_, unsigned BlockSize = dynamic_block_size, unsigned MaxBlockSize = 256, typename Architecture_=
    arch::DefaultArchitecture>
struct BlockStrategy : public detail::BlockSizeHolder<BlockSize> {
    using Accum = Accum_;
    using Architecture = Architecture_;
    using Size = typename Architecture::Size;
    using Index = typename Architecture::Index;
    using Letter = typename Architecture::Letter;
    using Bitmask = typename Architecture::Bitmask;
    using Degree = typename Architecture::Degree;

    using Context = detail::BlockContext<BlockStrategy>;
    using LaunchConfig = DeviceLaunchConfig;

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

    RPP_HOST RPP_NODISCARD
    constexpr unsigned launch_block_size() const noexcept {
        if constexpr (BlockSize < warp_size) {
            return MaxBlockSize;
        } else {
            return block_size;
        }
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    static constexpr Index objects_per_block() noexcept {
        if constexpr (BlockSize < warp_size) {
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
        if constexpr (BlockSize < warp_size) {
            return static_cast<Index>(block_index) * objects_per_block() + thread_index / block_size;
        } else {
            return static_cast<Index>(block_index);
        }
    }

    RPP_DEVICE static constexpr Context make_context(std::byte *smem_bytes) noexcept {
        if constexpr (BlockSize < warp_size) {
            return Context{smem_bytes + threadIdx.x / block_size};
        } else {
            return Context{smem_bytes};
        }
    }


    template<typename Op, typename Batches, typename Bases, typename... Extras>
    RPP_HOST Error<char const *> launch(
        LaunchConfig &&config,
        Batches &&batches,
        Bases &&bases,
        Index batch_size,
        Extras... extras
    ) const noexcept;
};


template<typename Accum, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
inline constexpr bool is_block_strategy_v<BlockStrategy<Accum, BlockSize, MaxBlockSize, Architecture> > = true;


template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture_>
template<typename Op, typename Batches, typename Bases, typename... Extras>
Error<char const *> BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture_>::launch(LaunchConfig &&launch_config,
    Batches &&batches, Bases &&bases, Index batch_size, Extras ... extras) const noexcept {

    DataMapper<Architecture> mapper(launch_config.stream);
    auto mapped_extras = map_data_args<std::tuple>(mapper, extras...);
    if (!mapped_extras) {
        return std::move(mapped_extras).error();
    }

    using BatchesT = std::remove_cv_t<std::remove_reference_t<Batches>>;
    using BasesT = std::remove_cv_t<std::remove_reference_t<Bases>>;
    using ExtrasT = typename std::remove_cv_t<std::remove_reference_t<decltype(mapped_extras)>>::value_type;

    constexpr auto kernel = block::kernel<Op, BatchesT, BasesT, ExtrasT>;

    cudaLaunchConfig_t config{};
    config.blockDim = dim3{launch_block_size()};
    config.stream = launch_config.stream;

    const auto num_objects = objects_per_block();
    config.gridDim = dim3{static_cast<unsigned>((batch_size + num_objects - 1) / num_objects)};

    config.dynamicSmemBytes = Op::scratch_space_size(*this, bases);

    config.attrs = launch_config.launch_attributes.data();
    config.numAttrs = static_cast<unsigned int>(launch_config.launch_attributes.size());

    auto err = cudaLaunchKernelEx(&config, kernel, batches, bases, batch_size, mapped_extras.value());
    return map_cuda_error(err);
}
} // namespace rpp::gpu::strategies


#endif // RPP_GPU_BLOCK_STRATEGY_HPP