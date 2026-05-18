#ifndef RPP_GPU_ARCHITECTURE_HPP
#define RPP_GPU_ARCHITECTURE_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <rpp/architecture.hpp>

namespace rpp {

namespace gpu {
enum class MemoryLocation {
    GlobalMemory,
    SharedMemory,
};

template <MemoryLocation = MemoryLocation::GlobalMemory>
struct GPUMemoryLocation {
    static constexpr auto memory_location = MemoryLocation::GlobalMemory;
};
} // namespace gpu

namespace traits {

template <typename T>
inline constexpr bool is_gpu_location_v = false;

template <gpu::MemoryLocation Loc>
inline constexpr bool is_gpu_location_v<gpu::GPUMemoryLocation<Loc>> = true;

} // namespace traits

namespace gpu::arch {

namespace detail {

using BitmaskBase = unsigned;

template <unsigned MaxDepth>
inline constexpr size_t kBitmaskArraySize =
    (MaxDepth + 8 * sizeof(BitmaskBase) - 1) / (8 * sizeof(BitmaskBase));


} // namespace detail

template <typename Size_, typename Letter_ = uint8_t, unsigned MaxDepth = 30>
struct GPUArchitecture : public ::rpp::arch::Architecture<Size_> {
    static constexpr unsigned warp_size = 32;

    using Letter = Letter_;
    using Bitmask = unsigned;

    static constexpr unsigned max_width = std::numeric_limits<Letter>::max();
    static constexpr unsigned max_depth = MaxDepth;
    static constexpr unsigned sector_alignment = 128;


    template <typename T>
    using GMemPtr = ::rpp::TaggedPtr<
        T,
        tags::ArchTag<GPUArchitecture>,
        tags::LocationTag<GPUMemoryLocation<MemoryLocation::GlobalMemory>>>;
    template <typename T>
    using SMemPtr = ::rpp::TaggedPtr<
        T,
        tags::ArchTag<GPUArchitecture>,
        tags::LocationTag<GPUMemoryLocation<MemoryLocation::SharedMemory>>>;


    template <typename T>
    using Ptr = GMemPtr<T>;
};

using NativeArchitecture = GPUArchitecture<std::size_t>;
using Architecture32 = GPUArchitecture<std::uint32_t>;
using Architecture64 = GPUArchitecture<std::uint64_t>;


using DefaultArchitecture = Architecture32;
} // namespace gpu::arch
} // namespace rpp

#endif // RPP_GPU_ARCHITECTURE_HPP
