#ifndef RPP_CPU_SINGLE_THREAD_STRATEGY_HPP
#define RPP_CPU_SINGLE_THREAD_STRATEGY_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <tuple>
#include <vector>
#include <string>

#include <rpp/config.h>

#include <rpp/support/error.hpp>
#include <rpp/support/data_mapping.hpp>
#include <rpp/operations/base_operation.hpp>

#include <rpp/cpu/device.hpp>

namespace rpp::cpu::strategies {
namespace detail {
template<typename Strategy_>
class ThreadContext {
public:
    using Strategy = Strategy_;
    using Architecture = typename Strategy::Architecture;

    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Letter = typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;
    using Accum = typename Strategy::Accum;

    static constexpr bool is_nothrow = std::is_integral_v<Accum> || std::is_floating_point_v<Accum>;

private:
    std::byte *scratch_ptr_;

public:
    explicit constexpr ThreadContext(std::byte *scratch_ptr = nullptr) noexcept
        : scratch_ptr_(scratch_ptr) {
    }

    RPP_HOST_DEVICE
    static void sync() noexcept {
    }


    template<typename Element>
    RPP_HOST_DEVICE
    RPP_NODISCARD decltype(auto) scratch_space() const noexcept {
        if constexpr (std::is_pointer_v<Element>) {
            return reinterpret_cast<Element>(scratch_ptr_);
        } else {
            return *reinterpret_cast<Element *>(scratch_ptr_);
        }
    }
};

struct LaunchConfig {
};
} // namespace detail

template<typename Accum_, typename Architecture_>
struct SingleThreadStrategy {
    using Accum = Accum_;
    using Architecture = Architecture_;
    using Degree = typename Architecture::Degree;
    using Index = typename Architecture::Index;
    using Letter = typename Architecture::Letter;
    using Bitmask = typename Architecture::Bitmask;
    using LaunchConfig = detail::LaunchConfig;

    using Context = detail::ThreadContext<SingleThreadStrategy<Accum_, Architecture_> >;

    static constexpr Context make_context(std::byte *scratch_bytes) noexcept {
        return Context(scratch_bytes);
    }


    template<typename Op, typename Batches, typename Bases, typename... Extras>
    RPP_HOST Error<std::string> launch(
        LaunchConfig launch_config,
        Batches batches,
        Bases bases,
        Index batch_size,
        Extras &&... extras
    ) const noexcept;
};


template<typename Accum_, typename Architecture_>
template<typename Op, typename Batches, typename Bases, typename... Extras>
Error<std::string> SingleThreadStrategy<Accum_, Architecture_>::launch(
    LaunchConfig launch_config RPP_MAYBE_UNUSED,
    Batches batches,
    Bases bases, Index batch_size, Extras &&... extras) const noexcept {
    DataMapper<Architecture> mapper;
    auto extras_mapped = map_data_args<std::tuple>(mapper, std::forward<Extras>(extras)...);
    if (!extras_mapped) {
        return std::move(extras_mapped).error();
    }

    const auto scratch_size = Op::scratch_space_size(*this, bases);
    std::vector<std::byte> scratch_bytes(scratch_size);
    const auto ctx = make_context(scratch_bytes.data());

    Op::init_scratch_space(ctx, bases);


    auto result = catch_exceptions([&, op=Op{}] {
        for (Index idx = 0; idx < batch_size; ++idx) {
            ops::invoke(op, ctx, [&](auto &batch) { return batch.view(bases, idx); }, batches,
                        extras_mapped.value());
        }
    });

    Op::destroy_scratch_space(ctx, bases);
    return result;
}
} // namespace rpp::cpu::strategies

#endif // RPP_CPU_SINGLE_THREAD_STRATEGY_HPP