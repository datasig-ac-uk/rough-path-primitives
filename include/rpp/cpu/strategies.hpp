#ifndef RPP_CPU_STRATEGIES_HPP
#define RPP_CPU_STRATEGIES_HPP

#include <cstddef>

#include <rpp/config.h>

namespace rpp::cpu::strategies {



namespace detail {

template <typename Strategy_>
class ThreadContext {
public:
    using Strategy = Strategy_;
    using Architecture = typename Strategy::Architecture;

    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Letter=  typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;

private:
    std::byte* scratch_ptr_;

public:

    explicit constexpr ThreadContext(std::byte* scratch_ptr=nullptr) noexcept
        : scratch_ptr_(scratch_ptr)
    {}

    RPP_HOST_DEVICE
    static void sync() noexcept {}


    template <typename Element>
    RPP_HOST_DEVICE
    RPP_NODISCARD Element* scratch_space() const noexcept {
        return reinterpret_cast<Element*>(scratch_ptr_);
    }


};

} // namespace detail

template <typename Accum_, typename Architecture_>
struct SingleThreadStrategy {
    using Accum = Accum_;
    using Architecture = Architecture_;
    using Degree = typename Architecture::Degree;
    using Index = typename Architecture::Index;
    using Letter = typename Architecture::Letter;
    using Bitmask = typename Architecture::Bitmask;

    using Context = detail::ThreadContext<SingleThreadStrategy<Accum_, Architecture_>>;

    static constexpr Context make_context(std::byte* scratch_bytes) noexcept {
         return Context(scratch_bytes);
    }
};

}

#endif //RPP_CPU_STRATEGIES_HPP
