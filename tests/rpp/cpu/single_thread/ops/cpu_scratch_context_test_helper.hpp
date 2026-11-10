#ifndef RPP_TESTS_CPU_SINGLE_THREAD_OPS_CPU_SCRATCH_CONTEXT_TEST_HELPER_HPP
#define RPP_TESTS_CPU_SINGLE_THREAD_OPS_CPU_SCRATCH_CONTEXT_TEST_HELPER_HPP

#include <cstddef>
#include <vector>

#include <rpp/cpu/single_thread/operations/basic/detail/ft_multiply.hpp>

namespace rpp::tests {

// Own the multiplication scratch storage used by direct operation tests.
template <typename Context,
          unsigned TilePower = cpu::single_thread::ft_multiply::default_tile_power>
class CpuScratchContext : public Context {
    using Storage =
        cpu::single_thread::ft_multiply::TileStorage<Context, TilePower>;
    std::vector<std::byte> bytes_;

public:
    CpuScratchContext() : Context(nullptr), bytes_(Storage::scratch_size()) {
        static_cast<Context&>(*this) = Context(bytes_.data());
        Storage::construct(*this);
    }
    CpuScratchContext(CpuScratchContext const&) = delete;
    CpuScratchContext& operator=(CpuScratchContext const&) = delete;
    ~CpuScratchContext() { Storage::destroy(*this); }

    Context const& context() const noexcept { return *this; }
};

} // namespace rpp::tests

#endif
