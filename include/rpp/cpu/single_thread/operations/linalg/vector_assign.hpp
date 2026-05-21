#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_LINALG_VECTOR_ASSIGN_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_LINALG_VECTOR_ASSIGN_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/linalg/vector_assign.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/cpu/single_thread/strategy.hpp>
namespace rpp::ops {

template <typename Accum_, typename Architecture>
class VectorAssign<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>>
    : public BaseOperation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;

public:
    static constexpr bool is_implemented = true;

    template <typename VectorOut, typename VectorArg>
    void operator()(Context const& ctx,
                    VectorOut& out,
                    VectorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        const auto min_deg = std::max(out.min_degree(), arg.min_degree());
        const auto max_deg = std::min(out.max_degree(), arg.max_degree());

        const auto begin_index = basis.start_of_degree(min_deg);
        const auto end_index = basis.end_of_degree(max_deg);


        auto out_begin = out.data() + begin_index;
        auto arg_begin = arg.data() + begin_index;
        auto arg_end = arg.data() + end_index;


        std::copy(arg_begin, arg_end, out_begin);
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_LINALG_VECTOR_ASSIGN_HPP