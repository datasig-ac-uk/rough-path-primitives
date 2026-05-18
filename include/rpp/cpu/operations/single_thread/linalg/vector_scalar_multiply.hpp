#ifndef RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_VECTOR_SCALAR_MULTIPLY_HPP
#define RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_VECTOR_SCALAR_MULTIPLY_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/linalg/vector_scalar_multiply.hpp>

#include <rpp/cpu/operations/single_thread/strategy.hpp>


namespace rpp::ops {

template <typename Accum_, typename Architecture_>
class VectorScalarMultiply<cpu::strategies::SingleThreadStrategy<Accum_, Architecture_> > : public BaseOperation<cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

public:
    static constexpr bool is_implemented = true;

    using Context = typename Strategy::Context;

    template<typename Vector>
    RPP_HOST_DEVICE
    void operator()(Context const &ctx, Vector &vec, Accum scalar) const noexcept {
        ignore_unused(ctx);

        auto it = vec.begin();
        const auto end = vec.end();
        for (; it!=end; ++it) {
            Accum val { *it };
            val *= scalar;
            *it = val;
        }
    }
};


} // namespace rpp::ops

#endif //RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_VECTOR_SCALAR_MULTIPLY_HPP
