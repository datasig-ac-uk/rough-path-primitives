#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_INPLACE_FMA_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_INPLACE_FMA_HPP

// IWYU pragma: always_keep

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/basic/ft_inplace_fma.hpp>

#include <rpp/cpu/single_thread/operations/basic/detail/ft_multiply.hpp>
#include <rpp/cpu/single_thread/strategy.hpp>

namespace rpp::ops {

// TODO: The inplace a <- a*b + c and a <- b*a + c variants need a different
// strategy

// template<typename Accum_, typename Architecture, InplaceFMAType FMAType>
// class FTInplaceFma<cpu::strategies::SingleThreadStrategy<Accum_,
// Architecture>, FMAType> {
//     using Strategy = cpu::strategies::SingleThreadStrategy<Accum_,
//     Architecture>; using Context = typename Strategy::Context; using Accum =
//     typename Strategy::Accum;
//
//     using Index = typename Strategy::Index;
//     using Degree = typename Strategy::Degree;
//
// public:
//     template<typename Basis>
//     static constexpr std::size_t scratch_space_size(Strategy const &strategy,
//     Basis const &basis) noexcept {
//         ignore_unused(strategy, basis);
//         return 0;
//     }
//
//     template<typename TensorA, typename TensorB, typename TensorC>
//     void operator()(
//         Context const &ctx,
//         TensorA &a,
//         TensorB const &b,
//         TensorC const &c,
//         Accum alpha = Accum{1},
//         Accum beta = Accum{1}
//     ) const noexcept {
//         RPP_UNREACHABLE();
//     }
//
// };


template <typename Accum_, typename Architecture>
class FTInplaceFma<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>,
                   FTInplaceFMAType::AEqualsBCPlusA>
    : public BaseOperation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;

public:
    static constexpr bool is_implemented = true;

    template <typename TensorA, typename TensorB, typename TensorC>
    void operator()(Context const& ctx,
                    TensorA& a,
                    TensorB const& b,
                    TensorC const& c,
                    Accum alpha = Accum{1},
                    Accum beta = Accum{1}) const noexcept {
        namespace ftm = cpu::single_thread::ft_multiply;

        ftm::multiply(
            ctx,
            ftm::detail::select_tile_config<4>(ctx, a.basis()),
            a,
            b,
            c,
            ftm::InplaceFmaTileWriter<Context,
                                      FTInplaceFMAType::AEqualsBCPlusA>{alpha,
                                                                        beta});
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_INPLACE_FMA_HPP
