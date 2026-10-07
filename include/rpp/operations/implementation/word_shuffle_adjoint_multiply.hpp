//
// Created by sam on 06/10/2026.
//

#ifndef RPP_OPERATIONS_IMPLEMENTATION_WORD_SHUFFLE_ADJOINT_MULTIPLY_HPP
#define RPP_OPERATIONS_IMPLEMENTATION_WORD_SHUFFLE_ADJOINT_MULTIPLY_HPP

#include <cstddef>
#include <functional>
#include <limits>


#include <rpp/config.h>

#include <rpp/support/span.hpp>

#include <rpp/operations/implementation/word_shuffle_product.hpp>

namespace rpp::ops::common {


template <typename Context,
          typename TensorArg,
          typename TensorOp,
          size_t N,
          typename ArgGetter = detail::DefaultGetter<TensorArg>,
          typename OpGetter = detail::DefaultGetter<TensorOp>,
          typename Multiply = std::multiplies<typename Context::Accum>>
RPP_HOST_DEVICE RPP_FORCEINLINE typename Context::Accum
word_shuffle_adjoint_multiply(Context const& ctx RPP_MAYBE_UNUSED,
                      typename Context::Index out_index,
                      typename Context::Degree out_degree,
                      typename Context::Index op_index,
                      typename Context::Degree op_degree,
                      Span<typename Context::Letter, N> letters,
                      TensorArg const& arg,
                      TensorOp const& op,
                      ArgGetter&& arg_getter = {},
                      OpGetter&& op_getter = {},
                      Multiply&& multiply = {}) noexcept {
    using Index = typename Context::Index;
    using Degree = typename Context::Degree;
    using Letter = typename Context::Letter;
    using Accum = typename Context::Accum;
    using Bitmask = typename Context::Bitmask;

    Accum acc{0};
    if (op_degree == 0) {
        return multiply(op_getter(op, 0, 0),
                        arg_getter(arg, out_degree, out_index));
    }

    const auto& basis = arg.basis();
    // Letter letters[Context::Strategy::Architecture::max_depth];
    // Span<Letter> out_letters{letters, static_cast<std::size_t>(out_degree)};
    // basis.unpack_index_to_letters(out_letters, out_degree, out_index);
    // Span<Letter> op_letters{letters + out_degree,
    //                         static_cast<std::size_t>(op_degree)};
    // basis.unpack_index_to_letters(op_letters, op_degree, op_index);

    const auto arg_degree = out_degree + op_degree;

    auto arg_index = [&](Bitmask mask) -> Index {
        const auto width = basis.width;
        Degree out_idx = out_degree;
        Degree op_idx = op_degree;

        Index idx = 0;
        for (Degree i = arg_degree; i > 0;) {
            --i;
            const auto m = (mask >> i) & Bitmask{1};

            auto letter =
                m == 0 ? letters[--out_idx] : letters[out_degree + (--op_idx)];

            idx = width * idx + static_cast<Index>(letter); // NOLINT(*-math-missing-parentheses)
        }

        return idx;
    };


    constexpr auto digits = std::numeric_limits<Bitmask>::digits;

    Bitmask mask = std::numeric_limits<Bitmask>::max() >> (digits - op_degree);
    const auto last = mask << out_degree;

    // Use the Gosper successsor algorith to walk through all the
    // bitmasks of weight op_degree starting from 0..01..1. For each of these,
    // resolve the arg_index and multiply the two corresponding coefficients.
    for (;;) {
        acc += multiply(op_getter(op, op_degree, op_index),
                        arg_getter(arg, arg_degree, arg_index(mask)));

        if (mask == last) {
            break;
        }

        const auto lowest = mask & -mask;
        const auto ripple = mask + lowest;

        mask = ripple | (((ripple ^ mask) >> 2) / lowest);
    }


    return acc;
}


} // namespace rpp::ops::common


#endif // RPP_OPERATIONS_IMPLEMENTATION_WORD_SHUFFLE_ADJOINT_MULTIPLY_HPP
