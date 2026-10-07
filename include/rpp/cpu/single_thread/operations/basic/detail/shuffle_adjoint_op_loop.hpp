#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_DETAIL_SHUFFLE_ADJOINT_OP_LOOP_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_DETAIL_SHUFFLE_ADJOINT_OP_LOOP_HPP

#include <cstddef>
#include <functional>

#include <rpp/operations/implementation/word_shuffle_adjoint_multiply.hpp>

namespace rpp::ops::common {

// Bounds are inclusive and expressed in the getters' effective degrees.
// The caller populates the output letters; this loop writes the op letters
// immediately after them. Storage must cover every combined degree visited.
template <typename Context,
          typename TensorOp,
          typename TensorArg,
          std::size_t N,
          typename ArgGetter = detail::DefaultGetter<TensorArg>,
          typename OpGetter = detail::DefaultGetter<TensorOp>,
          typename Multiply = std::multiplies<typename Context::Accum>>
typename Context::Accum
shuffle_adjoint_op_loop(Context const& ctx,
                        typename Context::Index out_index,
                        typename Context::Degree out_degree,
                        Span<typename Context::Letter, N> letters,
                        TensorOp const& op,
                        TensorArg const& arg,
                        typename Context::Degree op_min_degree,
                        typename Context::Degree op_max_degree,
                        ArgGetter&& arg_getter = {},
                        OpGetter&& op_getter = {},
                        Multiply&& multiply = {}) noexcept {
    using Degree = typename Context::Degree;
    using Index = typename Context::Index;
    using Accum = typename Context::Accum;

    Accum acc{0};
    for (Degree op_degree = op_min_degree; op_degree <= op_max_degree;
         ++op_degree) {
        const auto degree_size = op.basis().size_of_degree(op_degree);
        auto op_letters = letters.subspan(static_cast<std::size_t>(out_degree),
                                          static_cast<std::size_t>(op_degree));
        for (Index op_index = 0; op_index < degree_size; ++op_index) {
            op.basis().unpack_index_to_letters(op_letters, op_degree, op_index);
            acc += word_shuffle_adjoint_multiply(ctx,
                                                 out_index,
                                                 out_degree,
                                                 op_index,
                                                 op_degree,
                                                 letters,
                                                 arg,
                                                 op,
                                                 arg_getter,
                                                 op_getter,
                                                 multiply);
        }
    }
    return acc;
}

} // namespace rpp::ops::common

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_DETAIL_SHUFFLE_ADJOINT_OP_LOOP_HPP
