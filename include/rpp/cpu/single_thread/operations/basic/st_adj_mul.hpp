#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_ST_ADJ_MUL_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_ST_ADJ_MUL_HPP

#include <algorithm>
#include <array>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/basic/st_adj_mul.hpp>

#include <rpp/cpu/single_thread/strategy.hpp>
namespace rpp::ops {

template <typename Accum_, typename Architecture>
class STAdjMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> : public BaseOperation<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Letter = typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;

public:
    static constexpr bool is_implemented = true;

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    void operator()(Context const& ctx, TensorOut& out, TensorOp const& op, TensorArg const& arg) const noexcept {
        using Scalar = typename TensorOut::value_type;
        ignore_unused(ctx);

        std::fill(out.begin(), out.end(), Scalar{0});

        auto const& basis = out.basis();
        std::array<Letter, Strategy::Architecture::max_depth> letters{};

        for (Degree arg_degree = arg.min_degree(); arg_degree <= arg.max_degree(); ++arg_degree) {
            auto arg_level = arg.degree_view(arg_degree);

            for (Index i = 0; i < arg_level.size(); ++i) {
                const Accum arg_value{arg_level[i]};

                if (arg_degree == 0) {
                    if (op.has_degree(Degree{0}) && out.has_degree(Degree{0})) {
                        out[0] = static_cast<Scalar>(Accum{out[0]} + arg_value * Accum{op[0]});
                    }
                    continue;
                }

                basis.unpack_index_to_letters(letters, arg_degree, i);
                const auto mask_count = static_cast<Bitmask>(Bitmask{1} << arg_degree);
                for (Bitmask mask{0}; mask < mask_count; ++mask) {
                    Degree op_degree{0};
                    Index op_idx{0};
                    Degree out_degree{0};
                    Index out_idx{0};

                    basis.pack_masked_index(
                        letters,
                        arg_degree,
                        mask,
                        op_degree,
                        op_idx,
                        out_degree,
                        out_idx
                    );

                    if (op.has_degree(op_degree) && out.has_degree(out_degree)) {
                        auto out_level = out.degree_view(out_degree);
                        auto op_level = op.degree_view(op_degree);
                        const Accum value = Accum{out_level[out_idx]} + arg_value * Accum{op_level[op_idx]};
                        out_level[out_idx] = static_cast<Scalar>(value);
                    }
                }
            }
        }
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_ST_ADJ_MUL_HPP