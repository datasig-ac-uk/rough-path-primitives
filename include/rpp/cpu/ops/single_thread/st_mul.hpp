#ifndef RPP_CPU_OPS_SINGLE_THREAD_ST_MUL_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_ST_MUL_HPP

#include <algorithm>
#include <array>
#include <cstddef>

#include <rpp/cpu/strategies.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class STMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Letter = typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;

    template <typename Basis, typename TensorLhs, typename TensorRhs>
    static Accum shuffle_product_coefficient(
        Basis const& basis,
        TensorLhs const& lhs,
        TensorRhs const& rhs,
        Degree degree,
        Index index
    ) noexcept {
        std::array<Letter, Strategy::Architecture::max_depth> letters{};
        basis.unpack_index_to_letters(letters, degree, index);

        Accum acc{0};
        const auto mask_count = static_cast<Bitmask>(Bitmask{1} << degree);
        for (Bitmask mask{0}; mask < mask_count; ++mask) {
            Degree lhs_degree{0};
            Index lhs_idx{0};
            Degree rhs_degree{0};
            Index rhs_idx{0};
            basis.pack_masked_index(
                letters,
                static_cast<Degree>(degree - 1),
                mask,
                lhs_degree,
                lhs_idx,
                rhs_degree,
                rhs_idx
            );

            if (lhs.has_degree(lhs_degree) && rhs.has_degree(rhs_degree)) {
                acc += Accum{lhs.degree_view(lhs_degree)[lhs_idx]} * Accum{rhs.degree_view(rhs_degree)[rhs_idx]};
            }
        }
        return acc;
    }

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr std::size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorLhs, typename TensorRhs>
    void operator()(Context const& ctx, TensorOut& out, TensorLhs const& lhs, TensorRhs const& rhs, Accum beta = Accum{1}) const noexcept {
        using Scalar = typename TensorOut::value_type;
        ignore_unused(ctx);

        auto const& basis = out.basis();

        if (out.min_degree() == 0) {
            Accum value{0};
            if (lhs.has_degree(Degree{0}) && rhs.has_degree(Degree{0})) {
                value = beta * Accum{lhs[0]} * Accum{rhs[0]};
            }
            out[0] = static_cast<Scalar>(value);
        }

        const auto min_degree = std::max(Degree{1}, out.min_degree());
        for (Degree degree = min_degree; degree <= out.max_degree(); ++degree) {
            auto out_level = out.degree_view(degree);
            for (Index i = 0; i < out_level.size(); ++i) {
                out_level[i] = static_cast<Scalar>(beta * shuffle_product_coefficient(basis, lhs, rhs, degree, i));
            }
        }
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_OPS_SINGLE_THREAD_ST_MUL_HPP
