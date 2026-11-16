#ifndef RPP_TESTS_CPU_LEGACY_FT_MULTIPLY_TEST_HELPER_HPP
#define RPP_TESTS_CPU_LEGACY_FT_MULTIPLY_TEST_HELPER_HPP

#include <algorithm>

namespace rpp::tests::legacy {

// Frozen CPU algorithms from main at 67dac5296b1e5cf973c2f784a5b55d348eba15c1.
// Only the signatures and local type aliases are adapted for test use. Keep
// these independent of the tiled kernels so comparisons detect regressions.
template <typename TensorOut,
          typename TensorA,
          typename TensorB,
          typename TensorC,
          typename Accum>
void fma(TensorOut& out,
         TensorA const& a,
         TensorB const& b,
         TensorC const& c,
         Accum alpha,
         Accum beta) {
    using Degree = typename TensorOut::Basis::Degree;
    using Index = typename TensorOut::Basis::Index;

    using Scalar = typename TensorOut::value_type;
    auto out_min_degree = std::max(Degree{1}, out.min_degree());

    for (Degree out_degree = out.max_degree(); out_degree >= out_min_degree;
         --out_degree) {
        auto const lhs_deg_max =
            std::min(b.max_degree(), out_degree - c.min_degree());
        auto const lhs_deg_min =
            std::max(b.min_degree(), out_degree - c.max_degree());

        auto out_frag = out.degree_view(out_degree);
        if (a.has_degree(out_degree)) {
            const auto a_frag = a.degree_view(out_degree);
            for (Index i = 0; i < out_frag.size(); ++i) {
                out_frag[i] = static_cast<Scalar>(alpha * Accum{a_frag[i]});
            }
        }
        else {
            for (Index i = 0; i < out_frag.size(); ++i) {
                out_frag[i] = Scalar{0};
            }
        }

        for (Degree lhs_degree = lhs_deg_max; lhs_degree >= lhs_deg_min;
             --lhs_degree) {
            auto const rhs_degree = out_degree - lhs_degree;

            auto lhs_frag = b.degree_view(lhs_degree);
            auto rhs_frag = c.degree_view(rhs_degree);

            for (Index i = 0; i < lhs_frag.size(); ++i) {
                for (Index j = 0; j < rhs_frag.size(); ++j) {
                    auto& value = out_frag[i * rhs_frag.size() + j];
                    value = static_cast<Scalar>(Accum{value} +
                                                beta * Accum{lhs_frag[i]} *
                                                    Accum{rhs_frag[j]});
                }
            }
        }
    }

    if (out.min_degree() == 0) {
        Accum val{0};
        if (a.min_degree() == 0) {
            val += alpha * Accum{a[0]};
        }
        if (b.min_degree() == 0 && c.min_degree() == 0) {
            const Accum b_val{b[0]};
            const Accum c_val{c[0]};
            val += beta * b_val * c_val;
        }
        out[0] = static_cast<Scalar>(val);
    }
}

template <typename TensorLhs, typename TensorRhs, typename Accum>
void inplace_mul(TensorLhs& lhs, TensorRhs const& rhs, Accum beta) {
    using Degree = typename TensorLhs::Basis::Degree;
    using Index = typename TensorLhs::Basis::Index;

    using Scalar = typename TensorLhs::value_type;

    const auto out_min_degree = std::max(Degree{1}, lhs.min_degree());
    for (Degree out_degree = lhs.max_degree(); out_degree >= out_min_degree;
         --out_degree) {
        auto out_frag = lhs.degree_view(out_degree);

        if (rhs.min_degree() == 0) {
            for (Index i = 0; i < out_frag.size(); ++i) {
                out_frag[i] = static_cast<Scalar>(beta * Accum{out_frag[i]} *
                                                  Accum{rhs[0]});
            }
        }
        else {
            for (Index i = 0; i < out_frag.size(); ++i) {
                out_frag[i] = Scalar{0};
            }
        }

        const auto lhs_degree_min =
            std::max(lhs.min_degree(),
                     static_cast<Degree>(out_degree - rhs.max_degree()));
        const auto lhs_degree_max = static_cast<Degree>(
            out_degree - std::max(Degree{1}, rhs.min_degree()));
        if (lhs_degree_max < lhs_degree_min) {
            continue;
        }

        for (Degree lhs_degree = lhs_degree_max; lhs_degree >= lhs_degree_min;
             --lhs_degree) {
            const auto rhs_degree =
                static_cast<Degree>(out_degree - lhs_degree);
            if (!lhs.has_degree(lhs_degree) || !rhs.has_degree(rhs_degree)) {
                continue;
            }

            auto lhs_frag = lhs.degree_view(lhs_degree);
            auto rhs_frag = rhs.degree_view(rhs_degree);

            for (Index i = 0; i < lhs_frag.size(); ++i) {
                for (Index j = 0; j < rhs_frag.size(); ++j) {
                    const auto out_idx = i * rhs_frag.size() + j;
                    const Accum acc = Accum{out_frag[out_idx]} +
                        beta * Accum{lhs_frag[i]} * Accum{rhs_frag[j]};
                    out_frag[out_idx] = static_cast<Scalar>(acc);
                }
            }
        }
    }

    if (lhs.min_degree() == 0 && rhs.min_degree() == 0) {
        lhs[0] = static_cast<Scalar>(beta * Accum{lhs[0]} * Accum{rhs[0]});
    }
    else if (lhs.min_degree() == 0) {
        lhs[0] = Scalar{0};
    }
}

} // namespace rpp::tests::legacy

#endif // RPP_TESTS_CPU_LEGACY_FT_MULTIPLY_TEST_HELPER_HPP
