#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_DETAIL_FT_MULTIPLY_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_DETAIL_FT_MULTIPLY_HPP

// IWYU pragma: always_keep
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>

#include <rpp/config.h>

#include <rpp/operations/basic/ft_inplace_fma.hpp>
#include <rpp/support/span.hpp>
#include <rpp/utility.hpp>

namespace rpp::cpu::single_thread::ft_multiply {

template <typename Ctx>
using AccumOf = typename Ctx::Accum;

template <typename Ctx>
using IndexOf = typename Ctx::Index;

template <typename Ctx>
using DegreeOf = typename Ctx::Degree;


template <typename Ctx>
struct SubtilePosition {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;

    Degree width;
    Degree tile_letters;
    Index subtile_size; // Physical side length B, including unused lanes.

    Index middle_word;
    Degree middle_degree;

    Index prefix_subtile_offset = 0;
    Index suffix_subtile_offset = 0;

    Index tile_size; // W^tile_letters: full logical side length.
    Index prefix_subtile_size; // min(B, tile_size - prefix_subtile_offset).
    Index suffix_subtile_size; // min(B, tile_size - suffix_subtile_offset).
};


template <typename Ctx>
class ScaledTileWriter {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    Accum beta{1};

public:
    explicit ScaledTileWriter(Accum beta_ = Accum{1}) : beta(beta_) {}

    template <typename TensorOut, typename Tile>
    RPP_FORCEINLINE void
    operator()(Ctx const& ctx RPP_MAYBE_UNUSED,
               TensorOut& out,
               Tile tile,
               SubtilePosition<Ctx> const& subtile_position) {
        using Scalar = typename TensorOut::value_type;

        const auto degree =
            subtile_position.middle_degree + 2 * subtile_position.tile_letters;
        assert(out.has_degree(degree));
        auto out_level = out.degree_view(degree);

        auto& prefix = subtile_position.prefix_subtile_offset;
        auto& suffix = subtile_position.suffix_subtile_offset;
        auto& tile_size = subtile_position.tile_size;
        const auto middle_size =
            const_power(static_cast<Index>(subtile_position.width),
                        subtile_position.middle_degree);
        auto& middle_index = subtile_position.middle_word;
        auto& tile_width = subtile_position.subtile_size;

        for (Index i = 0; i < subtile_position.prefix_subtile_size; ++i) {
            for (Index j = 0; j < subtile_position.suffix_subtile_size; ++j) {
                auto index =
                    ((i + prefix) * middle_size + middle_index) * tile_size +
                    (j + suffix);
                out_level[index] =
                    static_cast<Scalar>(beta * tile[i * tile_width + j]);
            }
        }
    }
};

template <typename Ctx, typename TensorA>
class ScaledTileAddWriter {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    TensorA const& a;

    Accum alpha{1};
    Accum beta{1};

public:
    explicit ScaledTileAddWriter(TensorA const& a_,
                                 Accum alpha_ = Accum{1},
                                 Accum beta_ = Accum{1})
        : a(a_), alpha(alpha_), beta(beta_) {}

    template <typename TensorOut, typename Tile>
    RPP_FORCEINLINE void
    operator()(Ctx const& ctx RPP_MAYBE_UNUSED,
               TensorOut& out,
               Tile tile,
               SubtilePosition<Ctx> const& subtile_position) {
        using Scalar = typename TensorOut::value_type;

        const auto degree =
            subtile_position.middle_degree + 2 * subtile_position.tile_letters;
        assert(out.has_degree(degree));
        auto out_level = out.degree_view(degree);

        auto& prefix = subtile_position.prefix_subtile_offset;
        auto& suffix = subtile_position.suffix_subtile_offset;
        auto& tile_size = subtile_position.tile_size;
        const auto middle_size =
            const_power(static_cast<Index>(subtile_position.width),
                        subtile_position.middle_degree);
        auto& middle_index = subtile_position.middle_word;
        auto& tile_width = subtile_position.subtile_size;

        if (a.has_degree(degree)) {
            auto a_level = a.degree_view(degree);
            for (Index i = 0; i < subtile_position.prefix_subtile_size; ++i) {
                for (Index j = 0; j < subtile_position.suffix_subtile_size;
                     ++j) {
                    auto index = ((i + prefix) * middle_size + middle_index) *
                            tile_size +
                        (j + suffix);
                    out_level[index] = static_cast<Scalar>(
                        alpha * static_cast<Accum>(a_level[index]) +
                        beta * tile[i * tile_width + j]);
                }
            }
        }
        else {
            for (Index i = 0; i < subtile_position.prefix_subtile_size; ++i) {
                for (Index j = 0; j < subtile_position.suffix_subtile_size;
                     ++j) {
                    out_level[((i + prefix) * middle_size + middle_index) *
                                  tile_size +
                              (j + suffix)] =
                        static_cast<Scalar>(beta * tile[i * tile_width + j]);
                }
            }
        }
    }
};

// The caller accumulates the complete product tile before updating an operand,
// visiting output degrees from highest to lowest.
template <typename Ctx>
class InplaceMulTileWriter : public ScaledTileWriter<Ctx> {
public:
    using ScaledTileWriter<Ctx>::ScaledTileWriter;
};

// For 123 and 213 the tile contains a*b or b*a respectively, and c is
// the additive source. Operand selection belongs to the multiplication driver.
template <typename Ctx, ops::FTInplaceFMAType FMAType, typename TensorC = void>
class InplaceFmaTileWriter : public ScaledTileAddWriter<Ctx, TensorC> {
    static_assert(FMAType == ops::FTInplaceFMAType::AEqualsABPlusC ||
                  FMAType == ops::FTInplaceFMAType::AEqualsBAPlusC);

public:
    using ScaledTileAddWriter<Ctx, TensorC>::ScaledTileAddWriter;
};

// For 231 the additive source is the output itself. Read each old entry
// during the final write, after its product contributions have been gathered.
template <typename Ctx>
class InplaceFmaTileWriter<Ctx, ops::FTInplaceFMAType::AEqualsBCPlusA> {
    using Accum = AccumOf<Ctx>;
    Accum alpha;
    Accum beta;

public:
    explicit InplaceFmaTileWriter(Accum alpha_ = Accum{1},
                                  Accum beta_ = Accum{1})
        : alpha(alpha_), beta(beta_) {}

    template <typename TensorOut, typename Tile>
    RPP_FORCEINLINE void operator()(Ctx const& ctx,
                                    TensorOut& out,
                                    Tile tile,
                                    SubtilePosition<Ctx> const& pos) {
        ScaledTileAddWriter<Ctx, TensorOut>{out, alpha, beta}(
            ctx, out, tile, pos);
    }
};

namespace detail {


template <typename Ctx, unsigned TilePower>
RPP_FORCEINLINE void outer_product(Ctx const& ctx RPP_MAYBE_UNUSED,
                                   AccumOf<Ctx>* out_tile,
                                   AccumOf<Ctx> const* lhs_tile,
                                   AccumOf<Ctx> const* rhs_tile) {
    using Index = IndexOf<Ctx>;

    constexpr Index tile_size = Index{1} << TilePower;

    for (Index i = 0; i < tile_size; ++i) {
        for (Index j = 0; j < tile_size; ++j) {
            out_tile[i * tile_size + j] += lhs_tile[i] * rhs_tile[j];
        }
    }
}

template <typename Ctx, unsigned TilePower, typename TensorData>
RPP_FORCEINLINE void stride_read_panel(Ctx const& ctx RPP_MAYBE_UNUSED,
                                       AccumOf<Ctx>* panel,
                                       TensorData const& source,
                                       IndexOf<Ctx> count,
                                       IndexOf<Ctx> stride,
                                       IndexOf<Ctx> offset) {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;

    constexpr Index tile_size = Index{1} << TilePower;
    assert(count <= tile_size && "count must not exceed tile_size");

    for (Index i = 0; i < count; ++i) {
        panel[i] = static_cast<Accum>(source[i * stride + offset]);
    }
}


template <typename Ctx,
          unsigned TilePower,
          typename LhsTensor,
          typename RhsTensor>
void mid_terms(Ctx const& ctx RPP_MAYBE_UNUSED,
               Span<AccumOf<Ctx>, size_t{1} << (2 * TilePower)> out_tile,
               Span<AccumOf<Ctx>, size_t{1} << (2 * TilePower - 1)> left_panel,
               Span<AccumOf<Ctx>, size_t{1} << (2 * TilePower - 1)> right_panel,
               LhsTensor const& lhs,
               RhsTensor const& rhs,
               SubtilePosition<Ctx> const& subtile_position) {

    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    using Accum = AccumOf<Ctx>;

    static_assert(TilePower > 0, "panels require a positive tile power");
    constexpr Index tile_size = Index{1} << TilePower;
    constexpr Degree max_panels = Degree{1} << (TilePower - 1);

    const auto prefix_shift = subtile_position.tile_size;

    for (Degree split_deg_start = 0;
         split_deg_start <= subtile_position.middle_degree;
         split_deg_start += max_panels) {
        const auto num_panels = std::min<Degree>(
            subtile_position.middle_degree - split_deg_start + 1, max_panels);

        for (Degree panel_idx = 0; panel_idx < num_panels; ++panel_idx) {
            const auto split_deg = split_deg_start + panel_idx;
            Index subtile_split = const_power(
                static_cast<Index>(subtile_position.width), split_deg);

            const auto lhs_split = subtile_position.middle_word / subtile_split;
            const auto rhs_split = subtile_position.middle_word % subtile_split;

            const auto lhs_degree = subtile_position.middle_degree - split_deg +
                subtile_position.tile_letters;
            const auto rhs_degree = split_deg + subtile_position.tile_letters;

            if (!lhs.has_degree(lhs_degree) || !rhs.has_degree(rhs_degree)) {
                continue;
            }

            const Index lhs_stride =
                const_power(static_cast<Index>(subtile_position.width),
                            subtile_position.middle_degree - split_deg);
            const Index rhs_stride = 1;
            const auto lhs_idx = lhs_split;
            const auto rhs_idx = rhs_split * prefix_shift;

            stride_read_panel<Ctx, TilePower>(
                ctx,
                &left_panel[panel_idx * tile_size],
                lhs.degree_view(lhs_degree),
                subtile_position.prefix_subtile_size,
                lhs_stride,
                lhs_idx + subtile_position.prefix_subtile_offset * lhs_stride);
            stride_read_panel<Ctx, TilePower>(
                ctx,
                &right_panel[panel_idx * tile_size],
                rhs.degree_view(rhs_degree),
                subtile_position.suffix_subtile_size,
                rhs_stride,
                rhs_idx + subtile_position.suffix_subtile_offset * rhs_stride);

            outer_product<Ctx, TilePower>(ctx,
                                          out_tile.data(),
                                          &left_panel[panel_idx * tile_size],
                                          &right_panel[panel_idx * tile_size]);
        }
    }
}

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_DETAIL_FT_MULTIPLY_HPP
