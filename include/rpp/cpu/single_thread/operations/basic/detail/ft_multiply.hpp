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


// Chosen by the caller before entering the multiplication implementation.
template <typename Ctx, unsigned TilePower>
struct TileConfig {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    static_assert(TilePower > 0, "panels require a positive tile power");

    static constexpr Index subtile_size = Index{1} << TilePower;
    static constexpr size_t tile_extent = size_t{1} << (2 * TilePower);
    static constexpr size_t panel_extent = size_t{1} << (2 * TilePower - 1);

    Degree width;
    Degree tile_letters;
    Index tile_size; // W^tile_letters: full logical side length.
};

template <typename Ctx>
struct SubtilePosition {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;

    Index middle_word;
    Degree middle_degree;
    Index prefix_subtile_offset = 0;
    Index suffix_subtile_offset = 0;
    Index prefix_subtile_size; // Live rows in this subtile.
    Index suffix_subtile_size; // Live columns in this subtile.
};


template <typename Ctx>
class ScaledTileWriter {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    Accum beta{1};

public:
    explicit ScaledTileWriter(Accum beta_ = Accum{1}) : beta(beta_) {}

    template <unsigned TilePower, typename TensorOut, typename Tile>
    RPP_FORCEINLINE void
    operator()(Ctx const& ctx RPP_MAYBE_UNUSED,
               TensorOut& out,
               Tile tile,
               TileConfig<Ctx, TilePower> const& config,
               SubtilePosition<Ctx> const& subtile_position) {
        using Scalar = typename TensorOut::value_type;

        const auto degree =
            subtile_position.middle_degree + 2 * config.tile_letters;
        assert(out.has_degree(degree));
        auto out_level = out.degree_view(degree);

        auto& prefix = subtile_position.prefix_subtile_offset;
        auto& suffix = subtile_position.suffix_subtile_offset;
        auto const& tile_size = config.tile_size;
        const auto middle_size = const_power(static_cast<Index>(config.width),
                                             subtile_position.middle_degree);
        auto& middle_index = subtile_position.middle_word;
        constexpr auto tile_width = TileConfig<Ctx, TilePower>::subtile_size;

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

    // Low degrees use a contiguous tile rather than prefix/suffix rows.
    template <typename TensorOut, typename Tile>
    RPP_FORCEINLINE void operator()(Ctx const& ctx RPP_MAYBE_UNUSED,
                                    TensorOut& out,
                                    Tile tile,
                                    DegreeOf<Ctx> degree,
                                    Index offset) {
        using Scalar = typename TensorOut::value_type;
        assert(out.has_degree(degree));
        auto out_level = out.degree_view(degree);
        for (Index i = 0; i < static_cast<Index>(tile.size()); ++i) {
            out_level[offset + i] = static_cast<Scalar>(beta * tile[i]);
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

    template <unsigned TilePower, typename TensorOut, typename Tile>
    RPP_FORCEINLINE void
    operator()(Ctx const& ctx RPP_MAYBE_UNUSED,
               TensorOut& out,
               Tile tile,
               TileConfig<Ctx, TilePower> const& config,
               SubtilePosition<Ctx> const& subtile_position) {
        using Scalar = typename TensorOut::value_type;

        const auto degree =
            subtile_position.middle_degree + 2 * config.tile_letters;
        assert(out.has_degree(degree));
        auto out_level = out.degree_view(degree);

        auto& prefix = subtile_position.prefix_subtile_offset;
        auto& suffix = subtile_position.suffix_subtile_offset;
        auto const& tile_size = config.tile_size;
        const auto middle_size = const_power(static_cast<Index>(config.width),
                                             subtile_position.middle_degree);
        auto& middle_index = subtile_position.middle_word;
        constexpr auto tile_width = TileConfig<Ctx, TilePower>::subtile_size;

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

    template <typename TensorOut, typename Tile>
    RPP_FORCEINLINE void operator()(Ctx const& ctx RPP_MAYBE_UNUSED,
                                    TensorOut& out,
                                    Tile tile,
                                    DegreeOf<Ctx> degree,
                                    Index offset) {
        using Scalar = typename TensorOut::value_type;
        assert(out.has_degree(degree));
        auto out_level = out.degree_view(degree);
        if (a.has_degree(degree)) {
            const auto a_level = a.degree_view(degree);
            for (Index i = 0; i < static_cast<Index>(tile.size()); ++i) {
                out_level[offset + i] = static_cast<Scalar>(
                    alpha * static_cast<Accum>(a_level[offset + i]) +
                    beta * tile[i]);
            }
        }
        else {
            ScaledTileWriter<Ctx>{beta}(ctx, out, tile, degree, offset);
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

    template <unsigned TilePower, typename TensorOut, typename Tile>
    RPP_FORCEINLINE void operator()(Ctx const& ctx,
                                    TensorOut& out,
                                    Tile tile,
                                    TileConfig<Ctx, TilePower> const& config,
                                    SubtilePosition<Ctx> const& pos) {
        ScaledTileAddWriter<Ctx, TensorOut>{out, alpha, beta}(
            ctx, out, tile, config, pos);
    }

    template <typename TensorOut, typename Tile>
    RPP_FORCEINLINE void operator()(Ctx const& ctx,
                                    TensorOut& out,
                                    Tile tile,
                                    DegreeOf<Ctx> degree,
                                    IndexOf<Ctx> offset) {
        ScaledTileAddWriter<Ctx, TensorOut>{out, alpha, beta}(
            ctx, out, tile, degree, offset);
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
               TileConfig<Ctx, TilePower> const& config,
               SubtilePosition<Ctx> const& subtile_position) {

    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    using Accum = AccumOf<Ctx>;

    static_assert(TilePower > 0, "panels require a positive tile power");
    constexpr Index tile_size = Index{1} << TilePower;
    constexpr Degree max_panels = Degree{1} << (TilePower - 1);

    const auto prefix_shift = config.tile_size;

    for (Degree split_deg_start = 0;
         split_deg_start <= subtile_position.middle_degree;
         split_deg_start += max_panels) {
        const auto num_panels = std::min<Degree>(
            subtile_position.middle_degree - split_deg_start + 1, max_panels);

        for (Degree panel_idx = 0; panel_idx < num_panels; ++panel_idx) {
            const auto split_deg = split_deg_start + panel_idx;
            Index subtile_split =
                const_power(static_cast<Index>(config.width), split_deg);

            const auto lhs_split = subtile_position.middle_word / subtile_split;
            const auto rhs_split = subtile_position.middle_word % subtile_split;

            const auto lhs_degree = subtile_position.middle_degree - split_deg +
                config.tile_letters;
            const auto rhs_degree = split_deg + config.tile_letters;

            if (!lhs.has_degree(lhs_degree) || !rhs.has_degree(rhs_degree)) {
                continue;
            }

            const Index lhs_stride =
                const_power(static_cast<Index>(config.width),
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


// Split within the prefix block: prefix = short_lhs * remainder_size +
// remainder. Each remainder selects one contiguous rhs panel and a strided
// group of rows.
template <typename Ctx,
          unsigned TilePower,
          typename LhsTensor,
          typename RhsTensor>
void left_small_terms(
    Ctx const& ctx,
    Span<AccumOf<Ctx>, size_t{1} << (2 * TilePower)> out_tile,
    Span<AccumOf<Ctx>, size_t{1} << (2 * TilePower - 1)> left_panel,
    Span<AccumOf<Ctx>, size_t{1} << (2 * TilePower - 1)> right_panel,
    LhsTensor const& lhs,
    RhsTensor const& rhs,
    TileConfig<Ctx, TilePower> const& config,
    SubtilePosition<Ctx> const& pos) {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    constexpr Index tile_size = Index{1} << TilePower;
    if (config.tile_letters <= 1) {
        return;
    }
    // Multi-letter blocks fit in one tile; only single letters are subtiled.
    assert(config.tile_size <= tile_size);
    assert(pos.prefix_subtile_offset == 0 && pos.suffix_subtile_offset == 0);
    const auto middle_size =
        const_power(static_cast<Index>(config.width), pos.middle_degree);
    const auto degree = pos.middle_degree + 2 * config.tile_letters;
    for (Degree lhs_degree = 1; lhs_degree < config.tile_letters;
         ++lhs_degree) {
        const auto rhs_degree = degree - lhs_degree;
        if (!lhs.has_degree(lhs_degree) || !rhs.has_degree(rhs_degree)) {
            continue;
        }
        const auto short_size =
            const_power(static_cast<Index>(config.width), lhs_degree);
        const auto remainder_size = config.tile_size / short_size;
        stride_read_panel<Ctx, TilePower>(ctx,
                                          left_panel.data(),
                                          lhs.degree_view(lhs_degree),
                                          short_size,
                                          1,
                                          0);
        const auto right = rhs.degree_view(rhs_degree);
        for (Index remainder = 0; remainder < remainder_size; ++remainder) {
            const auto offset =
                (remainder * middle_size + pos.middle_word) * config.tile_size;
            stride_read_panel<Ctx, TilePower>(ctx,
                                              right_panel.data(),
                                              right,
                                              pos.suffix_subtile_size,
                                              1,
                                              offset);
            for (Index i = 0; i < short_size; ++i) {
                const auto row = i * remainder_size + remainder;
                for (Index j = 0; j < pos.suffix_subtile_size; ++j) {
                    out_tile[row * tile_size + j] +=
                        left_panel[i] * right_panel[j];
                }
            }
        }
    }
}

// Split within the suffix block: suffix = remainder * short_size + short_rhs.
// Each remainder selects a strided lhs panel and a contiguous group of columns.
template <typename Ctx,
          unsigned TilePower,
          typename LhsTensor,
          typename RhsTensor>
void right_small_terms(
    Ctx const& ctx,
    Span<AccumOf<Ctx>, size_t{1} << (2 * TilePower)> out_tile,
    Span<AccumOf<Ctx>, size_t{1} << (2 * TilePower - 1)> left_panel,
    Span<AccumOf<Ctx>, size_t{1} << (2 * TilePower - 1)> right_panel,
    LhsTensor const& lhs,
    RhsTensor const& rhs,
    TileConfig<Ctx, TilePower> const& config,
    SubtilePosition<Ctx> const& pos) {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    constexpr Index tile_size = Index{1} << TilePower;
    if (config.tile_letters <= 1) {
        return;
    }
    assert(config.tile_size <= tile_size);
    assert(pos.prefix_subtile_offset == 0 && pos.suffix_subtile_offset == 0);
    const auto middle_size =
        const_power(static_cast<Index>(config.width), pos.middle_degree);
    const auto degree = pos.middle_degree + 2 * config.tile_letters;
    for (Degree rhs_degree = 1; rhs_degree < config.tile_letters;
         ++rhs_degree) {
        const auto lhs_degree = degree - rhs_degree;
        if (!lhs.has_degree(lhs_degree) || !rhs.has_degree(rhs_degree)) {
            continue;
        }
        const auto short_size =
            const_power(static_cast<Index>(config.width), rhs_degree);
        const auto remainder_size = config.tile_size / short_size;
        stride_read_panel<Ctx, TilePower>(ctx,
                                          right_panel.data(),
                                          rhs.degree_view(rhs_degree),
                                          short_size,
                                          1,
                                          0);
        const auto left = lhs.degree_view(lhs_degree);
        for (Index remainder = 0; remainder < remainder_size; ++remainder) {
            const auto offset = pos.middle_word * remainder_size + remainder;
            stride_read_panel<Ctx, TilePower>(ctx,
                                              left_panel.data(),
                                              left,
                                              pos.prefix_subtile_size,
                                              middle_size * remainder_size,
                                              offset);
            for (Index i = 0; i < pos.prefix_subtile_size; ++i) {
                for (Index j = 0; j < short_size; ++j) {
                    const auto column = remainder * short_size + j;
                    out_tile[i * tile_size + column] +=
                        left_panel[i] * right_panel[j];
                }
            }
        }
    }
}

// The two scalar contributions lhs[0] * rhs[d] and lhs[d] * rhs[0].
// Each source row is contiguous; no word splitting is needed.
template <typename Ctx,
          unsigned TilePower,
          typename LhsTensor,
          typename RhsTensor>
void scalar_terms(AccumOf<Ctx>* out_tile,
                  LhsTensor const& lhs,
                  RhsTensor const& rhs,
                  TileConfig<Ctx, TilePower> const& config,
                  SubtilePosition<Ctx> const& pos) {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    constexpr Index tile_size = TileConfig<Ctx, TilePower>::subtile_size;
    const auto degree = pos.middle_degree + 2 * config.tile_letters;
    const auto middle_size =
        const_power(static_cast<Index>(config.width), pos.middle_degree);
    const auto row_stride = middle_size * config.tile_size;
    const auto offset =
        (pos.prefix_subtile_offset * middle_size + pos.middle_word) *
            config.tile_size +
        pos.suffix_subtile_offset;
    if (lhs.has_degree(0) && rhs.has_degree(degree)) {
        const auto scalar = static_cast<Accum>(lhs[0]);
        const auto right = rhs.degree_view(degree);
        for (Index i = 0; i < pos.prefix_subtile_size; ++i) {
            const auto source_offset = offset + i * row_stride;
            for (Index j = 0; j < pos.suffix_subtile_size; ++j) {
                out_tile[i * tile_size + j] +=
                    scalar * static_cast<Accum>(right[source_offset + j]);
            }
        }
    }
    if (lhs.has_degree(degree) && rhs.has_degree(0)) {
        const auto left = lhs.degree_view(degree);
        const auto scalar = static_cast<Accum>(rhs[0]);
        for (Index i = 0; i < pos.prefix_subtile_size; ++i) {
            const auto source_offset = offset + i * row_stride;
            for (Index j = 0; j < pos.suffix_subtile_size; ++j) {
                out_tile[i * tile_size + j] +=
                    static_cast<Accum>(left[source_offset + j]) * scalar;
            }
        }
    }
}

// Without subtiling, each remaining degree fits in one flat storage tile.
// Read operand levels directly and write only after the whole product is ready.
template <typename Ctx,
          unsigned TilePower,
          typename TensorOut,
          typename LhsTensor,
          typename RhsTensor,
          typename Writer>
void low_degree_terms(Ctx const& ctx,
                      TileConfig<Ctx, TilePower> const& config,
                      TensorOut& out,
                      LhsTensor const& lhs,
                      RhsTensor const& rhs,
                      Span<AccumOf<Ctx>> tile,
                      Writer& write) {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    const auto low_max =
        std::min<Degree>(out.max_degree(), 2 * config.tile_letters - 1);
    for (Degree degree = low_max; degree >= out.min_degree(); --degree) {
        const auto count =
            const_power(static_cast<Index>(config.width), degree);
        assert(static_cast<size_t>(count) <= tile.size());
        std::fill_n(tile.data(), count, Accum{0});
        const auto rhs_min =
            std::max<Degree>(rhs.min_degree(), degree - lhs.max_degree());
        const auto rhs_max =
            std::min<Degree>(rhs.max_degree(), degree - lhs.min_degree());
        for (Degree rhs_degree = rhs_min; rhs_degree <= rhs_max; ++rhs_degree) {
            const auto left = lhs.degree_view(degree - rhs_degree);
            const auto right = rhs.degree_view(rhs_degree);
            const auto rhs_size = static_cast<Index>(right.size());
            for (Index i = 0; i < left.size(); ++i) {
                const auto value = static_cast<Accum>(left[i]);
                for (Index j = 0; j < rhs_size; ++j) {
                    tile[i * rhs_size + j] +=
                        value * static_cast<Accum>(right[j]);
                }
            }
        }
        write(ctx,
              out,
              Span<Accum const>{tile.data(), static_cast<size_t>(count)},
              degree,
              Index{0});
    }
}

// With single-letter tiles, degrees >= 2 belong to the main tiled loop.
// Only the scalar-vector products at degree 1 and the scalar product remain.
template <typename Ctx,
          unsigned TilePower,
          typename TensorOut,
          typename LhsTensor,
          typename RhsTensor,
          typename Writer>
void subtiled_low_degree_terms(Ctx const& ctx,
                               TileConfig<Ctx, TilePower> const& config,
                               TensorOut& out,
                               LhsTensor const& lhs,
                               RhsTensor const& rhs,
                               Span<AccumOf<Ctx>> tile,
                               Writer& write) {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    assert(config.tile_letters == 1);
    if (out.has_degree(1)) {
        const auto width = static_cast<Index>(config.width);
        const auto capacity = static_cast<Index>(tile.size());
        for (Index offset = 0; offset < width; offset += capacity) {
            const auto count = std::min(capacity, width - offset);
            std::fill_n(tile.data(), count, Accum{0});
            if (lhs.has_degree(0) && rhs.has_degree(1)) {
                const auto scalar = static_cast<Accum>(lhs[0]);
                const auto right = rhs.degree_view(1);
                for (Index i = 0; i < count; ++i) {
                    tile[i] += scalar * static_cast<Accum>(right[offset + i]);
                }
            }
            if (lhs.has_degree(1) && rhs.has_degree(0)) {
                const auto left = lhs.degree_view(1);
                const auto scalar = static_cast<Accum>(rhs[0]);
                for (Index i = 0; i < count; ++i) {
                    tile[i] += static_cast<Accum>(left[offset + i]) * scalar;
                }
            }
            write(ctx,
                  out,
                  Span<Accum const>{tile.data(), static_cast<size_t>(count)},
                  1,
                  offset);
        }
    }
    if (out.has_degree(0)) {
        tile[0] = Accum{0};
        if (lhs.has_degree(0) && rhs.has_degree(0)) {
            tile[0] = static_cast<Accum>(lhs[0]) * static_cast<Accum>(rhs[0]);
        }
        write(ctx, out, Span<Accum const>{tile.data(), size_t{1}}, 0, Index{0});
    }
}


template <unsigned TilePower, typename Ctx, typename Basis>
TileConfig<Ctx, TilePower> select_tile_config(Ctx const& ctx RPP_MAYBE_UNUSED,
                                              Basis const& basis) {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    using Config = TileConfig<Ctx, TilePower>;

    Config config{basis.width, Degree{1}, static_cast<Index>(basis.width)};

    const auto width = static_cast<Index>(config.width);

    for (; config.tile_letters < basis.depth &&
         static_cast<size_t>(config.tile_size * config.tile_size * width * width) <= Config::tile_extent;
         ++config.tile_letters) {
        config.tile_size *= width;
    }

    return config;
}

} // namespace detail


// Tile dimensions and the implementation specialization are selected by the
// caller. The writer receives a complete tile, with either its subtile position
// or its degree and flat offset for low degrees. Output degrees are visited in
// descending order so a writer can also update either operand in place. Output
// and operands must use the same tensor basis.
template <unsigned TilePower,
          typename Ctx,
          typename TensorOut,
          typename LhsTensor,
          typename RhsTensor,
          typename Writer>
void multiply(Ctx const& ctx,
              TileConfig<Ctx, TilePower> const& config,
              TensorOut& out,
              LhsTensor const& lhs,
              RhsTensor const& rhs,
              Writer&& write) {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    constexpr Index tile_size = TileConfig<Ctx, TilePower>::subtile_size;
    constexpr size_t tile_extent = TileConfig<Ctx, TilePower>::tile_extent;
    constexpr size_t panel_extent = TileConfig<Ctx, TilePower>::panel_extent;
    const auto width = static_cast<Index>(config.width);
    const auto tile_letters = config.tile_letters;
    const auto side = config.tile_size;
    assert(width > 0 && tile_letters > 0 && side > 0);
    assert(side <= tile_size || tile_letters == 1);
    assert(out.basis().width == config.width);

    alignas(64) std::array<Accum, tile_extent> tile {};
    // Initialize once: unused lanes may retain values between panel reads.
    alignas(64) std::array<Accum, panel_extent> left_panel{};
    alignas(64) std::array<Accum, panel_extent> right_panel{};


    const auto min_middle =
        std::max<Degree>(0, out.min_degree() - 2 * tile_letters);
    const auto max_middle = out.max_degree() - 2 * tile_letters;

    for (Degree middle_degree = max_middle; middle_degree >= min_middle;
         --middle_degree) {
        const auto middle_size = const_power(width, middle_degree);
        for (Index middle_word = 0; middle_word < middle_size; ++middle_word) {
            for (Index prefix = 0; prefix < side; prefix += tile_size) {
                for (Index suffix = 0; suffix < side; suffix += tile_size) {
                    SubtilePosition<Ctx> pos{
                        middle_word,
                        static_cast<Degree>(middle_degree),
                        prefix,
                        suffix,
                        std::min(tile_size, side - prefix),
                        std::min(tile_size, side - suffix)};
                    std::fill(tile.begin(), tile.end(), Accum{0});
                    detail::scalar_terms<Ctx, TilePower>(
                        tile.data(), lhs, rhs, config, pos);
                    detail::left_small_terms<Ctx, TilePower>(ctx,
                                                             tile.data(),
                                                             left_panel.data(),
                                                             right_panel.data(),
                                                             lhs,
                                                             rhs,
                                                             config,
                                                             pos);
                    detail::mid_terms<Ctx, TilePower>(ctx,
                                                      tile.data(),
                                                      left_panel.data(),
                                                      right_panel.data(),
                                                      lhs,
                                                      rhs,
                                                      config,
                                                      pos);
                    detail::right_small_terms<Ctx, TilePower>(
                        ctx,
                        tile.data(),
                        left_panel.data(),
                        right_panel.data(),
                        lhs,
                        rhs,
                        config,
                        pos);
                    write(ctx,
                          out,
                          Span<Accum const, tile_extent>{tile.data(),
                                                         tile_extent},
                          config,
                          pos);
                }
            }
        }
    }

    if (width > static_cast<Index>(tile_extent)) {
        detail::subtiled_low_degree_terms(
            ctx, config, out, lhs, rhs, {tile.data(), tile.size()}, write);
    }
    else {
        detail::low_degree_terms(
            ctx, config, out, lhs, rhs, {tile.data(), tile.size()}, write);
    }
}


} // namespace rpp::cpu::single_thread::ft_multiply

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_DETAIL_FT_MULTIPLY_HPP
