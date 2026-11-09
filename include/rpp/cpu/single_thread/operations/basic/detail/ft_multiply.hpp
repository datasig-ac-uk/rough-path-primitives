#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_DETAIL_FT_MULTIPLY_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_DETAIL_FT_MULTIPLY_HPP

// IWYU pragma: always_keep
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <new>
#include <type_traits>

#include <rpp/config.h>

#include <rpp/basis/tensor_basis.hpp>
#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/ft_inplace_fma.hpp>
#include <rpp/utility.hpp>
#include <rpp/views/dense_tensor_view.hpp>

#define RPP_NOINLINE __attribute__((noinline))

namespace rpp::cpu::single_thread::ft_multiply {

inline constexpr unsigned default_tile_power = 5;

template <typename Ctx>
using AccumOf = typename Ctx::Accum;

template <typename Ctx>
using IndexOf = typename Ctx::Index;

template <typename Ctx>
using DegreeOf = typename Ctx::Degree;

template <typename S>
RPP_FORCEINLINE bool is_zero(S const& scalar) {
    return scalar == S{0};
}

// Chosen by the caller before entering the multiplication implementation.
template <typename Ctx, unsigned TilePower>
struct TileConfig {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    static_assert(TilePower > 0, "panels require a positive tile power");

    static constexpr Index vector_size = 4;
    static constexpr Index subtile_size = Index{1} << TilePower;
    static constexpr size_t tile_extent = size_t{1} << (2 * TilePower);
    static constexpr size_t panel_extent = size_t{1} << (2 * TilePower - 1);

    Degree width;
    Degree tile_letters;
    Index tile_size; // W^tile_letters: full logical side length.
    Degree first_tiled_degree = 0; // Resolved independently of view bounds.
};

// One live storage object per operation context, reused across multiplications.
// Reserve alignment padding because the underlying byte allocation may be
// unaligned.
template <typename Ctx, unsigned TilePower>
struct TileStorage {
    using Accum = AccumOf<Ctx>;
    using Config = TileConfig<Ctx, TilePower>;

    static constexpr size_t alignment = std::max(size_t{64}, alignof(Accum));
    alignas(alignment) std::array<Accum, Config::tile_extent> tile;
    alignas(alignment) std::array<Accum, Config::panel_extent> left_panel{};
    alignas(alignment) std::array<Accum, Config::panel_extent> right_panel{};

    static constexpr size_t scratch_size() noexcept {
        return sizeof(TileStorage) + alignof(TileStorage) - 1;
    }

    static void* address(Ctx const& ctx) noexcept {
        auto* data = ctx.template scratch_space<std::byte*>();
        assert(data != nullptr &&
               "multiplication requires initialized scratch space");
        return align_up<alignof(TileStorage)>(data);
    }

    static TileStorage& get(Ctx const& ctx) noexcept {
        return *std::launder(static_cast<TileStorage*>(address(ctx)));
    }

    static void construct(Ctx const& ctx) { ::new (address(ctx)) TileStorage; }

    static void destroy(Ctx const& ctx) { get(ctx).~TileStorage(); }
};

// Common scratch lifecycle for the CPU multiplication and FMA operations.
template <typename Strategy, unsigned TilePower>
class Operation : public ops::BaseOperation<Strategy> {
    using Context = typename Strategy::Context;
    using Storage = TileStorage<Context, TilePower>;

    using Index = IndexOf<Context>;
    using Degree = DegreeOf<Context>;

    using Config = TileConfig<Context, TilePower>;

    Config config_;

public:

    Config const& config() const noexcept { return config_; }

    template <typename BasisPack>
    explicit Operation(Context const& ctx, BasisPack const& bases) noexcept
        : ops::BaseOperation<Strategy>(ctx, bases)
    {
        auto const& tbasis = basis::get_basis(basis::TensorBasisTag{}, bases);
        constexpr auto tile_width = Index{1} << TilePower;

        config_.width = tbasis.width;
        config_.tile_letters = 1;
        config_.tile_size = static_cast<Index>(config_.width);

        const auto width = static_cast<Index>(config_.width);
        config_.first_tiled_degree = 2 + (width <= tile_width ? 1 : 0);


        if (width > 1 && width < tile_width) {
            for (; config_.tile_size * width <= tile_width;
                 ++config_.tile_letters, config_.tile_size *= width) {
            }

            // Degree 2k already fits. Extend the contiguous range from degree 2k+1.
            config_.first_tiled_degree = 2 * config_.tile_letters + 1;
            size_t level_size =
                static_cast<size_t>(config_.tile_size) * config_.tile_size * width;
            for (; level_size <= Config::tile_extent;
                 ++config_.first_tiled_degree, level_size *= width) {
            }
        }
    }

    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const&,
                                               Basis const&) noexcept {
        return Storage::scratch_size();
    }

    template <typename Basis>
    static void init_scratch_space(Context const& ctx, Basis const&) {
        Storage::construct(ctx);
    }

    template <typename Basis>
    static void destroy_scratch_space(Context const& ctx, Basis const&) {
        Storage::destroy(ctx);
    }
};

template <typename Ctx>
struct SubtilePosition {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;

    Index middle_word;
    Degree middle_degree;
    Index middle_size; // width^middle_degree, computed once by the driver.
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

    template <unsigned TilePower, typename TensorOut>
    RPP_FORCEINLINE void
    operator()(Ctx const& ctx RPP_MAYBE_UNUSED,
               TensorOut& out,
               Accum const* RPP_RESTRICT tile,
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
        const auto middle_size = subtile_position.middle_size;
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
    template <typename TensorOut>
    RPP_FORCEINLINE void operator()(Ctx const& ctx RPP_MAYBE_UNUSED,
                                    TensorOut& out,
                                    Accum const* RPP_RESTRICT tile,
                                    DegreeOf<Ctx> degree,
                                    Index offset,
                                    Index count) {
        using Scalar = typename TensorOut::value_type;
        assert(out.has_degree(degree));
        auto out_level = out.degree_view(degree);
        for (Index i = 0; i < count; ++i) {
            out_level[offset + i] = static_cast<Scalar>(beta * tile[i]);
        }
    }

    // The low-degree tile is a complete contiguous tensor prefix.
    template <typename TensorOut, typename TensorTile>
    RPP_FORCEINLINE void
    operator()(Ctx const&, TensorOut& out, TensorTile const& tile) {
        using Scalar = typename TensorOut::value_type;
        for (Index i = tile.begin_index(); i < tile.end_index(); ++i) {
            out[i] = static_cast<Scalar>(beta * tile[i]);
        }
    }

    // Width one: the tile contains consecutive degrees, one coefficient each.
    template <typename TensorOut>
    RPP_FORCEINLINE void operator()(Ctx const& ctx RPP_MAYBE_UNUSED,
                                    TensorOut& out,
                                    Accum const* RPP_RESTRICT tile,
                                    DegreeOf<Ctx> first_degree,
                                    IndexOf<Ctx> count) {
        using Scalar = typename TensorOut::value_type;
        for (Index i = 0; i < count; ++i) {
            out[first_degree + i] = static_cast<Scalar>(beta * tile[i]);
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

    template <unsigned TilePower, typename TensorOut>
    RPP_FORCEINLINE void
    operator()(Ctx const& ctx RPP_MAYBE_UNUSED,
               TensorOut& out,
               Accum const* RPP_RESTRICT tile,
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
        const auto middle_size = subtile_position.middle_size;
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

    template <typename TensorOut, typename TensorTile>
    RPP_FORCEINLINE void
    operator()(Ctx const&, TensorOut& out, TensorTile const& tile) {
        using Scalar = typename TensorOut::value_type;
        const auto first = tile.begin_index();
        const auto last = tile.end_index();
        const auto add_first = std::clamp<Index>(a.begin_index(), first, last);
        const auto add_last = std::clamp<Index>(a.end_index(), add_first, last);
        for (Index i = first; i < add_first; ++i) {
            out[i] = static_cast<Scalar>(beta * tile[i]);
        }
        for (Index i = add_first; i < add_last; ++i) {
            out[i] = static_cast<Scalar>(alpha * static_cast<Accum>(a[i]) +
                                         beta * tile[i]);
        }
        for (Index i = add_last; i < last; ++i) {
            out[i] = static_cast<Scalar>(beta * tile[i]);
        }
    }

    template <typename TensorOut>
    RPP_FORCEINLINE void operator()(Ctx const& ctx RPP_MAYBE_UNUSED,
                                    TensorOut& out,
                                    Accum const* RPP_RESTRICT tile,
                                    DegreeOf<Ctx> degree,
                                    Index offset,
                                    Index count) {
        using Scalar = typename TensorOut::value_type;
        assert(out.has_degree(degree));
        auto out_level = out.degree_view(degree);
        if (a.has_degree(degree)) {
            const auto a_level = a.degree_view(degree);
            for (Index i = 0; i < count; ++i) {
                out_level[offset + i] = static_cast<Scalar>(
                    alpha * static_cast<Accum>(a_level[offset + i]) +
                    beta * tile[i]);
            }
        }
        else {
            ScaledTileWriter<Ctx>{beta}(ctx, out, tile, degree, offset, count);
        }
    }

    template <typename TensorOut>
    RPP_FORCEINLINE void operator()(Ctx const& ctx RPP_MAYBE_UNUSED,
                                    TensorOut& out,
                                    Accum const* RPP_RESTRICT tile,
                                    DegreeOf<Ctx> first_degree,
                                    IndexOf<Ctx> count) {
        using Scalar = typename TensorOut::value_type;
        for (Index i = 0; i < count; ++i) {
            const auto degree = first_degree + i;
            Accum value = beta * tile[i];
            if (a.has_degree(degree)) {
                value += alpha * static_cast<Accum>(a[degree]);
            }
            out[degree] = static_cast<Scalar>(value);
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
template <typename Ctx, typename TensorC>
class InplaceFmaTileWriter<Ctx,
                           ops::FTInplaceFMAType::AEqualsBCPlusA,
                           TensorC> {
    using Accum = AccumOf<Ctx>;
    Accum alpha;
    Accum beta;

public:
    explicit InplaceFmaTileWriter(Accum alpha_ = Accum{1},
                                  Accum beta_ = Accum{1})
        : alpha(alpha_), beta(beta_) {}

    template <typename TensorOut, typename TensorTile>
    RPP_FORCEINLINE void
    operator()(Ctx const& ctx, TensorOut& out, TensorTile const& tile) {
        ScaledTileAddWriter<Ctx, TensorOut>{out, alpha, beta}(ctx, out, tile);
    }

    template <unsigned TilePower, typename TensorOut>
    RPP_FORCEINLINE void operator()(Ctx const& ctx,
                                    TensorOut& out,
                                    Accum const* RPP_RESTRICT tile,
                                    TileConfig<Ctx, TilePower> const& config,
                                    SubtilePosition<Ctx> const& pos) {
        ScaledTileAddWriter<Ctx, TensorOut>{out, alpha, beta}(
            ctx, out, tile, config, pos);
    }

    template <typename TensorOut>
    RPP_FORCEINLINE void operator()(Ctx const& ctx,
                                    TensorOut& out,
                                    Accum const* RPP_RESTRICT tile,
                                    DegreeOf<Ctx> degree,
                                    IndexOf<Ctx> offset,
                                    IndexOf<Ctx> count) {
        ScaledTileAddWriter<Ctx, TensorOut>{out, alpha, beta}(
            ctx, out, tile, degree, offset, count);
    }

    template <typename TensorOut>
    RPP_FORCEINLINE void operator()(Ctx const& ctx,
                                    TensorOut& out,
                                    Accum const* RPP_RESTRICT tile,
                                    DegreeOf<Ctx> first_degree,
                                    IndexOf<Ctx> count) {
        ScaledTileAddWriter<Ctx, TensorOut>{out, alpha, beta}(
            ctx, out, tile, first_degree, count);
    }
};

namespace detail {


// Clear only the rectangle touched by the vector-rounded outer product.
// The remaining physical rows and columns are never read or written back.
template <typename Ctx, unsigned TilePower>
void clear_tile(AccumOf<Ctx>* RPP_RESTRICT tile,
                SubtilePosition<Ctx> const& pos) {
    using Index = IndexOf<Ctx>;
    using Config = TileConfig<Ctx, TilePower>;
    constexpr Index vector_size = Config::vector_size;
    constexpr Index tile_size = Config::subtile_size;
    static_assert(tile_size >= vector_size,
                  "tile size must be at least the vector size");

    tile = assume_aligned<16>(tile);

    // for (Index ii=0; ii<pos.prefix_subtile_size; ii += vector_size) {
    //     for (Index jj=0; jj<pos.suffix_subtile_size; jj += vector_size) {
    //         for (Index i=0; i<vector_size; ++i) {
    //             for (Index j=0; j<vector_size; ++j) {
    //                 tile[(ii + i) * tile_size + (jj +j)] = AccumOf<Ctx>{0};
    //             }
    //         }
    //     }
    // }

    for (Index i = 0; i < pos.prefix_subtile_size; ++i) {
        for (Index jj = 0; jj < pos.suffix_subtile_size; jj += vector_size) {
            for (Index j = 0; j < vector_size; ++j) {
                tile[i * tile_size + (jj + j)] = AccumOf<Ctx>{0};
            }
        }
    }
}

template <typename Ctx, unsigned TilePower, typename TensorData>
void stride_read_panel(Ctx const& ctx RPP_MAYBE_UNUSED,
                       AccumOf<Ctx>* RPP_RESTRICT panel,
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

// Exact-count kernels can read tensor storage directly. Overrun variants round
// up to a lane group and require initialized padding in both input and output.
template <typename Ctx,
          unsigned TilePower,
          typename LhsScalar,
          typename RhsScalar>
void outer_product(Ctx const& ctx RPP_MAYBE_UNUSED,
                   TileConfig<Ctx, TilePower> const& config RPP_MAYBE_UNUSED,
                   AccumOf<Ctx>* RPP_RESTRICT out,
                   LhsScalar const* lhs,
                   IndexOf<Ctx> lhs_size,
                   RhsScalar const* rhs,
                   IndexOf<Ctx> rhs_size,
                   IndexOf<Ctx> out_stride) {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    for (Index i = 0; i < lhs_size; ++i) {
        const auto lhs_val = static_cast<Accum>(lhs[i]);
        for (Index j = 0; j < rhs_size; ++j) {
            out[i * out_stride + j] += lhs_val * static_cast<Accum>(rhs[j]);
        }
    }
}

template <typename Ctx,
          unsigned TilePower,
          typename LhsScalar,
          typename RhsScalar>
void outer_product_overrun(Ctx const& ctx RPP_MAYBE_UNUSED,
                           TileConfig<Ctx, TilePower> const& config
                               RPP_MAYBE_UNUSED,
                           AccumOf<Ctx>* RPP_RESTRICT out,
                           LhsScalar const* lhs,
                           IndexOf<Ctx> lhs_size,
                           RhsScalar const* rhs,
                           IndexOf<Ctx> rhs_size,
                           IndexOf<Ctx> out_stride) {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    constexpr Index vector_size = TileConfig<Ctx, TilePower>::vector_size;
    for (Index i = 0; i < lhs_size; ++i) {
        const auto lhs_val = static_cast<Accum>(lhs[i]);
        for (Index jj = 0; jj < rhs_size; jj += vector_size) {
            for (Index j = 0; j < vector_size; ++j) {
                out[i * out_stride + jj + j] +=
                    lhs_val * static_cast<Accum>(rhs[jj + j]);
            }
        }
    }
}

// Packed panels retain the alignment guarantees of the original tiled kernel.
template <typename Ctx, unsigned TilePower>
void outer_product(Ctx const& ctx,
                   TileConfig<Ctx, TilePower> const& config,
                   SubtilePosition<Ctx> const& pos,
                   AccumOf<Ctx>* RPP_RESTRICT out_tile,
                   AccumOf<Ctx> const* RPP_RESTRICT lhs_tile,
                   AccumOf<Ctx> const* RPP_RESTRICT rhs_tile) {
    constexpr auto tile_size = TileConfig<Ctx, TilePower>::subtile_size;
    static_assert(tile_size >= TileConfig<Ctx, TilePower>::vector_size,
                  "tile size must be at least the vector size");
    outer_product_overrun(ctx,
                          config,
                          assume_aligned<32>(out_tile),
                          assume_aligned<32>(lhs_tile),
                          pos.prefix_subtile_size,
                          assume_aligned<32>(rhs_tile),
                          pos.suffix_subtile_size,
                          tile_size);
}

template <typename Ctx, unsigned TilePower, typename RhsScalar>
void left_multiply_range(Ctx const& ctx RPP_MAYBE_UNUSED,
                         TileConfig<Ctx, TilePower> const& config
                             RPP_MAYBE_UNUSED,
                         AccumOf<Ctx>* RPP_RESTRICT out,
                         AccumOf<Ctx> lhs_val,
                         RhsScalar const* rhs,
                         IndexOf<Ctx> rhs_size,
                         IndexOf<Ctx> out_stride) {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    for (Index i = 0; i < rhs_size; ++i) {
        out[i * out_stride] += lhs_val * static_cast<Accum>(rhs[i]);
    }
}

template <typename Ctx, unsigned TilePower, typename RhsScalar>
void left_multiply_range_overrun(Ctx const& ctx RPP_MAYBE_UNUSED,
                                 TileConfig<Ctx, TilePower> const& config
                                     RPP_MAYBE_UNUSED,
                                 AccumOf<Ctx>* RPP_RESTRICT out,
                                 AccumOf<Ctx> lhs_val,
                                 RhsScalar const* rhs,
                                 IndexOf<Ctx> rhs_size,
                                 IndexOf<Ctx> out_stride) {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    constexpr Index vector_size = TileConfig<Ctx, TilePower>::vector_size;
    for (Index ii = 0; ii < rhs_size; ii += vector_size) {
        for (Index i = 0; i < vector_size; ++i) {
            out[(ii + i) * out_stride] +=
                lhs_val * static_cast<Accum>(rhs[ii + i]);
        }
    }
}

template <typename Ctx, unsigned TilePower, typename LhsScalar>
void right_multiply_range(Ctx const& ctx RPP_MAYBE_UNUSED,
                          TileConfig<Ctx, TilePower> const& config
                              RPP_MAYBE_UNUSED,
                          AccumOf<Ctx>* RPP_RESTRICT out,
                          LhsScalar const* lhs,
                          IndexOf<Ctx> lhs_size,
                          AccumOf<Ctx> rhs_val,
                          IndexOf<Ctx> out_stride) {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    for (Index i = 0; i < lhs_size; ++i) {
        out[i * out_stride] += static_cast<Accum>(lhs[i]) * rhs_val;
    }
}

template <typename Ctx, unsigned TilePower, typename LhsScalar>
void right_multiply_range_overrun(Ctx const& ctx RPP_MAYBE_UNUSED,
                                  TileConfig<Ctx, TilePower> const& config
                                      RPP_MAYBE_UNUSED,
                                  AccumOf<Ctx>* RPP_RESTRICT out,
                                  LhsScalar const* lhs,
                                  IndexOf<Ctx> lhs_size,
                                  AccumOf<Ctx> rhs_val,
                                  IndexOf<Ctx> out_stride) {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    constexpr Index vector_size = TileConfig<Ctx, TilePower>::vector_size;
    for (Index ii = 0; ii < lhs_size; ii += vector_size) {
        for (Index i = 0; i < vector_size; ++i) {
            out[(ii + i) * out_stride] +=
                static_cast<Accum>(lhs[ii + i]) * rhs_val;
        }
    }
}


// All decompositions of a sequential middle word into a prefix and suffix.
// Powers are fixed for an output degree; advancing uses a carry instead of
// dividing the word again for every split and subtile.
template <typename Ctx>
class MiddleWordSplits {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    using Architecture = typename Ctx::Architecture;

    struct Split {
        Index divisor;
        Index lhs;
        Index rhs;
    };

    std::array<Split, Architecture::max_depth + 1> splits;
    Degree middle_degree;

public:
    MiddleWordSplits(Index width, Degree degree) : middle_degree(degree) {
        assert(width > 1 && degree >= 0 &&
               static_cast<size_t>(degree) <= Architecture::max_depth);
        Index divisor = 1;
        for (Degree split = 0; split <= middle_degree; ++split) {
            splits[split] = {divisor, Index{0}, Index{0}};
            if (split < middle_degree) {
                divisor *= width;
            }
        }
    }

    auto const& operator[](Degree split) const noexcept {
        return splits[split];
    }

    void advance() noexcept {
        for (Degree split = 0; split <= middle_degree; ++split) {
            auto& state = splits[split];
            if (++state.rhs == state.divisor) {
                state.rhs = 0;
                ++state.lhs;
            }
        }
    }
};

template <typename Ctx,
          unsigned TilePower,
          typename LhsTensor,
          typename RhsTensor>
void mid_terms(Ctx const& ctx RPP_MAYBE_UNUSED,
               AccumOf<Ctx>* RPP_RESTRICT out_tile,
               AccumOf<Ctx>* RPP_RESTRICT left_panel,
               AccumOf<Ctx>* RPP_RESTRICT right_panel,
               LhsTensor const& lhs,
               RhsTensor const& rhs,
               TileConfig<Ctx, TilePower> const& config,
               SubtilePosition<Ctx> const& subtile_position,
               MiddleWordSplits<Ctx> const& splits) {

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
            const Index lhs_stride =
                splits[subtile_position.middle_degree - split_deg].divisor;

            const auto lhs_split = splits[split_deg].lhs;
            const auto rhs_split = splits[split_deg].rhs;

            const auto lhs_degree = subtile_position.middle_degree - split_deg +
                config.tile_letters;
            const auto rhs_degree = split_deg + config.tile_letters;

            if (!lhs.has_degree(lhs_degree) || !rhs.has_degree(rhs_degree)) {
                continue;
            }

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

            outer_product(ctx,
                          config,
                          subtile_position,
                          out_tile,
                          &left_panel[panel_idx * tile_size],
                          &right_panel[panel_idx * tile_size]);
        }
    }
}


// Split within the prefix block: prefix = short_lhs * remainder_size +
// remainder. Pack rhs panels by remainder, then visit output rows in order.
template <typename Ctx,
          unsigned TilePower,
          typename LhsTensor,
          typename RhsTensor>
void left_small_terms(Ctx const& ctx,
                      AccumOf<Ctx>* RPP_RESTRICT out_tile,
                      AccumOf<Ctx>* RPP_RESTRICT left_panel,
                      AccumOf<Ctx>* RPP_RESTRICT right_panel,
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
    const auto middle_size = pos.middle_size;
    const auto degree = pos.middle_degree + 2 * config.tile_letters;
    const auto min_degree = std::max<Degree>(
        1, std::max(lhs.min_degree(), degree - rhs.max_degree()));
    const auto max_degree =
        std::min<Degree>(config.tile_letters - 1,
                         std::min(lhs.max_degree(), degree - rhs.min_degree()));
    if (min_degree > max_degree) {
        return;
    }
    const auto width = static_cast<Index>(config.width);
    auto short_size = const_power(width, min_degree);
    for (Degree lhs_degree = min_degree; lhs_degree <= max_degree;
         ++lhs_degree, short_size *= width) {
        const auto rhs_degree = degree - lhs_degree;
        const auto remainder_size = config.tile_size / short_size;
        stride_read_panel<Ctx, TilePower>(
            ctx, left_panel, lhs.degree_view(lhs_degree), short_size, 1, 0);
        const auto right = rhs.degree_view(rhs_degree);
        assert(remainder_size * tile_size <=
               static_cast<Index>(TileConfig<Ctx, TilePower>::panel_extent));
        // Keep every remainder panel so output rows can be updated in order.
        const auto source_stride = middle_size * config.tile_size;
        auto source_offset = pos.middle_word * config.tile_size;
        for (Index remainder = 0; remainder < remainder_size; ++remainder) {
            stride_read_panel<Ctx, TilePower>(ctx,
                                              right_panel +
                                                  remainder * tile_size,
                                              right,
                                              pos.suffix_subtile_size,
                                              1,
                                              source_offset);
            source_offset += source_stride;
        }
        // Panel tails are initialized scratch; only live columns are written
        // back. Fixed lane groups avoid a separate scalar tail.
        for (Index i = 0; i < short_size; ++i) {
            const auto value = left_panel[i];
            auto* row = out_tile + i * remainder_size * tile_size;
            for (Index remainder = 0; remainder < remainder_size; ++remainder) {
                const auto* panel = right_panel + remainder * tile_size;
                left_multiply_range_overrun(ctx,
                                            config,
                                            row,
                                            value,
                                            panel,
                                            pos.suffix_subtile_size,
                                            Index{1});
                row += tile_size;
            }
        }
    }
}

// Split within the suffix block: suffix = remainder * short_size + short_rhs.
// Pack lhs remainders contiguously per row, then visit output columns in order.
template <typename Ctx,
          unsigned TilePower,
          typename LhsTensor,
          typename RhsTensor>
void right_small_terms(Ctx const& ctx,
                       AccumOf<Ctx>* RPP_RESTRICT out_tile,
                       AccumOf<Ctx>* RPP_RESTRICT left_panel,
                       AccumOf<Ctx>* RPP_RESTRICT right_panel,
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
    const auto middle_size = pos.middle_size;
    const auto degree = pos.middle_degree + 2 * config.tile_letters;
    const auto min_degree = std::max<Degree>(
        1, std::max(rhs.min_degree(), degree - lhs.max_degree()));
    const auto max_degree =
        std::min<Degree>(config.tile_letters - 1,
                         std::min(rhs.max_degree(), degree - lhs.min_degree()));
    if (min_degree > max_degree) {
        return;
    }
    const auto width = static_cast<Index>(config.width);
    auto short_size = const_power(width, min_degree);
    for (Degree rhs_degree = min_degree; rhs_degree <= max_degree;
         ++rhs_degree, short_size *= width) {
        const auto lhs_degree = degree - rhs_degree;
        const auto remainder_size = config.tile_size / short_size;
        stride_read_panel<Ctx, TilePower>(
            ctx, right_panel, rhs.degree_view(rhs_degree), short_size, 1, 0);
        const auto left = lhs.degree_view(lhs_degree);
        assert(pos.prefix_subtile_size * remainder_size <=
               static_cast<Index>(TileConfig<Ctx, TilePower>::panel_extent));
        // Read each source run contiguously, retaining all remainders per row.
        const auto source_stride = middle_size * remainder_size;
        auto source_offset = pos.middle_word * remainder_size;
        for (Index i = 0; i < pos.prefix_subtile_size; ++i) {
            stride_read_panel<Ctx, TilePower>(ctx,
                                              left_panel + i * remainder_size,
                                              left,
                                              remainder_size,
                                              1,
                                              source_offset);
            source_offset += source_stride;
        }
        for (Index i = 0; i < pos.prefix_subtile_size; ++i) {
            auto* row = out_tile + i * tile_size;
            const auto* panel = left_panel + i * remainder_size;
            outer_product(ctx,
                          config,
                          row,
                          panel,
                          remainder_size,
                          right_panel,
                          short_size,
                          short_size);
        }
    }
}

// The two scalar contributions lhs[0] * rhs[d] and lhs[d] * rhs[0].
// Each source row is contiguous; no word splitting is needed.
template <typename Ctx,
          unsigned TilePower,
          typename LhsTensor,
          typename RhsTensor>
void scalar_terms(Ctx const& ctx,
                  AccumOf<Ctx>* RPP_RESTRICT out_tile,
                  LhsTensor const& lhs,
                  RhsTensor const& rhs,
                  TileConfig<Ctx, TilePower> const& config,
                  SubtilePosition<Ctx> const& pos) {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    constexpr Index tile_size = TileConfig<Ctx, TilePower>::subtile_size;
    const auto degree = pos.middle_degree + 2 * config.tile_letters;
    const auto middle_size = pos.middle_size;
    const auto row_stride = middle_size * config.tile_size;
    const auto offset =
        (pos.prefix_subtile_offset * middle_size + pos.middle_word) *
            config.tile_size +
        pos.suffix_subtile_offset;
    if (lhs.has_degree(0) && rhs.has_degree(degree) && !is_zero(lhs[0])) {
        const auto scalar = static_cast<Accum>(lhs[0]);
        const auto right = rhs.degree_view(degree);
        for (Index i = 0; i < pos.prefix_subtile_size; ++i) {
            left_multiply_range(ctx,
                                config,
                                assume_aligned<16>(out_tile) + i * tile_size,
                                scalar,
                                raw_pointer_cast(right.begin()) +
                                    i * row_stride + offset,
                                pos.suffix_subtile_size,
                                Index{1});
        }
    }
    if (lhs.has_degree(degree) && rhs.has_degree(0) && !is_zero(rhs[0])) {
        const auto left = lhs.degree_view(degree);
        const auto scalar = static_cast<Accum>(rhs[0]);

        for (Index i = 0; i < pos.prefix_subtile_size; ++i) {
            right_multiply_range(ctx,
                                 config,
                                 assume_aligned<16>(out_tile) + i * tile_size,
                                 raw_pointer_cast(left.begin()) +
                                     i * row_stride + offset,
                                 pos.suffix_subtile_size,
                                 scalar,
                                 Index{1});
        }
    }
}

// Keep scalar-right products contiguous so the compiler can vectorize across
// the left level rather than an inner loop of length one.
template <typename Ctx,
          unsigned TilePower,
          typename LhsTensor,
          typename RhsTensor>
void contiguous_level_terms(Ctx const& ctx,
                            TileConfig<Ctx, TilePower> const& config,
                            AccumOf<Ctx>* RPP_RESTRICT level,
                            DegreeOf<Ctx> degree,
                            LhsTensor const& lhs,
                            RhsTensor const& rhs) {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    const auto rhs_min =
        std::max<Degree>(rhs.min_degree(), degree - lhs.max_degree());
    const auto rhs_max =
        std::min<Degree>(rhs.max_degree(), degree - lhs.min_degree());
    for (Degree rhs_degree = rhs_min; rhs_degree <= rhs_max; ++rhs_degree) {
        const auto left = lhs.degree_view(degree - rhs_degree);
        const auto right = rhs.degree_view(rhs_degree);
        const auto left_size = static_cast<Index>(left.size());
        const auto right_size = static_cast<Index>(right.size());
        if (rhs_degree == 0) {
            right_multiply_range(ctx,
                                 config,
                                 level,
                                 raw_pointer_cast(left.begin()),
                                 left_size,
                                 static_cast<Accum>(right[0]),
                                 Index{1});
        }
        else {
            outer_product(ctx,
                          config,
                          level,
                          raw_pointer_cast(left.begin()),
                          left_size,
                          raw_pointer_cast(right.begin()),
                          right_size,
                          right_size);
        }
    }
}

// Accumulate the remaining tensor prefix in one tile, then write it once.
// Read operand levels directly; no panel packing is needed.
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
                      AccumOf<Ctx>* RPP_RESTRICT tile,
                      Writer& write) {
    using Degree = DegreeOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    const auto low_max =
        std::min<Degree>(out.max_degree(), 2 * config.tile_letters - 1);
    if (low_max < out.min_degree()) {
        return;
    }
    DenseTensorView<Accum*, typename TensorOut::Basis> product(
        tile, out.basis(), out.min_degree(), low_max);
    assert((static_cast<size_t>(product.end_index()) <=
            TileConfig<Ctx, TilePower>::tile_extent));
    std::fill(product.begin(), product.end(), Accum{0});
    for (Degree degree = low_max; degree >= out.min_degree(); --degree) {
        contiguous_level_terms(
            ctx, config, product.degree_view(degree).begin(), degree, lhs, rhs);
    }
    write(ctx, out, product);
}

// With single-letter tiles, degrees >= 2 belong to the main tiled loop.
// Only the scalar-vector products at degree 1 and the scalar product remain.
template <typename Ctx,
          unsigned TilePower,
          typename TensorOut,
          typename LhsTensor,
          typename RhsTensor,
          typename Writer>
RPP_NOINLINE void
subtiled_low_degree_terms(Ctx const& ctx,
                          TileConfig<Ctx, TilePower> const& config,
                          TensorOut& out,
                          LhsTensor const& lhs,
                          RhsTensor const& rhs,
                          AccumOf<Ctx>* RPP_RESTRICT tile,
                          Writer& write) {
    using Index = IndexOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    assert(config.tile_letters == 1);
    if (out.has_degree(1)) {
        const auto width = static_cast<Index>(config.width);
        constexpr auto capacity =
            static_cast<Index>(TileConfig<Ctx, TilePower>::tile_extent);
        for (Index offset = 0; offset < width; offset += capacity) {
            const auto count = std::min(capacity, width - offset);
            std::fill_n(tile, count, Accum{0});
            if (lhs.has_degree(0) && rhs.has_degree(1)) {
                const auto scalar = static_cast<Accum>(lhs[0]);
                const auto right = rhs.degree_view(1);
                left_multiply_range(ctx,
                                    config,
                                    tile,
                                    scalar,
                                    raw_pointer_cast(right.begin()) + offset,
                                    count,
                                    Index{1});
            }
            if (lhs.has_degree(1) && rhs.has_degree(0)) {
                const auto left = lhs.degree_view(1);
                const auto scalar = static_cast<Accum>(rhs[0]);
                right_multiply_range(ctx,
                                     config,
                                     tile,
                                     raw_pointer_cast(left.begin()) + offset,
                                     count,
                                     scalar,
                                     Index{1});
            }
            write(ctx, out, tile, 1, offset, count);
        }
    }
    if (out.has_degree(0)) {
        tile[0] = Accum{0};
        if (lhs.has_degree(0) && rhs.has_degree(0)) {
            tile[0] = static_cast<Accum>(lhs[0]) * static_cast<Accum>(rhs[0]);
        }
        write(ctx, out, tile, 0, Index{0}, Index{1});
    }
}


// Width one is a scalar convolution across degrees. Complete each block
// before writing it, and visit blocks from high to low for in-place updates.
template <typename Ctx,
          unsigned TilePower,
          typename TensorOut,
          typename LhsTensor,
          typename RhsTensor,
          typename Writer>
RPP_NOINLINE void width_one_terms(Ctx const& ctx,
                                  TileConfig<Ctx, TilePower> const& config,
                                  TensorOut& out,
                                  LhsTensor const& lhs,
                                  RhsTensor const& rhs,
                                  AccumOf<Ctx>* RPP_RESTRICT tile,
                                  Writer& write) {
    using Degree = DegreeOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    constexpr auto capacity =
        static_cast<Degree>(TileConfig<Ctx, TilePower>::tile_extent);
    for (Degree last = out.max_degree(); last >= out.min_degree();) {
        const auto first =
            std::max<Degree>(out.min_degree(), last - capacity + 1);
        const auto count = last - first + 1;
        std::fill_n(tile, count, Accum{0});
        const auto lhs_min =
            std::max<Degree>(lhs.min_degree(), first - rhs.max_degree());
        const auto lhs_max =
            std::min<Degree>(lhs.max_degree(), last - rhs.min_degree());
        for (Degree ld = lhs_min; ld <= lhs_max; ++ld) {
            const auto value = static_cast<Accum>(lhs[ld]);
            const auto rhs_min = std::max<Degree>(rhs.min_degree(), first - ld);
            const auto rhs_max = std::min<Degree>(rhs.max_degree(), last - ld);
            if (rhs_min <= rhs_max) {
                left_multiply_range(
                    ctx,
                    config,
                    tile + ld + rhs_min - first,
                    value,
                    raw_pointer_cast(rhs.data()) + rhs_min,
                    static_cast<IndexOf<Ctx>>(rhs_max - rhs_min + 1),
                    IndexOf<Ctx>{1});
            }
        }
        write(ctx, out, tile, first, count);
        last = first - 1;
    }
}

// Levels with a middle block use panel traversal and middle-word split state.
// Single-letter blocks wider than a physical tile also need this for subtiling.
// Keep this boundary out of line so its split-state frame stays on this path.
template <typename Ctx,
          unsigned TilePower,
          typename TensorOut,
          typename LhsTensor,
          typename RhsTensor,
          typename Writer>
RPP_NOINLINE void tiled_terms(Ctx const& ctx,
                              TileConfig<Ctx, TilePower> const& config,
                              TensorOut& out,
                              LhsTensor const& lhs,
                              RhsTensor const& rhs,
                              DegreeOf<Ctx> first_degree,
                              Writer& write) {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    constexpr Index tile_size = TileConfig<Ctx, TilePower>::subtile_size;
    const auto width = static_cast<Index>(config.width);
    const auto tile_letters = config.tile_letters;
    const auto side = config.tile_size;
    auto& storage = TileStorage<Ctx, TilePower>::get(ctx);
    auto* tile = storage.tile.data();
    auto* left_panel = storage.left_panel.data();
    auto* right_panel = storage.right_panel.data();

    const auto min_middle = first_degree - 2 * tile_letters;
    const auto max_middle = out.max_degree() - 2 * tile_letters;

    for (Degree middle_degree = max_middle; middle_degree >= min_middle;
         --middle_degree) {
        const auto middle_size =
            static_cast<Index>(out.basis().size_of_degree(middle_degree));
        MiddleWordSplits<Ctx> splits(width, middle_degree);
        for (Index middle_word = 0; middle_word < middle_size; ++middle_word) {
            for (Index prefix = 0; prefix < side; prefix += tile_size) {
                for (Index suffix = 0; suffix < side; suffix += tile_size) {
                    SubtilePosition<Ctx> pos{
                        middle_word,
                        static_cast<Degree>(middle_degree),
                        middle_size,
                        prefix,
                        suffix,
                        std::min(tile_size, side - prefix),
                        std::min(tile_size, side - suffix)};
                    clear_tile<Ctx, TilePower>(tile, pos);
                    scalar_terms<Ctx, TilePower>(
                        ctx, tile, lhs, rhs, config, pos);
                    left_small_terms<Ctx, TilePower>(ctx,
                                                     tile,
                                                     left_panel,
                                                     right_panel,
                                                     lhs,
                                                     rhs,
                                                     config,
                                                     pos);
                    mid_terms<Ctx, TilePower>(ctx,
                                              tile,
                                              left_panel,
                                              right_panel,
                                              lhs,
                                              rhs,
                                              config,
                                              pos,
                                              splits);
                    right_small_terms<Ctx, TilePower>(ctx,
                                                      tile,
                                                      left_panel,
                                                      right_panel,
                                                      lhs,
                                                      rhs,
                                                      config,
                                                      pos);
                    write(ctx, out, tile, config, pos);
                }
            }
            splits.advance();
        }
    }
}

// Finish the level spanning just the prefix/suffix blocks, followed by the
// low-degree prefix. Keep the same descending write order for aliased operands.
template <typename Ctx,
          unsigned TilePower,
          typename TensorOut,
          typename LhsTensor,
          typename RhsTensor,
          typename Writer>
void contiguous_terms(Ctx const& ctx,
                      TileConfig<Ctx, TilePower> const& config,
                      TensorOut& out,
                      LhsTensor const& lhs,
                      RhsTensor const& rhs,
                      DegreeOf<Ctx> last_degree,
                      Writer& write) {
    using Index = IndexOf<Ctx>;
    using Degree = DegreeOf<Ctx>;
    using Accum = AccumOf<Ctx>;
    constexpr size_t tile_extent = TileConfig<Ctx, TilePower>::tile_extent;
    auto* tile = TileStorage<Ctx, TilePower>::get(ctx).tile.data();
    const auto first_degree =
        std::max<Degree>(out.min_degree(), 2 * config.tile_letters);
    for (Degree degree = last_degree; degree >= first_degree; --degree) {
        const auto level_size = out.basis().size_of_degree(degree);
        std::fill_n(tile, level_size, Accum{0});
        contiguous_level_terms(ctx, config, tile, degree, lhs, rhs);
        write(ctx, out, tile, degree, Index{0}, level_size);
    }

    // At width == capacity, degree one fits alone but its scalar prefix does
    // not.
    if (config.width >= static_cast<Index>(tile_extent)) {
        subtiled_low_degree_terms(ctx, config, out, lhs, rhs, tile, write);
    }
    else {
        low_degree_terms(ctx, config, out, lhs, rhs, tile, write);
    }
}

} // namespace detail


// Tile dimensions and the implementation specialization are selected by the
// caller. Writers receive a tile pointer and subtile position, a tensor view
// for the low-degree prefix, or a pointer and count for contiguous chunks.
// Output degrees are visited in descending order so a writer can also update
// either operand in place. Output and operands must use the same tensor basis.
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
    constexpr Index tile_size = TileConfig<Ctx, TilePower>::subtile_size;
    const auto width = static_cast<Index>(config.width);
    assert(width > 0 && config.tile_letters > 0 && config.tile_size > 0);
    assert(config.tile_size <= tile_size || config.tile_letters == 1);
    assert(out.basis().width == config.width);
    assert(config.first_tiled_degree > 0);

    if (width == 1) {
        auto* tile = TileStorage<Ctx, TilePower>::get(ctx).tile.data();
        detail::width_one_terms<Ctx, TilePower>(
            ctx, config, out, lhs, rhs, tile, write);
        return;
    }

    const auto first_tiled_degree =
        std::max<Degree>(out.min_degree(), config.first_tiled_degree);
    if (first_tiled_degree <= out.max_degree()) {
        detail::tiled_terms(
            ctx, config, out, lhs, rhs, first_tiled_degree, write);
    }
    detail::contiguous_terms(
        ctx,
        config,
        out,
        lhs,
        rhs,
        std::min<Degree>(out.max_degree(), first_tiled_degree - 1),
        write);
}


} // namespace rpp::cpu::single_thread::ft_multiply

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_DETAIL_FT_MULTIPLY_HPP
