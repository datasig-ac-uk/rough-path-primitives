#include <array>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations.hpp>

#include "cpu_typed_ft_ops_test_helper.hpp"
#include "legacy_ft_multiply_test_helper.hpp"

namespace {
namespace ft = rpp::cpu::single_thread::ft_multiply;

struct Shape {
    int width;
    int depth;
    int tile_letters;
    int first_tiled_degree;
};

// Cover whole blobs, separate contiguous levels, full/partial tiles, both
// sides of the short-run threshold, prefix carries, and subtiling boundaries.
template <unsigned TilePower>
auto shapes() {
    if constexpr (TilePower == 5) {
        return std::array{Shape{1, 8, 1, 3},
                          Shape{2, 2, 5, 11}, // Tile letters may exceed depth.
                          Shape{2, 6, 5, 11},
                          Shape{5, 3, 2, 5},
                          Shape{10, 3, 1, 4},
                          Shape{2, 12, 5, 11},
                          Shape{2, 15, 5, 11},
                          Shape{3, 8, 3, 7},
                          Shape{3, 10, 3, 7},
                          Shape{4, 6, 2, 6},
                          Shape{32, 3, 1, 3},
                          Shape{33, 2, 0, 2},
                          Shape{1024, 1, 0, 2},
                          Shape{1025, 1, 0, 2}};
    }
    else {
        static_assert(TilePower == 2);
        // A small physical tile makes exact-capacity and partial right-block
        // subtiling affordable, and also exercises reduced row alignment.
        return std::array{Shape{1, 16, 1, 3},
                          Shape{3, 3, 1, 3},
                          Shape{4, 3, 1, 3},
                          Shape{5, 3, 0, 2},
                          Shape{5, 4, 0, 2}, // Multiple middle-panel batches.
                          Shape{16, 2, 0, 2},
                          Shape{17, 2, 0, 2}};
    }
}

enum class WriteMode { Product, AliasLeft, AliasRight, Add, InplaceAdd };
enum class Driver { Right, Symmetric, PublicOperations };
enum class ViewMode {
    Full,
    Restricted,
    PositiveDegrees,
    EmptyProduct,
    ZeroScalars,
    TopLevel
};

template <typename Config>
class FreeTensorTilingConfigurations
    : public rpp::tests::TypedCpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Strategy;
    using Context = typename Strategy::Context;
    using Degree = typename Base::Degree;
    using Range = typename Base::DegreeRange;

    template <unsigned TilePower>
    static void check_configuration() {
        for (auto shape : shapes<TilePower>()) {
            SCOPED_TRACE(::testing::Message()
                         << "power=" << TilePower << " width=" << shape.width
                         << " depth=" << shape.depth);
            const typename Base::BasisData data(shape.width, shape.depth);
            rpp::tests::CpuScratchContext<Context, TilePower> ctx;
            ft::Operation<Strategy, TilePower> op(
                ctx, rpp::basis::make_basis_pack(data.basis));
            EXPECT_EQ(op.config().tile_letters, shape.tile_letters);
            EXPECT_EQ(op.config().first_tiled_degree, shape.first_tiled_degree);
            EXPECT_EQ(op.config().tile_size,
                      rpp::const_power(typename Base::Index{shape.width},
                                       shape.tile_letters));
        }
    }

    template <unsigned TilePower>
    static void check_products(ViewMode views) {
        for (auto shape : shapes<TilePower>()) {
            const typename Base::BasisData data(shape.width, shape.depth);
            const auto& basis = data.basis;
            const auto full = Base::full_range(basis);
            auto out_range = full;
            auto lhs_range = full;
            auto rhs_range = full;
            auto add_range = full;
            switch (views) {
            case ViewMode::Full:
            case ViewMode::ZeroScalars:
                break;
            case ViewMode::Restricted:
                out_range = lhs_range = {1, shape.depth};
                rhs_range = add_range = {0, shape.depth - 1};
                break;
            case ViewMode::PositiveDegrees:
                lhs_range = rhs_range = {1, shape.depth};
                break;
            case ViewMode::EmptyProduct:
                lhs_range = rhs_range = {shape.depth, shape.depth};
                break;
            case ViewMode::TopLevel:
                out_range = {shape.depth, shape.depth};
                break;
            }
            for (auto driver :
                 {Driver::Right, Driver::Symmetric, Driver::PublicOperations}) {
                if constexpr (TilePower != ft::default_tile_power) {
                    if (driver == Driver::PublicOperations) {
                        continue;
                    }
                }
                for (auto mode : {WriteMode::Product,
                                  WriteMode::AliasLeft,
                                  WriteMode::AliasRight,
                                  WriteMode::Add,
                                  WriteMode::InplaceAdd}) {
                    SCOPED_TRACE(
                        ::testing::Message()
                        << "power=" << TilePower << " width=" << shape.width
                        << " depth=" << shape.depth << " driver=" << int(driver)
                        << " mode=" << int(mode) << " views=" << int(views));
                    auto lhs = Base::make_tensor(2, basis);
                    auto rhs = Base::make_tensor(3, basis);
                    if (views == ViewMode::ZeroScalars) {
                        lhs[0] = Base::cast_scalar(0.0f);
                        rhs[0] = Base::cast_scalar(0.0f);
                    }
                    auto out = Base::make_tensor(1, basis);
                    auto add = Base::make_tensor(4, basis);
                    auto& actual = mode == WriteMode::AliasLeft ? lhs
                        : mode == WriteMode::AliasRight         ? rhs
                                                                : out;
                    auto expected = actual;
                    auto target =
                        Base::mutable_tensor_view(actual, basis, out_range);
                    auto left = Base::const_tensor_view(lhs, basis, lhs_range);
                    auto right = Base::const_tensor_view(rhs, basis, rhs_range);
                    // The expected output has separate storage, including for
                    // aliased products, so the legacy reference sees old
                    // inputs.
                    auto expected_view =
                        Base::mutable_tensor_view(expected, basis, out_range);
                    auto add_view = Base::const_tensor_view(
                        mode == WriteMode::InplaceAdd ? actual : add,
                        basis,
                        mode == WriteMode::Add ? add_range : full);
                    const Accum alpha =
                        mode == WriteMode::Add || mode == WriteMode::InplaceAdd
                        ? Accum{0.25}
                        : Accum{0};
                    const Accum beta{-0.75};
                    rpp::tests::legacy::fma(
                        expected_view, add_view, left, right, alpha, beta);

                    rpp::tests::CpuScratchContext<Context, TilePower> ctx;
                    ft::Operation<Strategy, TilePower> op(
                        ctx, rpp::basis::make_basis_pack(basis));
                    auto run = [&](auto writer) {
                        if (driver == Driver::Right && shape.width > 1) {
                            ft::detail::multiply_oneside(ctx.context(),
                                                         op.config(),
                                                         target,
                                                         left,
                                                         right,
                                                         writer);
                        }
                        else if (driver == Driver::Symmetric &&
                                 shape.width > 1) {
                            ft::detail::multiply_twoside(ctx.context(),
                                                         op.config(),
                                                         target,
                                                         left,
                                                         right,
                                                         writer);
                        }
                        else if (driver != Driver::PublicOperations) {
                            // Width one uses the shared convolution path.
                            ft::multiply(ctx.context(),
                                         op.config(),
                                         target,
                                         left,
                                         right,
                                         writer);
                        }
                        else if constexpr (TilePower ==
                                           ft::default_tile_power) {
                            // Check the public selection independently of the
                            // explicit right and symmetric driver checks.
                            const auto pack =
                                rpp::basis::make_basis_pack(basis);
                            if (mode == WriteMode::Add) {
                                rpp::ops::FTFma<Strategy>{ctx, pack}(ctx,
                                                                     target,
                                                                     add_view,
                                                                     left,
                                                                     right,
                                                                     alpha,
                                                                     beta);
                            }
                            else if (mode == WriteMode::InplaceAdd) {
                                rpp::ops::FTInplaceFma231<Strategy>{ctx, pack}(
                                    ctx, target, left, right, alpha, beta);
                            }
                            else {
                                rpp::ops::FTMul<Strategy>{ctx, pack}(
                                    ctx, target, left, right, beta);
                            }
                        }
                    };
                    if (mode == WriteMode::InplaceAdd) {
                        run(ft::InplaceFmaTileWriter<
                            Context,
                            rpp::ops::FTInplaceFMAType::AEqualsBCPlusA>{alpha,
                                                                        beta});
                    }
                    else if (mode == WriteMode::Add) {
                        run(ft::ScaledTileAddWriter<Context,
                                                    decltype(add_view)>{
                            add_view, alpha, beta});
                    }
                    else {
                        run(ft::ScaledTileWriter<Context>{beta});
                    }
                    Base::expect_tensor_near(actual, expected);
                }
            }
        }
    }

    static void check_old_inplace() {
        for (auto shape : shapes<5>()) {
            SCOPED_TRACE(::testing::Message() << "width=" << shape.width
                                              << " depth=" << shape.depth);
            const typename Base::BasisData data(shape.width, shape.depth);
            const auto full = Base::full_range(data.basis);
            auto actual = Base::make_tensor(2, data.basis);
            auto expected = actual;
            const auto rhs = Base::make_tensor(3, data.basis);
            auto target = Base::mutable_tensor_view(actual, data.basis, full);
            auto expected_view =
                Base::mutable_tensor_view(expected, data.basis, full);
            auto right = Base::const_tensor_view(rhs, data.basis, full);
            const Accum beta{1.5};
            rpp::tests::legacy::inplace_mul(expected_view, right, beta);
            const auto ctx = Base::make_context();
            rpp::ops::FTInplaceMul<Strategy>{
                ctx, rpp::basis::make_basis_pack(data.basis)}(
                ctx, target, right, beta);
            Base::expect_tensor_near(actual, expected);
        }
    }
};

using Configs = testing::Types<
    rpp::tests::TypedScalarAccumConfig<float, float>,
    rpp::tests::TypedScalarAccumConfig<double, double>,
    rpp::tests::TypedScalarAccumConfig<float, double>,
    rpp::tests::TypedScalarAccumConfig<rpp::tests::ExplicitScalar, double>>;
TYPED_TEST_SUITE(FreeTensorTilingConfigurations, Configs);

TYPED_TEST(FreeTensorTilingConfigurations,
           ResolvesDefaultAndSmallTileBoundaries) {
    TestFixture::template check_configuration<5>();
    TestFixture::template check_configuration<2>();
}

TYPED_TEST(FreeTensorTilingConfigurations,
           BothStrategiesMatchLegacyOnFullViews) {
    TestFixture::template check_products<5>(ViewMode::Full);
    TestFixture::template check_products<2>(ViewMode::Full);
}

TYPED_TEST(FreeTensorTilingConfigurations,
           BothStrategiesMatchLegacyOnRestrictedViews) {
    TestFixture::template check_products<5>(ViewMode::Restricted);
    TestFixture::template check_products<2>(ViewMode::Restricted);
}

TYPED_TEST(FreeTensorTilingConfigurations, MissingScalarAndEmptyProductRanges) {
    for (auto views : {ViewMode::PositiveDegrees,
                       ViewMode::EmptyProduct,
                       ViewMode::ZeroScalars,
                       ViewMode::TopLevel}) {
        TestFixture::template check_products<5>(views);
        TestFixture::template check_products<2>(views);
    }
}

TYPED_TEST(FreeTensorTilingConfigurations, InplaceMultiplicationMatchesLegacy) {
    TestFixture::check_old_inplace();
}
} // namespace
