#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/basic/left_hs_adj_lmul.hpp>
#include <rpp/gpu/block/operations/basic/left_hs_adj_rmul.hpp>
#include <rpp/gpu/block/operations/basic/left_hs_mul.hpp>

#include "../../../half_shuffle_adjoint_test_helper.hpp"
#include "gpu_typed_ft_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockHalfShuffleAdjointTests
    : public rpp::tests::TypedGpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Degree;
    using typename Base::DegreeRange;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::Index;
    using typename Base::Scalar;

    template <bool FixedLeft>
    static std::vector<Scalar> run_adjoint(Basis const& basis,
                                           std::vector<Scalar> const& initial,
                                           std::vector<Scalar> const& op,
                                           std::vector<Scalar> const& arg,
                                           DegreeRange out_range,
                                           DegreeRange op_range,
                                           DegreeRange arg_range,
                                           Accum beta) {
        DeviceVector device_out(initial), device_op(op), device_arg(arg);
        auto const strategy = GpuStrategy{Helper::block_size};
        rpp::gpu::DeviceLaunchConfig launch;
        launch.stream = nullptr;
        auto const out =
            rpp::make_tensor_batch(Helper::device_data(device_out) + 1,
                                   basis.size(),
                                   out_range.min,
                                   out_range.max);
        auto const op_batch =
            rpp::make_tensor_batch(Helper::device_data(device_op),
                                   basis.size(),
                                   op_range.min,
                                   op_range.max);
        auto const arg_batch =
            rpp::make_tensor_batch(Helper::device_data(device_arg),
                                   basis.size(),
                                   arg_range.min,
                                   arg_range.max);
        auto const err = [&] {
            if constexpr (FixedLeft) {
                return rpp::ops::left_hs_adj_lmul(strategy,
                                                  launch,
                                                  out,
                                                  op_batch,
                                                  arg_batch,
                                                  basis,
                                                  Index{1},
                                                  beta);
            }
            else {
                // Match the CPU signature: cotangent, then fixed integrand.
                return rpp::ops::left_hs_adj_rmul(strategy,
                                                  launch,
                                                  out,
                                                  arg_batch,
                                                  op_batch,
                                                  basis,
                                                  Index{1},
                                                  beta);
            }
        }();
        EXPECT_TRUE(static_cast<bool>(err)) << err.message();
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        auto const actual = Helper::copy_to_host(device_out);
        return {actual.begin(), actual.end()};
    }

    static std::vector<Scalar> run_forward(Basis const& basis,
                                           std::vector<Scalar> const& initial,
                                           std::vector<Scalar> const& op,
                                           std::vector<Scalar> const& arg,
                                           DegreeRange out_range,
                                           DegreeRange op_range,
                                           DegreeRange arg_range,
                                           Accum beta) {
        DeviceVector device_out(initial), device_op(op), device_arg(arg);
        auto const strategy = GpuStrategy{Helper::block_size};
        rpp::gpu::DeviceLaunchConfig launch;
        launch.stream = nullptr;
        auto const out =
            rpp::make_tensor_batch(Helper::device_data(device_out) + 1,
                                   basis.size(),
                                   out_range.min,
                                   out_range.max);
        auto const op_batch =
            rpp::make_tensor_batch(Helper::device_data(device_op),
                                   basis.size(),
                                   op_range.min,
                                   op_range.max);
        auto const arg_batch =
            rpp::make_tensor_batch(Helper::device_data(device_arg),
                                   basis.size(),
                                   arg_range.min,
                                   arg_range.max);
        auto const err = [&] {
            return rpp::ops::left_hs_mul(strategy,
                                         launch,
                                         out,
                                         op_batch,
                                         arg_batch,
                                         basis,
                                         Index{1},
                                         beta);
        }();
        EXPECT_TRUE(static_cast<bool>(err)) << err.message();
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        auto const actual = Helper::copy_to_host(device_out);
        return {actual.begin(), actual.end()};
    }

    template <bool FixedLeft>
    static void check_ranges() {
        struct BasisConfig {
            Degree width, depth;
        };
        // At 128 threads, degree-one widths 127/128/129 exercise the
        // cutoff on either side, including a level requiring multiple passes.
        BasisConfig const configs[] = {
            {1, 4}, {2, 2}, {4, 4}, {4, 0}, {127, 1}, {128, 1}, {129, 1}};
        for (auto const& config : configs) {
            SCOPED_TRACE(testing::Message() << "width=" << config.width
                                            << ", depth=" << config.depth);
            auto const basis_data =
                typename Helper::BasisData(config.width, config.depth);
            auto const& basis = basis_data.basis;
            for (auto const& ranges :
                 rpp::tests::half_shuffle_adjoint_range_cases<DegreeRange>(
                     basis.depth)) {
                SCOPED_TRACE(testing::Message()
                             << ranges.name << " out=[" << ranges.out.min << ","
                             << ranges.out.max << "]"
                             << " op=[" << ranges.op.min << "," << ranges.op.max
                             << "]"
                             << " arg=[" << ranges.arg.min << ","
                             << ranges.arg.max << "]");
                for (float beta : {0.0f, 0.5f, -1.0f}) {
                    SCOPED_TRACE(beta);
                    auto op =
                        rpp::tests::make_half_shuffle_adjoint_operator<Scalar>(
                            basis);
                    auto arg =
                        rpp::tests::make_half_shuffle_cotangent<Scalar>(basis);
                    auto const coefficients =
                        rpp::tests::reference_half_shuffle_adjoint(basis,
                                                                   op,
                                                                   arg,
                                                                   ranges.out,
                                                                   ranges.op,
                                                                   ranges.arg,
                                                                   FixedLeft,
                                                                   beta);
                    // Populate inactive input storage with NaNs to catch
                    // forbidden reads.
                    for (Degree degree = 0; degree <= basis.depth; ++degree) {
                        for (auto idx = basis.start_of_degree(degree);
                             idx < basis.end_of_degree(degree);
                             ++idx) {
                            if (degree < ranges.op.min ||
                                degree > ranges.op.max) {
                                op[idx] = static_cast<Scalar>(
                                    std::numeric_limits<float>::quiet_NaN());
                            }
                            if (degree < ranges.arg.min ||
                                degree > ranges.arg.max) {
                                arg[idx] = static_cast<Scalar>(
                                    std::numeric_limits<float>::quiet_NaN());
                            }
                        }
                    }
                    std::vector<Scalar> initial(
                        static_cast<std::size_t>(basis.size()) + 2,
                        static_cast<Scalar>(7.0f));
                    initial.back() = static_cast<Scalar>(-7.0f);
                    auto expected = initial;
                    auto const begin = basis.start_of_degree(ranges.out.min);
                    auto const end = basis.end_of_degree(ranges.out.max);
                    for (auto idx = begin; idx < end; ++idx) {
                        initial[idx + 1] = static_cast<Scalar>(
                            std::numeric_limits<float>::quiet_NaN());
                        expected[idx + 1] =
                            static_cast<Scalar>(coefficients[idx]);
                    }
                    auto const actual =
                        run_adjoint<FixedLeft>(basis,
                                               initial,
                                               op,
                                               arg,
                                               ranges.out,
                                               ranges.op,
                                               ranges.arg,
                                               static_cast<Accum>(beta));
                    RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(
                        GpuBlockHalfShuffleAdjointTests, actual, expected);
                    for (std::size_t i = 0; i < actual.size(); ++i) {
                        if (i <= static_cast<std::size_t>(begin) ||
                            i > static_cast<std::size_t>(end)) {
                            EXPECT_EQ(static_cast<double>(actual[i]),
                                      static_cast<double>(expected[i]))
                                << i;
                        }
                        else {
                            EXPECT_TRUE(
                                std::isfinite(static_cast<double>(actual[i])))
                                << i;
                            if (coefficients[i - 1] == 0.0) {
                                EXPECT_EQ(static_cast<double>(actual[i]), 0.0)
                                    << i;
                            }
                        }
                    }
                }
            }
        }
    }

    template <bool FixedLeft>
    static void check_pairing() {
        auto const basis_data = typename Helper::BasisData(3, 3);
        auto const& basis = basis_data.basis;
        auto const integrator =
            rpp::tests::make_half_shuffle_adjoint_operator<Scalar>(basis);
        auto const integrand =
            rpp::tests::make_half_shuffle_cotangent<Scalar>(basis);
        auto cotangent = integrand;
        for (std::size_t i = 0; i < cotangent.size(); ++i) {
            cotangent[i] =
                static_cast<Scalar>(float(int((i * 11) % 7) - 3) / 8.0f);
        }
        struct Ranges {
            DegreeRange integrator, integrand, cotangent;
        };
        Ranges const cases[] = {
            {{0, 3}, {0, 3}, {0, 3}},
            {{1, 2}, {0, 1}, {2, 3}},
            {{0, 0}, {0, 3}, {0, 3}},
            {{1, 1}, {0, 0}, {1, 1}},
            {{0, 3}, {0, 0}, {1, 2}},
        };
        std::vector<Scalar> initial(static_cast<std::size_t>(basis.size()) + 2,
                                    std::numeric_limits<Scalar>::quiet_NaN());
        for (auto const& ranges : cases) {
            auto const beta = Accum{-0.5};
            auto const out_range =
                FixedLeft ? ranges.integrand : ranges.integrator;
            auto const op_range =
                FixedLeft ? ranges.integrator : ranges.integrand;
            auto const& fixed_operand = FixedLeft ? integrator : integrand;
            auto const& varying_operand = FixedLeft ? integrand : integrator;
            auto const adjoint = run_adjoint<FixedLeft>(basis,
                                                        initial,
                                                        fixed_operand,
                                                        cotangent,
                                                        out_range,
                                                        op_range,
                                                        ranges.cotangent,
                                                        beta);
            auto const product = run_forward(basis,
                                             initial,
                                             integrator,
                                             integrand,
                                             ranges.cotangent,
                                             ranges.integrator,
                                             ranges.integrand,
                                             beta);
            double lhs = 0, rhs = 0;
            for (auto i = basis.start_of_degree(ranges.cotangent.min);
                 i < basis.end_of_degree(ranges.cotangent.max);
                 ++i) {
                lhs += double(cotangent[i]) * double(product[i + 1]);
            }
            for (auto i = basis.start_of_degree(out_range.min);
                 i < basis.end_of_degree(out_range.max);
                 ++i) {
                rhs += double(varying_operand[i]) * double(adjoint[i + 1]);
            }
            EXPECT_NEAR(lhs, rhs, 1e-10);
        }
    }

    template <bool FixedLeft>
    static void check_zero_inputs() {
        auto const basis_data =
            typename Helper::BasisData(Degree{4}, Degree{4});
        auto const& basis = basis_data.basis;
        DegreeRange const range{0, basis.depth};
        for (bool zero_operator : {false, true}) {
            auto op =
                rpp::tests::make_half_shuffle_adjoint_operator<Scalar>(basis);
            auto arg = rpp::tests::make_half_shuffle_cotangent<Scalar>(basis);
            auto& zero_input = zero_operator ? op : arg;
            std::fill(zero_input.begin(),
                      zero_input.end(),
                      static_cast<Scalar>(0.0f));
            std::vector<Scalar> initial(
                static_cast<std::size_t>(basis.size()) + 2,
                static_cast<Scalar>(std::numeric_limits<float>::quiet_NaN()));
            initial.front() = static_cast<Scalar>(7.0f);
            initial.back() = static_cast<Scalar>(-7.0f);
            auto const actual =
                run_adjoint<FixedLeft>(basis,
                                       initial,
                                       op,
                                       arg,
                                       range,
                                       range,
                                       range,
                                       static_cast<Accum>(1.0f));
            EXPECT_EQ(static_cast<double>(actual.front()), 7.0);
            EXPECT_EQ(static_cast<double>(actual.back()), -7.0);
            for (std::size_t i = 1; i + 1 < actual.size(); ++i) {
                EXPECT_EQ(static_cast<double>(actual[i]), 0.0) << i;
            }
        }
    }
};

TYPED_TEST_SUITE(GpuBlockHalfShuffleAdjointTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockHalfShuffleAdjointTests,
           FixedLeftAdjointMatchesTransposeAndRespectsBounds) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_ranges<true>();
}

TYPED_TEST(GpuBlockHalfShuffleAdjointTests,
           ZeroInputsOverwriteEveryOutputCoefficient) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_zero_inputs<true>();
}

TYPED_TEST(GpuBlockHalfShuffleAdjointTests, SatisfiesFixedLeftAdjointPairing) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_pairing<true>();
}

TYPED_TEST(GpuBlockHalfShuffleAdjointTests,
           FixedRightAdjointMatchesTransposeAndRespectsBounds) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_ranges<false>();
}

TYPED_TEST(GpuBlockHalfShuffleAdjointTests,
           FixedRightZeroInputsOverwriteEveryOutputCoefficient) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_zero_inputs<false>();
}

TYPED_TEST(GpuBlockHalfShuffleAdjointTests, SatisfiesFixedRightAdjointPairing) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_pairing<false>();
}

} // namespace
