#ifndef RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_ADJOINT_TEST_HELPER_CUH
#define RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_ADJOINT_TEST_HELPER_CUH

#include <algorithm>
#include <cmath>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <rpp/gpu/block/strategy.hpp>

#include "gpu_block_test_helper.cuh"

namespace rpp::tests {

template <typename Scalar_, typename Accum_>
struct TypedScalarAccumConfig {
    using Scalar = Scalar_;
    using Accum = Accum_;
};

template <typename Config>
struct TypedScalarAccumTypeName;

template <>
struct TypedScalarAccumTypeName<TypedScalarAccumConfig<float, float>> {
    static std::string value() { return "FloatFloat"; }
};

template <>
struct TypedScalarAccumTypeName<TypedScalarAccumConfig<double, double>> {
    static std::string value() { return "DoubleDouble"; }
};

template <>
struct TypedScalarAccumTypeName<TypedScalarAccumConfig<__half, __half>> {
    static std::string value() { return "Fp16Fp16"; }
};

template <>
struct TypedScalarAccumTypeName<TypedScalarAccumConfig<__half, float>> {
    static std::string value() { return "Fp16Float"; }
};

template <>
struct TypedScalarAccumTypeName<
    TypedScalarAccumConfig<__nv_bfloat16, __nv_bfloat16>> {
    static std::string value() { return "Bf16Bf16"; }
};

template <>
struct TypedScalarAccumTypeName<TypedScalarAccumConfig<__nv_bfloat16, float>> {
    static std::string value() { return "Bf16Float"; }
};

struct TypedScalarAccumNameGenerator {
    template <typename Config>
    static std::string GetName(int) {
        return TypedScalarAccumTypeName<Config>::value();
    }
};

template <typename Scalar>
Scalar cast_scalar(float value) {
    return static_cast<Scalar>(value);
}

template <>
inline __half cast_scalar<__half>(float value) {
    return __float2half(value);
}

template <>
inline __nv_bfloat16 cast_scalar<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename Scalar>
double scalar_to_double(Scalar value) {
    return static_cast<double>(value);
}

template <>
inline double scalar_to_double(__half value) {
    return static_cast<double>(__half2float(value));
}

template <>
inline double scalar_to_double(__nv_bfloat16 value) {
    return static_cast<double>(__bfloat162float(value));
}

template <typename Scalar, typename Accum>
struct NumericTolerance {
    static constexpr double value =
        std::is_same_v<Scalar, double> ? 1e-10 : 1.5e-4;
};

template <>
struct NumericTolerance<__half, __half> {
    static constexpr double value = 2e-2;
};

template <>
struct NumericTolerance<__half, float> {
    static constexpr double value = 2e-3;
};

template <>
struct NumericTolerance<__nv_bfloat16, __nv_bfloat16> {
    static constexpr double value = 4e-2;
};

template <>
struct NumericTolerance<__nv_bfloat16, float> {
    static constexpr double value = 5e-3;
};

template <typename Config>
class TypedGpuAdjointTestBase : public testing::Test {
protected:
    using Helper = GpuBlockTestHelper;
    using Scalar = typename Config::Scalar;
    using Accum = typename Config::Accum;
    using Basis = typename Helper::Basis;
    using Degree = typename Helper::Degree;
    using Index = typename Helper::Index;
    using HostVector = typename Helper::template HostVector<Scalar>;
    using DeviceVector = typename Helper::template DeviceVector<Scalar>;
    using PairingDeviceVector = typename Helper::template DeviceVector<Accum>;
    using GpuStrategy = rpp::gpu::strategies::BlockStrategy<
        Accum, Helper::block_size, 256, typename Helper::GpuArchitecture>;

    [[nodiscard]] static HostVector make_batch(unsigned seed, Basis const& basis) {
        HostVector result(
            static_cast<std::size_t>(Helper::tensor_count * basis.size()));
        for (std::size_t i = 0; i < result.size(); ++i) {
            auto const centered =
                static_cast<int>((seed * 17u + static_cast<unsigned>(i) * 5u) %
                                     9u) -
                4;
            auto const magnitude =
                static_cast<float>(centered) * 0.03125f +
                static_cast<float>((i + seed) % 3u) * 0.0078125f;
            result[i] = cast_scalar<Scalar>(magnitude);
        }
        return result;
    }

    [[nodiscard]] static HostVector make_zero_batch(Basis const& basis) {
        return HostVector(
            static_cast<std::size_t>(Helper::tensor_count * basis.size()),
            cast_scalar<Scalar>(0.0f));
    }

    [[nodiscard]] static HostVector make_identity_operator(Basis const& basis) {
        auto result = make_zero_batch(basis);
        result[0] = cast_scalar<Scalar>(1.0f);
        return result;
    }

    static void expect_scalar_near(Accum actual, Accum expected) {
        auto const actual_d = scalar_to_double(actual);
        auto const expected_d = scalar_to_double(expected);
        auto const scale =
            std::max({1.0, std::abs(actual_d), std::abs(expected_d)});
        auto const tolerance = NumericTolerance<Scalar, Accum>::value * scale;
        EXPECT_NEAR(actual_d, expected_d, tolerance);
    }

    static void expect_tensor_near(HostVector const& actual,
                                   HostVector const& expected) {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            auto const actual_d = scalar_to_double(actual[i]);
            auto const expected_d = scalar_to_double(expected[i]);
            auto const scale =
                std::max({1.0, std::abs(actual_d), std::abs(expected_d)});
            auto const tolerance = NumericTolerance<Scalar, Accum>::value * scale;
            EXPECT_NEAR(actual_d, expected_d, tolerance)
                << "at coefficient " << i;
        }
    }
};

using TypedGpuAdjointTestTypes = testing::Types<
    TypedScalarAccumConfig<float, float>,
    TypedScalarAccumConfig<double, double>,
    TypedScalarAccumConfig<__half, __half>,
    TypedScalarAccumConfig<__half, float>,
    TypedScalarAccumConfig<__nv_bfloat16, __nv_bfloat16>,
    TypedScalarAccumConfig<__nv_bfloat16, float>>;

} // namespace rpp::tests

#endif // RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_ADJOINT_TEST_HELPER_CUH
