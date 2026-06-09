#ifndef RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_ADJOINT_TEST_HELPER_CUH
#define RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_ADJOINT_TEST_HELPER_CUH

#include <algorithm>
#include <cmath>
#include <sstream>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>

#include <rpp/gpu/block/strategy.hpp>

#include "gpu_block_test_helper.cuh"

#define RPP_EXPECT_GPU_TYPED_SCALAR_NEAR(Fixture, actual, expected)            \
    Fixture::expect_scalar_near_impl(                                          \
        actual,                                                                 \
        expected,                                                               \
        __FILE__,                                                               \
        __LINE__,                                                               \
        ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name(), \
        ::testing::UnitTest::GetInstance()->current_test_info()->name())

#define RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(Fixture, actual, expected)            \
    Fixture::expect_tensor_near_impl(                                          \
        actual,                                                                 \
        expected,                                                               \
        __FILE__,                                                               \
        __LINE__,                                                               \
        ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name(), \
        ::testing::UnitTest::GetInstance()->current_test_info()->name())

#define RPP_EXPECT_GPU_TYPED_TENSOR_NEAR_WITH_BASE_TOLERANCE(                  \
    Fixture, actual, expected, base_tolerance)                                 \
    Fixture::expect_tensor_near_impl(                                          \
        actual,                                                                 \
        expected,                                                               \
        __FILE__,                                                               \
        __LINE__,                                                               \
        ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name(), \
        ::testing::UnitTest::GetInstance()->current_test_info()->name(),       \
        base_tolerance)

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

    struct DegreeRange {
        Degree min;
        Degree max;
    };

    [[nodiscard]] static constexpr bool contains(DegreeRange range,
                                                 Degree degree) noexcept {
        return range.min <= degree && degree <= range.max;
    }

    [[nodiscard]] static constexpr DegreeRange
    full_range(Basis const& basis) noexcept {
        return DegreeRange{Degree{0}, basis.depth};
    }

    [[nodiscard]] static constexpr DegreeRange
    overlap_range(DegreeRange lhs, DegreeRange rhs) noexcept {
        return DegreeRange{std::max(lhs.min, rhs.min), std::min(lhs.max, rhs.max)};
    }

    [[nodiscard]] static constexpr bool is_empty(DegreeRange range) noexcept {
        return range.max < range.min;
    }

    [[nodiscard]] static HostVector make_batch(unsigned seed, Basis const& basis) {
        return make_batch(seed, basis, 1.0f);
    }

    [[nodiscard]] static HostVector
    make_batch(unsigned seed, Basis const& basis, float scale) {
        auto const base = Helper::make_batch(seed, basis, typename Helper::Scalar{scale});
        HostVector result(base.size());
        for (std::size_t i = 0; i < base.size(); ++i) {
            result[i] = cast_scalar<Scalar>(base[i]);
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

    [[nodiscard]] static HostVector make_unit_tensor(Basis const& basis) {
        return make_identity_operator(basis);
    }

    [[nodiscard]] static Scalar scalar_from_accum(Accum value) {
        if constexpr (std::is_same_v<Scalar, __half> ||
                      std::is_same_v<Scalar, __nv_bfloat16>) {
            return cast_scalar<Scalar>(static_cast<float>(value));
        }
        else {
            return static_cast<Scalar>(value);
        }
    }

    static void expect_scalar_near_impl(Accum actual,
                                        Accum expected,
                                        char const* file,
                                        int line,
                                        char const* suite_name,
                                        char const* test_name) {
        auto const actual_d = scalar_to_double(actual);
        auto const expected_d = scalar_to_double(expected);
        auto const scale =
            std::max({1.0, std::abs(actual_d), std::abs(expected_d)});
        auto const tolerance = NumericTolerance<Scalar, Accum>::value * scale;
        if (std::abs(actual_d - expected_d) > tolerance) {
            ADD_FAILURE_AT(file, line)
                << suite_name << "." << test_name
                << ": scalar mismatch"
                << ", actual=" << actual_d
                << ", expected=" << expected_d
                << ", error=" << std::abs(actual_d - expected_d)
                << ", tolerance=" << tolerance;
        }
    }

    static void expect_scalar_near(Accum actual, Accum expected) {
        expect_scalar_near_impl(actual,
                                expected,
                                __FILE__,
                                __LINE__,
                                "TypedGpuAdjointTestBase",
                                "expect_scalar_near");
    }

    static void expect_tensor_near_impl(HostVector const& actual,
                                        HostVector const& expected,
                                        char const* file,
                                        int line,
                                        char const* suite_name,
                                        char const* test_name) {
        expect_tensor_near_impl(actual,
                                expected,
                                file,
                                line,
                                suite_name,
                                test_name,
                                NumericTolerance<Scalar, Accum>::value);
    }

    static void expect_tensor_near_impl(HostVector const& actual,
                                        HostVector const& expected,
                                        char const* file,
                                        int line,
                                        char const* suite_name,
                                        char const* test_name,
                                        double base_tolerance) {
        ASSERT_EQ(actual.size(), expected.size());

        struct FailureEntry {
            std::size_t index;
            double error;
        };

        struct EntryBuilder {
            HostVector const* actual;
            HostVector const* expected;

            FailureEntry operator()(std::size_t i) const {
                auto const actual_d = scalar_to_double((*actual)[i]);
                auto const expected_d = scalar_to_double((*expected)[i]);
                auto const scale =
                    std::max({1.0, std::abs(actual_d), std::abs(expected_d)});
                return FailureEntry{
                    i,
                    std::abs(actual_d - expected_d)};
            }
        };

        struct FailureCount {
            HostVector const* actual;
            HostVector const* expected;

            int operator()(FailureEntry const& entry) const noexcept {
                auto const actual_d = scalar_to_double((*actual)[entry.index]);
                auto const expected_d = scalar_to_double((*expected)[entry.index]);
                auto const scale =
                    std::max({1.0, std::abs(actual_d), std::abs(expected_d)});
                auto const tolerance = base_tolerance * scale;
                return entry.error > tolerance ? 1 : 0;
            }

            double base_tolerance;
        };

        struct FailurePredicate {
            HostVector const* actual;
            HostVector const* expected;

            bool operator()(FailureEntry const& entry) const noexcept {
                auto const actual_d = scalar_to_double((*actual)[entry.index]);
                auto const expected_d = scalar_to_double((*expected)[entry.index]);
                auto const scale =
                    std::max({1.0, std::abs(actual_d), std::abs(expected_d)});
                auto const tolerance = base_tolerance * scale;
                return entry.error > tolerance;
            }

            double base_tolerance;
        };

        auto const first = thrust::make_counting_iterator<std::size_t>(0);
        auto const last = first + actual.size();
        auto const entries = thrust::make_transform_iterator(
            first, EntryBuilder{&actual, &expected});

        auto const failure_count = thrust::transform_reduce(
            thrust::host,
            entries,
            entries + actual.size(),
            FailureCount{&actual, &expected, base_tolerance},
            0,
            thrust::plus<int>{});

        if (failure_count == 0) {
            SUCCEED();
            return;
        }

        thrust::host_vector<FailureEntry> failures(
            static_cast<std::size_t>(failure_count));
        auto const failure_end = thrust::copy_if(
            thrust::host,
            entries,
            entries + actual.size(),
            failures.begin(),
            FailurePredicate{&actual, &expected, base_tolerance});
        failures.resize(static_cast<std::size_t>(failure_end - failures.begin()));

        std::ostringstream report;
        report << "tensor mismatch at " << failures.size() << " coefficient(s)";
        if (failures.size() > 5) {
            auto const minmax = std::minmax_element(
                failures.begin(),
                failures.end(),
                [](FailureEntry const& lhs, FailureEntry const& rhs) {
                    return lhs.index < rhs.index;
                });
            auto const max_error = std::max_element(
                failures.begin(),
                failures.end(),
                [](FailureEntry const& lhs, FailureEntry const& rhs) {
                    return lhs.error < rhs.error;
                });
            report << ": min_index=" << (*minmax.first).index
                   << ", max_index=" << (*minmax.second).index
                   << ", max_error=" << (*max_error).error;
        }
        else {
            report << ":";
            for (auto const& entry : failures) {
                report << "\n  [" << entry.index << "] error=" << entry.error;
            }
        }

        ADD_FAILURE_AT(file, line)
            << suite_name << "." << test_name << ": " << report.str();
    }

    static void expect_tensor_near(HostVector const& actual,
                                   HostVector const& expected) {
        expect_tensor_near_impl(actual,
                                expected,
                                __FILE__,
                                __LINE__,
                                "TypedGpuAdjointTestBase",
                                "expect_tensor_near");
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
