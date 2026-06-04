#ifndef RPP_TESTS_GPU_OPS_BLOCK_GPU_BLOCK_TEST_HELPER_CUH
#define RPP_TESTS_GPU_OPS_BLOCK_GPU_BLOCK_TEST_HELPER_CUH

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <rpp/basis/tensor_basis.hpp>
#include <rpp/cpu/strategies.hpp>
#include <rpp/gpu/architecture.hpp>
#include <rpp/gpu/strategies.hpp>
#include <rpp/views/batch.hpp>

#include "rpp/views/dense_graded_vector_view.hpp"
#include "rpp/views/dense_tensor_view.hpp"
#include "rpp/views/scalar_view.hpp"

#define RPP_CUDA_ASSERT(expr)                                                  \
    do {                                                                       \
        const cudaError_t rpp_cuda_err = (expr);                               \
        ASSERT_EQ(rpp_cuda_err, cudaSuccess)                                   \
            << #expr << " failed: " << cudaGetErrorString(rpp_cuda_err);       \
    }                                                                          \
    while (false)

#define RPP_REQUIRE_CUDA_DEVICE()                                              \
    do {                                                                       \
        int rpp_cuda_device_count = 0;                                         \
        const cudaError_t rpp_cuda_err =                                       \
            cudaGetDeviceCount(&rpp_cuda_device_count);                        \
        if (rpp_cuda_err == cudaErrorNoDevice ||                               \
            rpp_cuda_err == cudaErrorInsufficientDriver ||                     \
            rpp_cuda_err == cudaErrorSystemDriverMismatch) {                   \
            GTEST_SKIP() << "No CUDA device is available";                     \
        }                                                                      \
        ASSERT_EQ(rpp_cuda_err, cudaSuccess)                                   \
            << "cudaGetDeviceCount failed: "                                   \
            << cudaGetErrorString(rpp_cuda_err);                               \
        if (rpp_cuda_device_count == 0) {                                      \
            GTEST_SKIP() << "No CUDA device is available";                     \
        }                                                                      \
    }                                                                          \
    while (false)

namespace rpp::tests {

struct GpuBlockTestHelper {
    using Scalar = float;
    using GpuArchitecture = gpu::arch::Architecture32;
    using HostArchitecture = arch::NativeArchitecture;
    using Degree = typename HostArchitecture::Degree;
    using Index = typename HostArchitecture::Index;
    using Basis = basis::TensorBasis<HostArchitecture>;

    template <typename T>
    using HostVector = thrust::host_vector<T>;

    template <typename T>
    using DeviceVector = thrust::device_vector<T>;

    struct CpuArchitecture {
        using Degree = GpuBlockTestHelper::Degree;
        using Index = GpuBlockTestHelper::Index;
        using Letter = std::uint8_t;
        using Bitmask = std::uint32_t;

        static constexpr unsigned max_depth = 16;
    };

    static constexpr unsigned block_size = 128;
    using CpuStrategy =
        cpu::strategies::SingleThreadStrategy<Scalar, CpuArchitecture>;
    using GpuStrategy = gpu::strategies::
        BlockStrategy<Scalar, block_size, 256, GpuArchitecture>;

    static constexpr Index tensor_count = 1;

    struct TensorConfig {
        Degree width;
        Degree depth;
    };

    struct BasisData {
        HostVector<Index> degree_begin;
        Basis basis;

        BasisData(Degree width, Degree depth)
            : degree_begin(make_degree_begin(width, depth)),
              basis{width, depth, degree_begin.data()} {}
    };

    struct DeviceBasis {
        DeviceVector<Index> degree_begin;
        Basis basis;

        explicit DeviceBasis(BasisData const& basis_data)
            : degree_begin(basis_data.degree_begin),
              basis(basis_data.basis.width,
                    basis_data.basis.depth,
                    thrust::raw_pointer_cast(degree_begin.data())) {}
    };

    struct RandomScalar {
        unsigned seed;
        Scalar scale;

        __host__ __device__ Scalar operator()(std::size_t index) const {
            thrust::default_random_engine rng(seed);
            thrust::uniform_real_distribution<Scalar> dist(Scalar{-1},
                                                           Scalar{1});
            rng.discard(index);
            return scale * dist(rng);
        }
    };

    [[nodiscard]] static HostVector<Index> make_degree_begin(Degree width,
                                                             Degree depth) {
        HostVector<Index> result(static_cast<std::size_t>(depth + 2));
        for (Degree degree = 1; degree <= depth + 1; ++degree) {
            result[static_cast<std::size_t>(degree)] =
                1 + width * result[static_cast<std::size_t>(degree - 1)];
        }
        return result;
    }

    [[nodiscard]] static HostVector<Scalar>
    make_batch(unsigned seed, Basis const& basis, Scalar scale = Scalar{1}) {
        HostVector<Scalar> result(
            static_cast<std::size_t>(tensor_count * basis.size()));
        auto const first = thrust::make_counting_iterator<std::size_t>(0);
        thrust::transform(thrust::host,
                          first,
                          first + result.size(),
                          result.begin(),
                          RandomScalar{seed, scale});
        return result;
    }

    [[nodiscard]] static HostVector<Scalar>
    make_zero_batch(Basis const& basis) {
        HostVector<Scalar> result(
            static_cast<std::size_t>(tensor_count * basis.size()));
        thrust::fill(result.begin(), result.end(), Scalar{0});
        return result;
    }

    [[nodiscard]] static CpuStrategy cpu_strategy() noexcept {
        return CpuStrategy{};
    }

    [[nodiscard]] static GpuStrategy gpu_strategy() noexcept {
        return GpuStrategy{block_size};
    }

    template <typename T>
    [[nodiscard]] static T* host_data(HostVector<T>& data) {
        return thrust::raw_pointer_cast(data.data());
    }

    template <typename T>
    [[nodiscard]] static T const* host_data(HostVector<T> const& data) {
        return thrust::raw_pointer_cast(data.data());
    }

    template <typename T>
    [[nodiscard]] static auto device_data(DeviceVector<T>& data) {
        return typename GpuArchitecture::template Ptr<T>(
            thrust::raw_pointer_cast(data.data()));
    }

    template <typename T>
    [[nodiscard]] static auto device_data(DeviceVector<T> const& data) {
        return typename GpuArchitecture::template Ptr<const T>(
            thrust::raw_pointer_cast(data.data()));
    }

    template <typename T>
    [[nodiscard]] static HostVector<T>
    copy_to_host(DeviceVector<T> const& data) {
        HostVector<T> result(data.size());
        thrust::copy(data.begin(), data.end(), result.begin());
        return result;
    }

    template <typename T>
    [[nodiscard]] static auto host_vector_batch(HostVector<T>& data,
                                                Basis const& basis) {
        return rpp::make_graded_vector_batch(
            host_data(data), basis.size(), basis, Degree{0}, basis.depth);
    }

    template <typename T>
    [[nodiscard]] static auto host_vector_batch(HostVector<T> const& data,
                                                Basis const& basis) {
        return rpp::make_graded_vector_batch(
            host_data(data), basis.size(), basis, Degree{0}, basis.depth);
    }

    template <typename T>
    [[nodiscard]] static auto device_vector_batch(DeviceVector<T>& data,
                                                  Basis const& basis) {
        return rpp::make_tensor_batch(
            device_data(data), basis.size(), Degree{0}, basis.depth);
    }

    template <typename T>
    [[nodiscard]] static auto
    device_vector_batch(DeviceVector<T> const& data, Basis const& basis) {
        return rpp::make_tensor_batch(
            device_data(data), basis.size(), Degree{0}, basis.depth);
    }

    template <typename T>
    [[nodiscard]] static auto host_tensor_batch(HostVector<T>& data,
                                                Basis const& basis) {
        return rpp::make_tensor_batch(
            host_data(data), basis.size(), basis, Degree{0}, basis.depth);
    }

    template <typename T>
    [[nodiscard]] static auto host_tensor_batch(HostVector<T> const& data,
                                                Basis const& basis) {
        return rpp::make_tensor_batch(
            host_data(data), basis.size(),  Degree{0}, basis.depth);
    }

    template <typename T>
    [[nodiscard]] static auto device_tensor_batch(DeviceVector<T>& data,
                                                  Basis const& basis) {
        return rpp::make_tensor_batch(
            device_data(data), basis.size(), Degree{0}, basis.depth);
    }

    template <typename T>
    [[nodiscard]] static auto
    device_tensor_batch(DeviceVector<T> const& data, Basis const& basis) {
        return rpp::make_tensor_batch(
            device_data(data), basis.size(), Degree{0}, basis.depth);
    }

    template <typename T>
    [[nodiscard]] static auto device_scalar_batch(DeviceVector<T>& data) {
        return rpp::make_scalar_batch(device_data(data));
    }

    template <typename T>
    [[nodiscard]] static auto
    device_scalar_batch(DeviceVector<T> const& data) {
        return rpp::make_scalar_batch(device_data(data));
    }

    template <typename GpuOp>
    [[nodiscard]] static std::size_t shared_memory_size(Basis const& basis) {
        auto const strategy = gpu_strategy();
        return GpuOp::scratch_space_size(strategy, basis);
    }

    static void expect_near(Scalar actual,
                            Scalar expected,
                            Scalar absolute_tolerance,
                            Scalar relative_tolerance) {
        auto const difference = std::abs(actual - expected);
        auto const scale =
            std::max(Scalar{1}, std::max(std::abs(actual), std::abs(expected)));
        auto const tolerance =
            std::max(absolute_tolerance, relative_tolerance * scale);
        EXPECT_LE(difference, tolerance)
            << "actual: " << actual << ", expected: " << expected
            << ", absolute tolerance: " << absolute_tolerance
            << ", relative tolerance: " << relative_tolerance;
    }

    static void expect_near(Scalar actual, Scalar expected, Scalar tolerance) {
        expect_near(actual, expected, tolerance, tolerance);
    }

    static void expect_near(HostVector<Scalar> const& actual,
                            HostVector<Scalar> const& expected,
                            Scalar absolute_tolerance,
                            Scalar relative_tolerance) {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            auto const difference = std::abs(actual[i] - expected[i]);
            auto const scale =
                std::max(Scalar{1},
                         std::max(std::abs(actual[i]), std::abs(expected[i])));
            auto const tolerance =
                std::max(absolute_tolerance, relative_tolerance * scale);
            EXPECT_LE(difference, tolerance)
                << "at coefficient " << i << ", actual: " << actual[i]
                << ", expected: " << expected[i]
                << ", absolute tolerance: " << absolute_tolerance
                << ", relative tolerance: " << relative_tolerance;
        }
    }

    static void expect_near(HostVector<Scalar> const& actual,
                            HostVector<Scalar> const& expected,
                            Scalar tolerance) {
        expect_near(actual, expected, tolerance, tolerance);
    }
};

inline constexpr GpuBlockTestHelper::TensorConfig gpu_block_test_configs[] = {
    {2, 3},
    {3, 2},
};

} // namespace rpp::tests

#endif // RPP_TESTS_GPU_OPS_BLOCK_GPU_BLOCK_TEST_HELPER_CUH
