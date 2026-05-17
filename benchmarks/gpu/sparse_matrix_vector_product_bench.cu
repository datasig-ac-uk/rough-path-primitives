#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <rpp/basis/tensor_basis.hpp>
#include <rpp/dense/batch.hpp>
#include <rpp/gpu/architecture.hpp>
#include <rpp/gpu/operations/block/basic/sparse_matrix_vector.hpp>
#include <rpp/gpu/strategies.hpp>
#include <rpp/sparse/matrix.hpp>

namespace {

using Scalar = float;
using Architecture = rpp::gpu::arch::Architecture32;
using Degree = typename Architecture::Degree;
using Index = typename Architecture::Index;
using Basis = rpp::TensorBasis<Architecture>;
using Strategy = rpp::gpu::strategies::BlockStrategy<Scalar, 256, Architecture>;
using Op = rpp::ops::SparseMatrixVectorProduct<Strategy>;

using HostScalarVector = thrust::host_vector<Scalar>;
using HostIndexVector = thrust::host_vector<Index>;
using DeviceScalarVector = thrust::device_vector<Scalar>;
using DeviceIndexVector = thrust::device_vector<Index>;

using DeviceCsrMatrix = rpp::sparse::OwnedCompressedMatrix<
    DeviceScalarVector,
    DeviceIndexVector,
    DeviceIndexVector,
    rpp::sparse::MatrixFormat::CSR
>;
using DeviceCscMatrix = rpp::sparse::OwnedCompressedMatrix<
    DeviceScalarVector,
    DeviceIndexVector,
    DeviceIndexVector,
    rpp::sparse::CompressedFormat::CSC
>;
using DeviceCsrMatrixView = rpp::sparse::CompressedMatrix<
    typename DeviceCsrMatrix::DataPointer,
    typename DeviceCsrMatrix::IndexPointer,
    typename DeviceCsrMatrix::OffsetPointer,
    rpp::sparse::CompressedFormat::CSR
>;
using DeviceCscMatrixView = rpp::sparse::CompressedMatrix<
    typename DeviceCscMatrix::DataPointer,
    typename DeviceCscMatrix::IndexPointer,
    typename DeviceCscMatrix::OffsetPointer,
    rpp::sparse::CompressedFormat::CSC
>;

constexpr unsigned kBlockSize = 128;

void check_cuda(cudaError_t status, char const* expression)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " failed: " + cudaGetErrorString(status));
    }
}

bool require_cuda_device(benchmark::State& state)
{
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        status == cudaErrorSystemDriverMismatch) {
        state.SkipWithError("No CUDA device is available");
        return false;
    }
    if (status != cudaSuccess) {
        state.SkipWithError(cudaGetErrorString(status));
        return false;
    }
    if (device_count == 0) {
        state.SkipWithError("No CUDA device is available");
        return false;
    }
    return true;
}

class CudaEvent {
    cudaEvent_t event_ = nullptr;

public:
    CudaEvent()
    {
        check_cuda(cudaEventCreate(&event_), "cudaEventCreate");
    }

    CudaEvent(CudaEvent const&) = delete;
    CudaEvent& operator=(CudaEvent const&) = delete;

    ~CudaEvent()
    {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }
};

[[nodiscard]] HostIndexVector make_degree_begin(Degree width, Degree depth)
{
    HostIndexVector result(static_cast<std::size_t>(depth + 2));
    for (Degree degree = 1; degree <= depth + 1; ++degree) {
        result[static_cast<std::size_t>(degree)] =
            Index{1} + static_cast<Index>(width) * result[static_cast<std::size_t>(degree - 1)];
    }
    return result;
}

[[nodiscard]] Scalar value_for(std::size_t index, std::uint32_t salt) noexcept
{
    return static_cast<Scalar>((index * 17 + salt * 31) % 97 - 48) / Scalar{257};
}

[[nodiscard]] HostScalarVector make_batch(Index tensor_count, Basis const& basis, std::uint32_t salt)
{
    HostScalarVector result(static_cast<std::size_t>(tensor_count * basis.size()));
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = value_for(i, salt);
    }
    return result;
}

struct BasisData {
    HostIndexVector degree_begin;
    Basis host_basis;
    DeviceIndexVector device_degree_begin;
    Basis device_basis;

    BasisData(Degree width, Degree depth)
        : degree_begin(make_degree_begin(width, depth)),
          host_basis(width, depth, thrust::raw_pointer_cast(degree_begin.data())),
          device_degree_begin(degree_begin),
          device_basis(width, depth, thrust::raw_pointer_cast(device_degree_begin.data()))
    {
    }
};

struct SparseStorage {
    HostScalarVector values;
    HostIndexVector indices;
    HostIndexVector offsets;
    HostScalarVector csc_values;
    HostIndexVector csc_indices;
    HostIndexVector csc_offsets;

    explicit SparseStorage(Index size)
    {
        offsets.reserve(static_cast<std::size_t>(size + 1));
        values.reserve(static_cast<std::size_t>(5 * size));
        indices.reserve(static_cast<std::size_t>(5 * size));

        for (Index row = 0; row < size; ++row) {
            offsets.push_back(static_cast<Index>(values.size()));
            for (const auto col : columns_for(row, size)) {
                values.push_back(value_for(static_cast<std::size_t>(row) * 131 + static_cast<std::size_t>(col), 5));
                indices.push_back(col);
            }
        }
        offsets.push_back(static_cast<Index>(values.size()));

        build_csc(size);
    }

    [[nodiscard]] std::size_t nnz() const noexcept { return values.size(); }

private:
    [[nodiscard]] static HostIndexVector columns_for(Index row, Index size)
    {
        HostIndexVector cols;
        auto add_col = [&](Index col) {
            if (col >= 0 && col < size) {
                cols.push_back(col);
            }
        };

        for (Index delta = -2; delta <= 2; ++delta) {
            add_col(row + delta);
        }

        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        return cols;
    }

    void build_csc(Index size)
    {
        csc_values.assign(values.size(), Scalar{});
        csc_indices.assign(indices.size(), Index{0});
        csc_offsets.assign(static_cast<std::size_t>(size + 1), Index{0});

        for (Index row = 0; row < size; ++row) {
            for (auto entry = offsets[static_cast<std::size_t>(row)];
                 entry < offsets[static_cast<std::size_t>(row + 1)];
                 ++entry) {
                ++csc_offsets[static_cast<std::size_t>(indices[static_cast<std::size_t>(entry)] + 1)];
            }
        }

        for (Index col = 0; col < size; ++col) {
            csc_offsets[static_cast<std::size_t>(col + 1)] += csc_offsets[static_cast<std::size_t>(col)];
        }

        auto write_locs = csc_offsets;
        for (Index row = 0; row < size; ++row) {
            for (auto entry = offsets[static_cast<std::size_t>(row)];
                 entry < offsets[static_cast<std::size_t>(row + 1)];
                 ++entry) {
                const auto col = indices[static_cast<std::size_t>(entry)];
                const auto dest = write_locs[static_cast<std::size_t>(col)]++;
                csc_values[static_cast<std::size_t>(dest)] = values[static_cast<std::size_t>(entry)];
                csc_indices[static_cast<std::size_t>(dest)] = row;
            }
        }
    }
};

[[nodiscard]] DeviceCsrMatrix make_device_csr(SparseStorage const& storage, Index rows, Index cols)
{
    return {
        DeviceScalarVector(storage.values),
        DeviceIndexVector(storage.indices),
        DeviceIndexVector(storage.offsets),
        static_cast<typename DeviceCsrMatrix::difference_type>(storage.values.size()),
        rows,
        cols
    };
}

[[nodiscard]] DeviceCscMatrix make_device_csc(SparseStorage const& storage, Index rows, Index cols)
{
    return {
        DeviceScalarVector(storage.csc_values),
        DeviceIndexVector(storage.csc_indices),
        DeviceIndexVector(storage.csc_offsets),
        static_cast<typename DeviceCscMatrix::difference_type>(storage.csc_values.size()),
        cols,
        rows
    };
}

struct GpuSparseCase {
    BasisData basis_data;
    Index tensor_count;
    Strategy strategy{kBlockSize};
    SparseStorage storage;
    DeviceScalarVector out;
    DeviceScalarVector arg;
    DeviceCsrMatrix csr_matrix;
    DeviceCscMatrix csc_matrix;

    GpuSparseCase(Degree width, Degree depth, Index tensor_count_)
        : basis_data(width, depth),
          tensor_count(tensor_count_),
          storage(basis_data.host_basis.size()),
          out(make_batch(tensor_count, basis_data.host_basis, 1)),
          arg(make_batch(tensor_count, basis_data.host_basis, 2)),
          csr_matrix(make_device_csr(storage, basis_data.host_basis.size(), basis_data.host_basis.size())),
          csc_matrix(make_device_csc(storage, basis_data.host_basis.size(), basis_data.host_basis.size()))
    {
    }

    [[nodiscard]] auto out_batch()
    {
        return rpp::dense::make_vector_batch(
            thrust::raw_pointer_cast(out.data()),
            basis_data.host_basis.size(),
            Degree{0},
            basis_data.host_basis.depth
        );
    }

    [[nodiscard]] auto arg_batch() const
    {
        return rpp::dense::make_vector_batch(
            thrust::raw_pointer_cast(arg.data()),
            basis_data.host_basis.size(),
            Degree{0},
            basis_data.host_basis.depth
        );
    }

    [[nodiscard]] DeviceCsrMatrixView csr_view() const
    {
        return static_cast<DeviceCsrMatrixView>(csr_matrix);
    }

    [[nodiscard]] DeviceCscMatrixView csc_view() const
    {
        return static_cast<DeviceCscMatrixView>(csc_matrix);
    }

    [[nodiscard]] std::size_t shared_memory_size() const noexcept
    {
        return Op::scratch_space_size(strategy, basis_data.host_basis);
    }
};

void apply_sparse_configs(benchmark::internal::Benchmark* benchmark)
{
    benchmark
        ->Args({2, 6, 4096})
        ->Args({3, 5, 2048})
        ->Args({4, 4, 2048})
        ->Args({5, 3, 4096});
}

template <typename MatrixFactory>
void run_sparse_benchmark(benchmark::State& state, MatrixFactory&& make_matrix)
{
    if (!require_cuda_device(state)) {
        return;
    }

    try {
        GpuSparseCase test_case(
            static_cast<Degree>(state.range(0)),
            static_cast<Degree>(state.range(1)),
            static_cast<Index>(state.range(2))
        );
        auto matrix = make_matrix(test_case);
        CudaEvent start;
        CudaEvent stop;

        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        for (auto _ : state) {
            check_cuda(cudaEventRecord(start.get()), "cudaEventRecord(start)");
            rpp::gpu::block::sparse_matrix_vector_product_kernel<<<
                static_cast<unsigned>(test_case.tensor_count),
                test_case.strategy.block_size,
                test_case.shared_memory_size()
            >>>(
                test_case.out_batch(),
                matrix,
                test_case.arg_batch(),
                test_case.basis_data.device_basis,
                test_case.basis_data.device_basis,
                test_case.strategy,
                test_case.tensor_count
            );
            check_cuda(cudaGetLastError(), "sparse_matrix_vector_product_kernel");
            check_cuda(cudaEventRecord(stop.get()), "cudaEventRecord(stop)");
            check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize(stop)");

            float elapsed_ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()), "cudaEventElapsedTime");
            state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
            benchmark::DoNotOptimize(thrust::raw_pointer_cast(test_case.out.data()));
        }

        const auto dim = test_case.basis_data.host_basis.size();
        state.counters["width"] = static_cast<double>(test_case.basis_data.host_basis.width);
        state.counters["depth"] = static_cast<double>(test_case.basis_data.host_basis.depth);
        state.counters["dim"] = static_cast<double>(dim);
        state.counters["batch"] = static_cast<double>(test_case.tensor_count);
        state.counters["nnz"] = static_cast<double>(test_case.storage.nnz());
        state.counters["nnz/row"] = static_cast<double>(test_case.storage.nnz()) / static_cast<double>(dim);
        state.counters["density"] =
            static_cast<double>(test_case.storage.nnz()) / static_cast<double>(dim * dim);
        state.SetItemsProcessed(
            state.iterations() *
            static_cast<std::int64_t>(test_case.tensor_count) *
            static_cast<std::int64_t>(test_case.storage.nnz())
        );
        state.SetBytesProcessed(
            state.iterations() *
            static_cast<std::int64_t>(test_case.tensor_count) *
            static_cast<std::int64_t>(
                test_case.storage.nnz() * (2 * sizeof(Scalar) + sizeof(Index)) +
                static_cast<std::size_t>(dim) * sizeof(Scalar)
            )
        );
    } catch (std::exception const& err) {
        state.SkipWithError(err.what());
    }
}

void BM_GpuSparseMatrixVectorProductCSR(benchmark::State& state)
{
    run_sparse_benchmark(state, [](GpuSparseCase const& test_case) { return test_case.csr_view(); });
}

void BM_GpuSparseMatrixVectorProductCSC(benchmark::State& state)
{
    run_sparse_benchmark(state, [](GpuSparseCase const& test_case) { return test_case.csc_view(); });
}

BENCHMARK(BM_GpuSparseMatrixVectorProductCSR)->Apply(apply_sparse_configs)->UseManualTime();
BENCHMARK(BM_GpuSparseMatrixVectorProductCSC)->Apply(apply_sparse_configs)->UseManualTime();

} // namespace
