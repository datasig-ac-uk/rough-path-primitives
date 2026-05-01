#ifndef RPP_BENCHMARKS_CPU_BENCHMARK_HELPERS_HPP
#define RPP_BENCHMARKS_CPU_BENCHMARK_HELPERS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <benchmark/benchmark.h>

#include <rpp/basis.hpp>
#include <rpp/cpu/ops/single_thread.hpp>
#include <rpp/cpu/strategies.hpp>
#include <rpp/dense/views.hpp>
#include <rpp/sparse/compressed_matrix.hpp>

namespace rpp::benchmarks::cpu {

using Scalar = double;
using Basis = StandardTensorBasis;
using Degree = Basis::Degree;
using Index = Basis::Index;

struct BenchmarkArchitecture {
    using Degree = Basis::Degree;
    using Index = Basis::Index;
    using Letter = std::uint8_t;
    using Bitmask = std::uint32_t;

    static constexpr unsigned max_depth = 16;
};

using Strategy = rpp::cpu::strategies::SingleThreadStrategy<Scalar, BenchmarkArchitecture>;
using Context = Strategy::Context;

using VectorView = dense::DenseVectorView<Scalar*, Basis>;
using ConstVectorView = dense::DenseVectorView<Scalar const*, Basis>;
using TensorView = dense::DenseTensorView<Scalar*, Basis>;
using ConstTensorView = dense::DenseTensorView<Scalar const*, Basis>;

[[nodiscard]] inline std::vector<Index> make_degree_begin(Degree width, Degree depth)
{
    std::vector<Index> result(static_cast<std::size_t>(depth + 2));
    for (Degree degree = 1; degree <= depth + 1; ++degree) {
        result[static_cast<std::size_t>(degree)] =
            Index{1} + static_cast<Index>(width) * result[static_cast<std::size_t>(degree - 1)];
    }
    return result;
}

[[nodiscard]] inline Scalar value_for(std::size_t index, std::uint32_t salt) noexcept
{
    return static_cast<Scalar>((index * 17 + salt * 31) % 97 - 48) / Scalar{257};
}

inline void fill_values(std::vector<Scalar>& data, std::uint32_t salt)
{
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = value_for(i, salt);
    }
}

struct TensorCase {
    std::vector<Index> degree_begin;
    Basis basis;
    std::vector<Scalar> out;
    std::vector<Scalar> a;
    std::vector<Scalar> b;
    std::vector<Scalar> c;
    std::vector<Scalar> zero;
    std::vector<Scalar> identity;
    std::vector<std::byte> scratch;
    Strategy strategy{};

    explicit TensorCase(Degree width, Degree depth)
        : degree_begin(make_degree_begin(width, depth)),
          basis(width, depth, degree_begin.data()),
          out(static_cast<std::size_t>(basis.size())),
          a(static_cast<std::size_t>(basis.size())),
          b(static_cast<std::size_t>(basis.size())),
          c(static_cast<std::size_t>(basis.size())),
          zero(static_cast<std::size_t>(basis.size()), Scalar{0}),
          identity(static_cast<std::size_t>(basis.size()), Scalar{0})
    {
        fill_values(out, 1);
        fill_values(a, 2);
        fill_values(b, 3);
        fill_values(c, 4);
        if (!identity.empty()) {
            identity[0] = Scalar{1};
        }
    }

    template <typename Op>
    [[nodiscard]] Context context()
    {
        const auto scratch_size = Op::scratch_space_size(strategy, basis);
        scratch.assign(std::max<std::size_t>(scratch_size, 1), std::byte{});
        return Strategy::make_context(scratch.data());
    }

    [[nodiscard]] VectorView out_vector() { return {out.data(), basis}; }
    [[nodiscard]] TensorView out_tensor() { return {out.data(), basis}; }
    [[nodiscard]] ConstVectorView a_vector() const { return {a.data(), basis}; }
    [[nodiscard]] ConstTensorView a_tensor() const { return {a.data(), basis}; }
    [[nodiscard]] ConstTensorView b_tensor() const { return {b.data(), basis}; }
    [[nodiscard]] ConstTensorView c_tensor() const { return {c.data(), basis}; }
    [[nodiscard]] ConstTensorView zero_tensor() const { return {zero.data(), basis}; }
    [[nodiscard]] ConstTensorView identity_tensor() const { return {identity.data(), basis}; }
};

inline void apply_configs(benchmark::internal::Benchmark* benchmark)
{
    benchmark
        ->Args({2, 6})
        ->Args({3, 5})
        ->Args({4, 4})
        ->Args({5, 3});
}

inline void record_case_metrics(benchmark::State& state, TensorCase const& test_case)
{
    state.counters["width"] = static_cast<double>(test_case.basis.width);
    state.counters["depth"] = static_cast<double>(test_case.basis.depth);
    state.counters["dim"] = static_cast<double>(test_case.basis.size());
    state.SetItemsProcessed(state.iterations() * test_case.basis.size());
}

template <typename Op, typename Fn>
void run_tensor_benchmark(benchmark::State& state, Fn&& fn)
{
    TensorCase test_case(static_cast<Degree>(state.range(0)), static_cast<Degree>(state.range(1)));
    Op op;
    auto ctx = test_case.template context<Op>();

    for (auto _ : state) {
        fn(op, ctx, test_case);
        benchmark::ClobberMemory();
    }

    record_case_metrics(state, test_case);
}

struct SparseCase {
    TensorCase tensors;
    std::vector<Scalar> values;
    std::vector<Index> indices;
    std::vector<Index> offsets;

    explicit SparseCase(Degree width, Degree depth)
        : tensors(width, depth)
    {
        const auto size = tensors.basis.size();
        offsets.reserve(static_cast<std::size_t>(size + 1));
        values.reserve(static_cast<std::size_t>(3 * size));
        indices.reserve(static_cast<std::size_t>(3 * size));

        for (Index row = 0; row < size; ++row) {
            offsets.push_back(static_cast<Index>(values.size()));
            for (Index delta = -1; delta <= 1; ++delta) {
                const auto col = row + delta;
                if (col >= 0 && col < size) {
                    values.push_back(value_for(static_cast<std::size_t>(row + col), 5));
                    indices.push_back(col);
                }
            }
        }
        offsets.push_back(static_cast<Index>(values.size()));
    }

    [[nodiscard]] auto csr() const
    {
        return sparse::make_csr_matrix(
            values.data(),
            indices.data(),
            offsets.data(),
            values.size(),
            tensors.basis.size(),
            tensors.basis.size()
        );
    }

    [[nodiscard]] auto csc() const
    {
        return sparse::make_csc_matrix(
            values.data(),
            indices.data(),
            offsets.data(),
            values.size(),
            tensors.basis.size(),
            tensors.basis.size()
        );
    }
};

} // namespace rpp::benchmarks::cpu

#endif // RPP_BENCHMARKS_CPU_BENCHMARK_HELPERS_HPP
