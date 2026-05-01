#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

template <typename MatrixFactory>
void run_sparse_benchmark(benchmark::State& state, MatrixFactory&& make_matrix)
{
    using Op = rpp::ops::SparseMatrixVectorProduct<rb::Strategy>;
    rb::SparseCase test_case(static_cast<rb::Degree>(state.range(0)), static_cast<rb::Degree>(state.range(1)));
    Op op;
    auto ctx = test_case.tensors.template context<Op>();

    for (auto _ : state) {
        auto out = test_case.tensors.out_vector();
        auto arg = test_case.tensors.a_vector();
        auto matrix = make_matrix(test_case);
        op(ctx, out, matrix, arg);
        benchmark::ClobberMemory();
    }

    state.counters["width"] = static_cast<double>(test_case.tensors.basis.width);
    state.counters["depth"] = static_cast<double>(test_case.tensors.basis.depth);
    state.counters["dim"] = static_cast<double>(test_case.tensors.basis.size());
    state.counters["nnz"] = static_cast<double>(test_case.values.size());
    state.SetItemsProcessed(state.iterations() * test_case.tensors.basis.size());
}

void BM_SparseMatrixVectorProductCSR(benchmark::State& state)
{
    run_sparse_benchmark(state, [](rb::SparseCase const& test_case) { return test_case.csr(); });
}

void BM_SparseMatrixVectorProductCSC(benchmark::State& state)
{
    run_sparse_benchmark(state, [](rb::SparseCase const& test_case) { return test_case.csc(); });
}

BENCHMARK(BM_SparseMatrixVectorProductCSR)->Apply(rb::apply_configs);
BENCHMARK(BM_SparseMatrixVectorProductCSC)->Apply(rb::apply_configs);

} // namespace
