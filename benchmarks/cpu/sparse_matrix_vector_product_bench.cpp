#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

template <typename MatrixFactory>
void run_sparse_benchmark(benchmark::State& state, MatrixFactory&& make_matrix)
{
    rb::SparseCase test_case(
        static_cast<rb::Degree>(state.range(0)),
        static_cast<rb::Degree>(state.range(1))
    );
    auto matrix = make_matrix(test_case);
    using Matrix = decltype(matrix);

    using Op = rpp::ops::SparseMatrixVectorProduct<rb::Strategy, rpp::sparse::matrix_format_v<Matrix>>;
    auto ctx = test_case.tensors.template context<Op>();
    Op op;

    for (auto _ : state) {
        auto out = test_case.tensors.out_vector();
        auto arg = test_case.tensors.a_vector();
        op(ctx, out, matrix, arg);
        benchmark::ClobberMemory();
    }

    state.counters["width"] = static_cast<double>(test_case.tensors.basis.width);
    state.counters["depth"] = static_cast<double>(test_case.tensors.basis.depth);
    state.counters["dim"] = static_cast<double>(test_case.tensors.basis.size());
    state.counters["nnz"] = static_cast<double>(test_case.nnz());
    state.counters["nnz/row"] =
        static_cast<double>(test_case.nnz()) / static_cast<double>(test_case.tensors.basis.size());
    state.counters["density"] =
        static_cast<double>(test_case.nnz()) /
        static_cast<double>(test_case.tensors.basis.size() * test_case.tensors.basis.size());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(test_case.nnz()));
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
