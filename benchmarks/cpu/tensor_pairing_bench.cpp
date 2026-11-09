#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_TensorPairing(benchmark::State& state) {
    using Op = rpp::ops::TensorPairing<rb::Strategy>;
    rb::TensorCase test_case(static_cast<rb::Degree>(state.range(0)),
                             static_cast<rb::Degree>(state.range(1)));
    auto ctx = test_case.template context<Op>();
    Op op{ctx, rpp::basis::make_basis_pack(test_case.basis)};
    rb::Scalar result{0};

    for (auto _ : state) {
        auto functional = test_case.a_tensor();
        auto arg = test_case.b_tensor();
        op(ctx, result, functional, arg);
        benchmark::DoNotOptimize(result);
    }

    rb::record_case_metrics(state, test_case);
}

BENCHMARK(BM_TensorPairing)->Apply(rb::apply_configs);

} // namespace
