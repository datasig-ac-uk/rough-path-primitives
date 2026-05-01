#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_FTExp(benchmark::State& state)
{
    using Op = rpp::ops::FTExp<rb::Strategy>;
    rb::run_tensor_benchmark<Op>(state, [](Op const& op, rb::Context const& ctx, rb::TensorCase& test_case) {
        auto out = test_case.out_tensor();
        auto arg = test_case.a_tensor();
        op(ctx, out, arg);
    });
}

BENCHMARK(BM_FTExp)->Apply(rb::apply_configs);

} // namespace
