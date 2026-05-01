#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_FTMul(benchmark::State& state)
{
    using Op = rpp::ops::FTMul<rb::Strategy>;
    rb::run_tensor_benchmark<Op>(state, [](Op const& op, rb::Context const& ctx, rb::TensorCase& test_case) {
        auto out = test_case.out_tensor();
        auto lhs = test_case.a_tensor();
        auto rhs = test_case.b_tensor();
        op(ctx, out, lhs, rhs);
    });
}

BENCHMARK(BM_FTMul)->Apply(rb::apply_configs);

} // namespace
