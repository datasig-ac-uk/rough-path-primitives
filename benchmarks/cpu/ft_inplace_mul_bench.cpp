#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_FTInplaceMul(benchmark::State& state)
{
    using Op = rpp::ops::FTInplaceMul<rb::Strategy>;
    rb::run_tensor_benchmark<Op>(state, [](Op const& op, rb::Context const& ctx, rb::TensorCase& test_case) {
        auto lhs = test_case.out_tensor();
        auto rhs = test_case.identity_tensor();
        op(ctx, lhs, rhs);
    });
}

BENCHMARK(BM_FTInplaceMul)->Apply(rb::apply_configs);

} // namespace
