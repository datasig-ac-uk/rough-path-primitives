#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_FTFMExp(benchmark::State& state)
{
    using Op = rpp::ops::FTFMExp<rb::Strategy>;
    rb::run_tensor_benchmark<Op>(state, [](Op const& op, rb::Context const& ctx, rb::TensorCase& test_case) {
        auto out = test_case.out_tensor();
        auto multiplier = test_case.a_tensor();
        auto exponent = test_case.b_tensor();
        op(ctx, out, multiplier, exponent);
    });
}

BENCHMARK(BM_FTFMExp)->Apply(rb::apply_configs);

} // namespace
