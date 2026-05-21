#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_FTAdjLMul(benchmark::State& state) {
    using Op = rpp::ops::FTAdjLMul<rb::Strategy>;
    rb::run_tensor_benchmark<Op>(
        state,
        [](Op const& op, rb::Context const& ctx, rb::TensorCase& test_case) {
            auto out = test_case.out_tensor();
            auto op_arg = test_case.a_tensor();
            auto arg = test_case.zero_tensor();
            op(ctx, out, op_arg, arg);
        });
}

BENCHMARK(BM_FTAdjLMul)->Apply(rb::apply_configs);

} // namespace
