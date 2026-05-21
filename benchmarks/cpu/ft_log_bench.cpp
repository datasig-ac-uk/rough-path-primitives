#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_FTLog(benchmark::State& state) {
    using Op = rpp::ops::FTLog<rb::Strategy>;
    rb::run_tensor_benchmark<Op>(
        state,
        [](Op const& op, rb::Context const& ctx, rb::TensorCase& test_case) {
            auto out = test_case.out_tensor();
            auto arg = test_case.a_tensor();
            op(ctx, out, arg);
        });
}

BENCHMARK(BM_FTLog)->Apply(rb::apply_configs);

} // namespace
