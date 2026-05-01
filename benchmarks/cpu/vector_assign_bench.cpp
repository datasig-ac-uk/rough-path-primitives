#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_VectorAssign(benchmark::State& state)
{
    using Op = rpp::ops::VectorAssign<rb::Strategy>;
    rb::run_tensor_benchmark<Op>(state, [](Op const& op, rb::Context const& ctx, rb::TensorCase& test_case) {
        auto out = test_case.out_vector();
        auto arg = test_case.a_vector();
        op(ctx, out, arg);
    });
}

BENCHMARK(BM_VectorAssign)->Apply(rb::apply_configs);

} // namespace
