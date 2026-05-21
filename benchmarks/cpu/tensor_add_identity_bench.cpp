#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_TensorAddIdentity(benchmark::State& state) {
    using Op = rpp::ops::TensorAddIdentity<rb::Strategy>;
    rb::run_tensor_benchmark<Op>(
        state,
        [](Op const& op, rb::Context const& ctx, rb::TensorCase& test_case) {
            auto out = test_case.out_tensor();
            op(ctx, out, rb::Scalar{1});
        });
}

BENCHMARK(BM_TensorAddIdentity)->Apply(rb::apply_configs);

} // namespace
