#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_FTFma(benchmark::State& state) {
    using Op = rpp::ops::FTFma<rb::Strategy>;
    rb::run_tensor_benchmark<Op>(
        state,
        [](Op const& op, rb::Context const& ctx, rb::TensorCase& test_case) {
            auto out = test_case.out_tensor();
            auto a = test_case.a_tensor();
            auto b = test_case.b_tensor();
            auto c = test_case.c_tensor();
            op(ctx, out, a, b, c, rb::Scalar{0.25}, rb::Scalar{0.75});
        });
}

BENCHMARK(BM_FTFma)->Apply(rb::apply_configs);

} // namespace
