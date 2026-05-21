#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_STInplaceFma(benchmark::State& state) {
    using Op = rpp::ops::STInplaceFma<rb::Strategy>;
    rb::run_tensor_benchmark<Op>(
        state,
        [](Op const& op, rb::Context const& ctx, rb::TensorCase& test_case) {
            auto out = test_case.out_tensor();
            auto b = test_case.b_tensor();
            auto c = test_case.c_tensor();
            op(ctx, out, b, c, rb::Scalar{0}, rb::Scalar{1});
        });
}

BENCHMARK(BM_STInplaceFma)->Apply(rb::apply_configs);

} // namespace
