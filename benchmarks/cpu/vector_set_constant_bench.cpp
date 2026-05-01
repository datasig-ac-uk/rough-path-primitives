#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_VectorSetConstant(benchmark::State& state)
{
    using Op = rpp::ops::VectorSetConstant<rb::Strategy>;
    rb::run_tensor_benchmark<Op>(state, [](Op const& op, rb::Context const& ctx, rb::TensorCase& test_case) {
        auto out = test_case.out_vector();
        op(ctx, out, rb::Scalar{0.125});
    });
}

BENCHMARK(BM_VectorSetConstant)->Apply(rb::apply_configs);

} // namespace
