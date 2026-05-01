#include "benchmark_helpers.hpp"

namespace {
namespace rb = rpp::benchmarks::cpu;

void BM_VectorInplaceAdd(benchmark::State& state)
{
    using Op = rpp::ops::VectorInplaceAdd<rb::Strategy>;
    rb::run_tensor_benchmark<Op>(state, [](Op const& op, rb::Context const& ctx, rb::TensorCase& test_case) {
        auto out = test_case.out_vector();
        auto arg = rb::ConstVectorView(test_case.zero.data(), test_case.basis);
        op(ctx, out, arg);
    });
}

BENCHMARK(BM_VectorInplaceAdd)->Apply(rb::apply_configs);

} // namespace
