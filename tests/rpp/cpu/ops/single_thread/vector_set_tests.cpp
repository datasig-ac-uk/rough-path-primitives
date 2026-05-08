#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/operations/single_thread/basic/vector_set_constant.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"

namespace {

TEST(VectorSetConstantWrapperTests, MatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const value = Wrapper::make_scalar({{{{'v', 1}}, 5, 3}});

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;

    rpp::cpu::single_thread::vector_set_constant_kernel(
        Wrapper::vector_batch(actual, basis),
        basis,
        strategy,
        Wrapper::tensor_count,
        value
    );
    Wrapper::apply_direct<rpp::ops::VectorSetConstant<Wrapper::Strategy>>(
        basis,
        [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto vec = Wrapper::vector_view(expected, basis, tensor_idx);
            op(ctx, vec, value);
        }
    );

    EXPECT_EQ(actual, expected);
}

} // namespace
