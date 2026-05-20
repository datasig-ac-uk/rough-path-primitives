#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/st_fma.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class ShuffleTensorFmaTests
    : public testing::Test,
      public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;
};

TEST_F(ShuffleTensorFmaTests, AccumulatesShuffleProduct)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto out = make_tensor('c', basis);
    auto const lhs = make_tensor('a', basis);
    auto const rhs = make_tensor('b', basis);

    TensorView<Scalar*> out_view(out.data(), basis);
    TensorView<Scalar const*> lhs_view(lhs.data(), basis);
    TensorView<Scalar const*> rhs_view(rhs.data(), basis);

    auto const ctx = make_context();
    rpp::ops::STFma<Strategy>{}(ctx, out_view, out_view, lhs_view, rhs_view);

    std::vector<Scalar> expected;
    expected.reserve(static_cast<std::size_t>(basis.size()));

    for_each_index(
        basis,
        [&expected, &basis](Degree degree, Index level_index) {
            auto const word = unpack_level_index(basis, degree, level_index);
            auto entry = symbol('c', basis, degree, level_index);
            auto const mask_count = std::uint32_t{1} << degree;

            for (std::uint32_t mask = 0; mask < mask_count; ++mask) {
                std::vector<std::size_t> lhs_word;
                std::vector<std::size_t> rhs_word;
                lhs_word.reserve(word.size());
                rhs_word.reserve(word.size());

                for (Degree i = 0; i < degree; ++i) {
                    if (((mask >> i) & std::uint32_t{1}) != 0) {
                        lhs_word.push_back(word[static_cast<std::size_t>(i)]);
                    } else {
                        rhs_word.push_back(word[static_cast<std::size_t>(i)]);
                    }
                }

                auto const lhs_degree = static_cast<Degree>(lhs_word.size());
                auto const rhs_degree = static_cast<Degree>(rhs_word.size());
                auto const lhs_index = pack_word(basis, lhs_word.begin(), lhs_word.end());
                auto const rhs_index = pack_word(basis, rhs_word.begin(), rhs_word.end());

                entry += make_scalar({
                    {
                        {
                            {'a', symbol_index(basis, lhs_degree, lhs_index)},
                            {'b', symbol_index(basis, rhs_degree, rhs_index)}
                        },
                        1,
                        1
                    }
                });
            }

            expected.emplace_back(std::move(entry));
        }
    );

    EXPECT_EQ(out, expected);
}

TEST_F(ShuffleTensorFmaTests, KernelWrapperMatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const alpha = Wrapper::make_scalar({{{{'p', 1}}, 3, 2}});
    auto const beta = Wrapper::make_scalar({{{{'q', 2}}, 5, 3}});

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto const a = Wrapper::make_batch('a', basis);
    auto const b = Wrapper::make_batch('b', basis);
    auto const c = Wrapper::make_batch('c', basis);

    auto const err = rpp::ops::st_fma(
        strategy,
        typename Wrapper::Strategy::LaunchConfig{},
        Wrapper::tensor_batch(actual, basis),
        Wrapper::tensor_batch(a, basis),
        Wrapper::tensor_batch(b, basis),
        Wrapper::tensor_batch(c, basis),
        basis,
        Wrapper::tensor_count,
        alpha,
        beta
    );
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::STFma<Wrapper::Strategy>>(
        basis,
        [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto addend = Wrapper::tensor_view(a, basis, tensor_idx);
            auto left = Wrapper::tensor_view(b, basis, tensor_idx);
            auto right = Wrapper::tensor_view(c, basis, tensor_idx);
            op(ctx, out, addend, left, right, alpha, beta);
        }
    );

    EXPECT_EQ(actual, expected);
}

} // namespace
