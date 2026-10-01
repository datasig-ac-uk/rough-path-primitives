#include <initializer_list>

#include <rpp/gpu/block/operations/basic/left_hs_mul.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

class GpuBlockLeftHalfShuffleTests : public testing::Test,
                                     public rpp::tests::GpuBlockTestHelper {
protected:
    BasisData data{3, 5};
    Basis const& basis = data.basis;
    using Tensor = HostVector<Scalar>;

    Tensor word(std::initializer_list<Index> letters) const {
        Tensor result(basis.size(), Scalar{0});
        Index idx{0};
        for (auto letter : letters) {
            idx = idx * basis.width + letter;
        }
        result[basis.start_of_degree(letters.size()) + idx] = Scalar{1};
        return result;
    }

    Tensor half(Tensor const& lhs,
                Tensor const& rhs,
                Degree out_min = 0,
                Degree out_max = 5,
                Degree lhs_min = 0,
                Degree lhs_max = 5,
                Degree rhs_min = 0,
                Degree rhs_max = 5,
                Scalar beta = Scalar{1}) {
        Tensor initial(basis.size(), Scalar{17});
        DeviceVector<Scalar> out(initial), left(lhs), right(rhs);
        auto const err = rpp::ops::left_hs_mul(
            GpuStrategy{block_size},
            rpp::gpu::DeviceLaunchConfig{},
            rpp::make_tensor_batch(
                device_data(out), basis.size(), out_min, out_max),
            rpp::make_tensor_batch(
                device_data(left), basis.size(), lhs_min, lhs_max),
            rpp::make_tensor_batch(
                device_data(right), basis.size(), rhs_min, rhs_max),
            basis,
            tensor_count,
            beta);
        EXPECT_TRUE(static_cast<bool>(err)) << err.message();
        auto const sync_err = cudaDeviceSynchronize();
        EXPECT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);
        return copy_to_host(out);
    }
};

TEST_F(GpuBlockLeftHalfShuffleTests, EmptyWordRulesAndScalarOnlyOutput) {
    RPP_REQUIRE_CUDA_DEVICE();
    auto const unit = word({});
    auto const f = make_batch(3, basis);
    Tensor zero(basis.size(), Scalar{0});
    auto positive_f = f;
    positive_f[0] = Scalar{0};
    expect_near(half(unit, f), zero, Scalar{0});
    expect_near(half(f, unit), positive_f, Scalar{0});
    auto expected = Tensor(basis.size(), Scalar{17});
    expected[0] = Scalar{0};
    expect_near(half(unit, unit, 0, 0), expected, Scalar{0});
}

TEST_F(GpuBlockLeftHalfShuffleTests,
       ExplicitWordProductsFixOrientationAndMultiplicity) {
    RPP_REQUIRE_CUDA_DEVICE();
    expect_near(half(word({0}), word({1})), word({0, 1}), Scalar{0});
    expect_near(half(word({1}), word({0})), word({1, 0}), Scalar{0});
    auto expected = word({0, 1, 2});
    auto const other = word({0, 2, 1});
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] += other[i];
    }
    expect_near(half(word({0, 1}), word({2})), expected, Scalar{0});
    expected = word({0, 0, 0, 0});
    for (auto& entry : expected) {
        entry *= Scalar{3};
    }
    expect_near(half(word({0, 0}), word({0, 0})), expected, Scalar{0});
}

TEST_F(GpuBlockLeftHalfShuffleTests,
       OneLetterPowersObeyBinomialRuleWithinDegreeViews) {
    RPP_REQUIRE_CUDA_DEVICE();
    // a^m prec a^n = binomial(m+n-1,n) a^(m+n), m >= 1.
    // Exercise every prefix letter, degree, and both scalar-only operand views.
    for (Index letter = 0; letter < basis.width; ++letter) {
        for (Degree m = 0; m <= basis.depth; ++m) {
            for (Degree n = 0; n <= basis.depth - m; ++n) {
                Tensor lhs(basis.size(), Scalar{0}), rhs(lhs),
                    expected(basis.size(), Scalar{17});
                Index lhs_idx{0}, rhs_idx{0}, out_idx{0};
                for (Degree i = 0; i < m; ++i) {
                    lhs_idx = lhs_idx * basis.width + letter;
                }
                for (Degree i = 0; i < n; ++i) {
                    rhs_idx = rhs_idx * basis.width + letter;
                }
                for (Degree i = 0; i < m + n; ++i) {
                    out_idx = out_idx * basis.width + letter;
                }
                lhs[basis.start_of_degree(m) + lhs_idx] = Scalar{1};
                rhs[basis.start_of_degree(n) + rhs_idx] = Scalar{1};
                for (Index i = basis.start_of_degree(m + n);
                     i < basis.end_of_degree(m + n);
                     ++i) {
                    expected[i] = Scalar{0};
                }
                if (m > 0) {
                    int coefficient = 1;
                    for (Degree k = 1; k <= n; ++k) {
                        coefficient = coefficient * (m - 1 + k) / k;
                    }
                    expected[basis.start_of_degree(m + n) + out_idx] =
                        Scalar{2} * coefficient;
                }
                SCOPED_TRACE(testing::Message() << "letter=" << letter
                                                << " m=" << m << " n=" << n);
                expect_near(half(lhs, rhs, m + n, m + n, m, m, n, n, Scalar{2}),
                            expected,
                            Scalar{0});
            }
        }
    }
}

} // namespace
