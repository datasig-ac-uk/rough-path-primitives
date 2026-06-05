#ifndef RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_ST_OPS_TEST_HELPER_CUH
#define RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_ST_OPS_TEST_HELPER_CUH

#include "gpu_typed_adjoint_test_helper.cuh"

namespace rpp::tests {

template <typename Config>
class TypedGpuShuffleTensorOpTestBase : public TypedGpuAdjointTestBase<Config> {
protected:
    using Base = TypedGpuAdjointTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Degree;
    using typename Base::DegreeRange;
    using typename Base::HostVector;
    using typename Base::Index;
    using Base::contains;
    using Base::full_range;

    [[nodiscard]] static Accum shuffle_product_coefficient(
        Basis const& basis,
        HostVector const& lhs,
        HostVector const& rhs,
        Degree degree,
        Index global_idx,
        DegreeRange lhs_range,
        DegreeRange rhs_range) {
        using Bitmask = std::uint32_t;

        auto const relative_idx = global_idx - basis.start_of_degree(degree);
        typename Base::GpuStrategy::Letter letters
            [Base::GpuStrategy::Architecture::max_depth];
        basis.unpack_index_to_letters(letters, degree, relative_idx);

        auto const mask_count = Bitmask{1} << degree;
        Accum acc{0};
        for (Bitmask mask{0}; mask < mask_count; ++mask) {
            Index lhs_idx;
            Index rhs_idx;
            Degree lhs_deg;
            Degree rhs_deg;
            basis.pack_masked_index(
                letters, degree, mask, lhs_deg, lhs_idx, rhs_deg, rhs_idx);
            if (!contains(lhs_range, lhs_deg) || !contains(rhs_range, rhs_deg)) {
                continue;
            }
            lhs_idx += basis.start_of_degree(lhs_deg);
            rhs_idx += basis.start_of_degree(rhs_deg);
            acc += Accum{lhs[static_cast<std::size_t>(lhs_idx)]} *
                Accum{rhs[static_cast<std::size_t>(rhs_idx)]};
        }
        return acc;
    }

    [[nodiscard]] static HostVector reference_mul(Basis const& basis,
                                                  HostVector const& lhs,
                                                  HostVector const& rhs,
                                                  Accum beta = Accum{1}) {
        return reference_mul(
            basis, lhs, rhs, full_range(basis), full_range(basis),
            full_range(basis), beta);
    }

    [[nodiscard]] static HostVector
    reference_mul(Basis const& basis,
                  HostVector const& lhs,
                  HostVector const& rhs,
                  DegreeRange out_range,
                  DegreeRange lhs_range,
                  DegreeRange rhs_range,
                  Accum beta = Accum{1}) {
        auto result = Base::make_zero_batch(basis);
        for (Degree degree = out_range.min; degree <= out_range.max; ++degree) {
            auto const begin = basis.start_of_degree(degree);
            auto const end = basis.end_of_degree(degree);
            for (Index idx = begin; idx < end; ++idx) {
                result[static_cast<std::size_t>(idx)] = Base::scalar_from_accum(
                    beta * shuffle_product_coefficient(
                               basis, lhs, rhs, degree, idx, lhs_range, rhs_range));
            }
        }
        return result;
    }

    [[nodiscard]] static HostVector reference_fma(Basis const& basis,
                                                  HostVector const& a,
                                                  HostVector const& b,
                                                  HostVector const& c,
                                                  Accum alpha = Accum{1},
                                                  Accum beta = Accum{1}) {
        return reference_fma(
            basis, a, b, c, full_range(basis), full_range(basis),
            full_range(basis), full_range(basis), alpha, beta);
    }

    [[nodiscard]] static HostVector
    reference_fma(Basis const& basis,
                  HostVector const& a,
                  HostVector const& b,
                  HostVector const& c,
                  DegreeRange out_range,
                  DegreeRange a_range,
                  DegreeRange b_range,
                  DegreeRange c_range,
                  Accum alpha = Accum{1},
                  Accum beta = Accum{1}) {
        auto result = Base::make_zero_batch(basis);
        for (Degree degree = out_range.min; degree <= out_range.max; ++degree) {
            auto const begin = basis.start_of_degree(degree);
            auto const end = basis.end_of_degree(degree);
            for (Index idx = begin; idx < end; ++idx) {
                Accum acc{0};
                if (contains(a_range, degree)) {
                    acc += alpha * Accum{a[static_cast<std::size_t>(idx)]};
                }
                acc += beta * shuffle_product_coefficient(
                    basis, b, c, degree, idx, b_range, c_range);
                result[static_cast<std::size_t>(idx)] = Base::scalar_from_accum(acc);
            }
        }
        return result;
    }
};

} // namespace rpp::tests

#endif // RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_ST_OPS_TEST_HELPER_CUH
