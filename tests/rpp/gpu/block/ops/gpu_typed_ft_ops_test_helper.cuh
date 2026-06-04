#ifndef RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_FT_OPS_TEST_HELPER_CUH
#define RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_FT_OPS_TEST_HELPER_CUH

#include "gpu_typed_adjoint_test_helper.cuh"

namespace rpp::tests {

template <typename Config>
class TypedGpuFreeTensorOpTestBase : public TypedGpuAdjointTestBase<Config> {
protected:
    using Base = TypedGpuAdjointTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Degree;
    using typename Base::HostVector;
    using typename Base::Index;
    using typename Base::Scalar;

    struct DegreeRange {
        Degree min;
        Degree max;
    };

    [[nodiscard]] static constexpr bool contains(DegreeRange range,
                                                 Degree degree) noexcept {
        return range.min <= degree && degree <= range.max;
    }

    [[nodiscard]] static constexpr DegreeRange
    full_range(Basis const& basis) noexcept {
        return DegreeRange{Degree{0}, basis.depth};
    }

    [[nodiscard]] static HostVector make_unit_tensor(Basis const& basis) {
        auto result = Base::make_zero_batch(basis);
        result[0] = cast_scalar<Scalar>(1.0f);
        return result;
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

        for (Degree out_deg = out_range.min; out_deg <= out_range.max; ++out_deg) {
            auto const out_begin = basis.start_of_degree(out_deg);
            auto const out_level_size = basis.size_of_degree(out_deg);

            for (Degree lhs_deg = lhs_range.min;
                 lhs_deg <= std::min(out_deg, lhs_range.max);
                 ++lhs_deg) {
                auto const rhs_deg = static_cast<Degree>(out_deg - lhs_deg);
                if (!contains(rhs_range, rhs_deg)) {
                    continue;
                }

                auto const lhs_begin = basis.start_of_degree(lhs_deg);
                auto const rhs_begin = basis.start_of_degree(rhs_deg);
                auto const lhs_level_size = basis.size_of_degree(lhs_deg);
                auto const rhs_level_size = basis.size_of_degree(rhs_deg);

                for (Index lhs_idx = 0; lhs_idx < lhs_level_size; ++lhs_idx) {
                    auto const lhs_val =
                        Accum{lhs[static_cast<std::size_t>(lhs_begin + lhs_idx)]};
                    auto const out_offset =
                        static_cast<Index>(lhs_idx * rhs_level_size);
                    for (Index rhs_idx = 0; rhs_idx < rhs_level_size; ++rhs_idx) {
                        auto const rhs_val =
                            Accum{rhs[static_cast<std::size_t>(rhs_begin + rhs_idx)]};
                        auto const out_idx = static_cast<Index>(
                            out_begin + out_offset + rhs_idx);
                        auto const old_val =
                            Accum{result[static_cast<std::size_t>(out_idx)]};
                        result[static_cast<std::size_t>(out_idx)] =
                            cast_scalar<Scalar>(old_val + beta * lhs_val * rhs_val);
                    }
                }
            }

            if (out_level_size == 0) {
                break;
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

        for (Degree out_deg = out_range.min; out_deg <= out_range.max; ++out_deg) {
            auto const out_begin = basis.start_of_degree(out_deg);
            auto const out_level_size = basis.size_of_degree(out_deg);

            for (Index out_idx = 0; out_idx < out_level_size; ++out_idx) {
                auto const global_idx =
                    static_cast<Index>(out_begin + out_idx);
                Accum acc{0};
                if (contains(a_range, out_deg)) {
                    acc += alpha * Accum{a[static_cast<std::size_t>(global_idx)]};
                }
                result[static_cast<std::size_t>(global_idx)] =
                    cast_scalar<Scalar>(acc);
            }

            for (Degree b_deg = b_range.min;
                 b_deg <= std::min(out_deg, b_range.max);
                 ++b_deg) {
                auto const c_deg = static_cast<Degree>(out_deg - b_deg);
                if (!contains(c_range, c_deg)) {
                    continue;
                }

                auto const b_begin = basis.start_of_degree(b_deg);
                auto const c_begin = basis.start_of_degree(c_deg);
                auto const b_level_size = basis.size_of_degree(b_deg);
                auto const c_level_size = basis.size_of_degree(c_deg);

                for (Index b_idx = 0; b_idx < b_level_size; ++b_idx) {
                    auto const b_val =
                        Accum{b[static_cast<std::size_t>(b_begin + b_idx)]};
                    auto const out_offset =
                        static_cast<Index>(b_idx * c_level_size);
                    for (Index c_idx = 0; c_idx < c_level_size; ++c_idx) {
                        auto const c_val =
                            Accum{c[static_cast<std::size_t>(c_begin + c_idx)]};
                        auto const global_idx = static_cast<Index>(
                            out_begin + out_offset + c_idx);
                        auto const old_val =
                            Accum{result[static_cast<std::size_t>(global_idx)]};
                        result[static_cast<std::size_t>(global_idx)] = cast_scalar<
                            Scalar>(old_val + beta * b_val * c_val);
                    }
                }
            }

            if (out_level_size == 0) {
                break;
            }
        }

        return result;
    }
};

} // namespace rpp::tests

#endif // RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_FT_OPS_TEST_HELPER_CUH
