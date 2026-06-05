#ifndef RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_VECTOR_OPS_TEST_HELPER_CUH
#define RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_VECTOR_OPS_TEST_HELPER_CUH

#include <algorithm>

#include "gpu_typed_adjoint_test_helper.cuh"

namespace rpp::tests {

template <typename Config>
class TypedGpuVectorOpTestBase : public TypedGpuAdjointTestBase<Config> {
protected:
    using Base = TypedGpuAdjointTestBase<Config>;

    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Degree;
    using typename Base::DeviceVector;
    using typename Base::DegreeRange;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using Base::full_range;
    using Base::is_empty;
    using Base::overlap_range;
    using Base::scalar_from_accum;

    static HostVector linear_combo(HostVector const& lhs,
                                   Accum lhs_scale,
                                   HostVector const& rhs,
                                   Accum rhs_scale) {
        HostVector result(lhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            result[i] = scalar_from_accum(lhs_scale * static_cast<Accum>(lhs[i]) +
                                          rhs_scale * static_cast<Accum>(rhs[i]));
        }
        return result;
    }

    static HostVector assign_slice(HostVector result,
                                   Basis const& basis,
                                   DegreeRange range,
                                   typename Base::Scalar value) {
        for (auto idx = basis.start_of_degree(range.min);
             idx < basis.end_of_degree(range.max);
             ++idx) {
            result[static_cast<std::size_t>(idx)] = value;
        }
        return result;
    }
};

} // namespace rpp::tests

#endif // RPP_TESTS_GPU_BLOCK_OPS_GPU_TYPED_VECTOR_OPS_TEST_HELPER_CUH
