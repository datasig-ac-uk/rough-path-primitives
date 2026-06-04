#ifndef RPP_TESTS_CPU_SINGLE_THREAD_OPS_CPU_TYPED_VECTOR_OPS_TEST_HELPER_HPP
#define RPP_TESTS_CPU_SINGLE_THREAD_OPS_CPU_TYPED_VECTOR_OPS_TEST_HELPER_HPP

#include "cpu_typed_ft_ops_test_helper.hpp"

namespace rpp::tests {

template <typename Config>
class TypedCpuVectorOpTestBase : public TypedCpuFreeTensorOpTestBase<Config> {
protected:
    using Base = TypedCpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Degree;
    using typename Base::DegreeRange;
    using typename Base::Scalar;
    using VectorView = rpp::DenseGradedVectorView<Scalar*, Basis>;
    using ConstVectorView = rpp::DenseGradedVectorView<Scalar const*, Basis>;

    [[nodiscard]] static VectorView mutable_vector_view(std::vector<Scalar>& data,
                                                        Basis const& basis,
                                                        DegreeRange range) {
        return {data.data(), basis, range.min, range.max};
    }

    [[nodiscard]] static ConstVectorView
    const_vector_view(std::vector<Scalar> const& data,
                      Basis const& basis,
                      DegreeRange range) {
        return {data.data(), basis, range.min, range.max};
    }
};

} // namespace rpp::tests

#endif // RPP_TESTS_CPU_SINGLE_THREAD_OPS_CPU_TYPED_VECTOR_OPS_TEST_HELPER_HPP
