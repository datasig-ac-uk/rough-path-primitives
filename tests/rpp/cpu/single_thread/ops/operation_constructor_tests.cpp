#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/basis/tensor_basis.hpp>
#include <rpp/operations/basic/ft_mul.hpp>
#include <rpp/operations/basic/lie_to_tensor.hpp>
#include <rpp/operations/basic/st_mul.hpp>
#include <rpp/operations/basic/tensor_to_lie.hpp>
#include <rpp/operations/intermediate/ft_exp.hpp>
#include <rpp/operations/intermediate/ft_fmexp.hpp>
#include <rpp/operations/intermediate/ft_log.hpp>

namespace rpp::tests {

struct OperationConstruction {
    int operation;
    void const* context;
    void const* bases;
};

struct ConstructorTestStrategy {
    struct Context {
        std::vector<OperationConstruction>* constructions;
    };
    using Accum = double;
    using Index = std::ptrdiff_t;
    using Degree = int;
};

// These leaves deliberately require construction arguments, and record their
// identities so a composite cannot silently substitute a context or basis pack.
template <int Id>
class ConstructorTestLeaf : public ops::BaseOperation<ConstructorTestStrategy> {
public:
    static constexpr bool is_implemented = true;

    template <typename BasisPack>
    explicit ConstructorTestLeaf(Context const& ctx, BasisPack const& pack)
        : BaseOperation(ctx, pack) {
        ctx.constructions->push_back({Id, &ctx, &pack});
    }
};

} // namespace rpp::tests

namespace rpp::ops {

template <>
class FTInplaceMul<tests::ConstructorTestStrategy>
    : public tests::ConstructorTestLeaf<1> {
public:
    using ConstructorTestLeaf::ConstructorTestLeaf;
};

template <>
class TensorSetIdentity<tests::ConstructorTestStrategy>
    : public tests::ConstructorTestLeaf<2> {
public:
    using ConstructorTestLeaf::ConstructorTestLeaf;
};

template <>
class TensorAddIdentity<tests::ConstructorTestStrategy>
    : public tests::ConstructorTestLeaf<3> {
public:
    using ConstructorTestLeaf::ConstructorTestLeaf;
};

template <>
class VectorSetConstant<tests::ConstructorTestStrategy>
    : public tests::ConstructorTestLeaf<4> {
public:
    using ConstructorTestLeaf::ConstructorTestLeaf;
};

template <>
class VectorAssign<tests::ConstructorTestStrategy>
    : public tests::ConstructorTestLeaf<5> {
public:
    using ConstructorTestLeaf::ConstructorTestLeaf;
};

template <>
class FTInplaceFma<tests::ConstructorTestStrategy,
                   FTInplaceFMAType::AEqualsABPlusC>
    : public tests::ConstructorTestLeaf<6> {
public:
    using ConstructorTestLeaf::ConstructorTestLeaf;
};

template <>
class FTInplaceFma<tests::ConstructorTestStrategy,
                   FTInplaceFMAType::AEqualsBCPlusA>
    : public tests::ConstructorTestLeaf<7> {
public:
    using ConstructorTestLeaf::ConstructorTestLeaf;
};

template <>
class STInplaceFma<tests::ConstructorTestStrategy>
    : public tests::ConstructorTestLeaf<8> {
public:
    using ConstructorTestLeaf::ConstructorTestLeaf;
};

template <>
class SparseMatrixVectorProduct<tests::ConstructorTestStrategy,
                                sparse::MatrixFormat::CSR>
    : public tests::ConstructorTestLeaf<9> {
public:
    using ConstructorTestLeaf::ConstructorTestLeaf;
};

template <>
class SparseMatrixVectorProduct<tests::ConstructorTestStrategy,
                                sparse::MatrixFormat::CSC>
    : public tests::ConstructorTestLeaf<10> {
public:
    using ConstructorTestLeaf::ConstructorTestLeaf;
};

} // namespace rpp::ops

namespace {

template <typename Op>
void expect_construction(std::initializer_list<int> expected) {
    static_assert(!std::is_default_constructible_v<Op>);
    std::vector<rpp::tests::OperationConstruction> constructions;
    const rpp::tests::ConstructorTestStrategy::Context ctx{&constructions};
    const rpp::basis::StandardTensorBasis::Index starts[] = {0, 1, 3};
    const rpp::basis::StandardTensorBasis basis(2, 1, starts);
    const auto pack = rpp::basis::make_basis_pack(basis);
    const Op operation(ctx, pack);
    (void)operation;

    ASSERT_EQ(constructions.size(), expected.size());
    auto expected_id = expected.begin();
    for (auto const& construction : constructions) {
        EXPECT_EQ(construction.operation, *expected_id++);
        EXPECT_EQ(construction.context, &ctx);
        EXPECT_EQ(construction.bases, &pack);
    }
}

TEST(OperationConstructors, IntermediatesForwardContextAndBasesToEveryLeaf) {
    using Strategy = rpp::tests::ConstructorTestStrategy;
    expect_construction<rpp::ops::FTExp<Strategy>>({1, 2, 3});
    expect_construction<rpp::ops::FTLog<Strategy>>({4, 1, 3});
    expect_construction<rpp::ops::FTFMExp<Strategy>>({6, 5});
}

TEST(OperationConstructors,
     MultiplicationAndConversionsConstructTheirImplementation) {
    using Strategy = rpp::tests::ConstructorTestStrategy;
    expect_construction<rpp::ops::FTMul<Strategy>>({7});
    expect_construction<rpp::ops::STMul<Strategy>>({8});
    expect_construction<rpp::ops::LieToTensor<
        Strategy,
        rpp::ops::L2TImplementationType::CSRSparseMatrix>>({9});
    expect_construction<rpp::ops::LieToTensor<
        Strategy,
        rpp::ops::L2TImplementationType::CSCSparseMatrix>>({10});
    expect_construction<rpp::ops::TensorToLie<
        Strategy,
        rpp::ops::T2LImplementationType::CSRSparseMatrix>>({9});
    expect_construction<rpp::ops::TensorToLie<
        Strategy,
        rpp::ops::T2LImplementationType::CSCSparseMatrix>>({10});
}

} // namespace
