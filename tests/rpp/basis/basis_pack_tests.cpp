#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/basis/basis_pack.hpp>
#include <rpp/basis/lie_basis.hpp>
#include <rpp/basis/tensor_basis.hpp>

namespace {

using Degree = rpp::StandardTensorBasis::Degree;
using Index = rpp::StandardTensorBasis::Index;
using TensorBasis = rpp::StandardTensorBasis;

struct TensorBasisData {
    std::vector<Index> degree_begin;
    TensorBasis basis;

    TensorBasisData(Degree width, Degree depth)
        : degree_begin(static_cast<std::size_t>(depth + 2)),
          basis(width, depth, degree_begin.data())
    {
        degree_begin[0] = 0;
        for (Degree d = 1; d <= depth + 1; ++d) {
            degree_begin[static_cast<std::size_t>(d)] =
                Index{1} + static_cast<Index>(width) * degree_begin[static_cast<std::size_t>(d - 1)];
        }
    }
};

TEST(BasisPackTests, RetrievesPlainBasisByExactTag)
{
    auto const tensor_basis_data = TensorBasisData(2, 3);
    auto const pack = rpp::make_basis_pack(tensor_basis_data.basis);

    auto const& basis = rpp::get_basis(rpp::TensorBasisTag{}, pack);

    EXPECT_EQ(basis.width, Degree{2});
    EXPECT_EQ(basis.depth, Degree{3});
    EXPECT_EQ(basis.degree_begin, tensor_basis_data.basis.degree_begin);
}

TEST(BasisPackTests, RetrievesRoleAnnotatedBases)
{
    auto const input_basis_data = TensorBasisData(2, 2);
    auto const output_basis_data = TensorBasisData(3, 1);

    auto const pack = rpp::make_basis_pack(
        rpp::basis::in(input_basis_data.basis),
        rpp::basis::out(output_basis_data.basis)
    );

    auto const& input_basis = rpp::get_basis(rpp::InputBasisTag<rpp::TensorBasisTag>{}, pack);
    auto const& output_basis = rpp::get_basis(rpp::OutputBasisTag<rpp::TensorBasisTag>{}, pack);

    EXPECT_EQ(input_basis.width, Degree{2});
    EXPECT_EQ(input_basis.depth, Degree{2});
    EXPECT_EQ(output_basis.width, Degree{3});
    EXPECT_EQ(output_basis.depth, Degree{1});
}

TEST(BasisPackTests, RetrievesIndexedBasesOfSameType)
{
    auto const first_basis_data = TensorBasisData(2, 2);
    auto const second_basis_data = TensorBasisData(3, 2);

    auto const pack = rpp::make_basis_pack(
        rpp::basis::idx<0>(first_basis_data.basis),
        rpp::basis::idx<1>(second_basis_data.basis)
    );

    auto const& first_basis = rpp::get_basis(rpp::IndexedBasisTag<0, rpp::TensorBasisTag>{}, pack);
    auto const& second_basis = rpp::get_basis(rpp::IndexedBasisTag<1, rpp::TensorBasisTag>{}, pack);

    EXPECT_EQ(first_basis.width, Degree{2});
    EXPECT_EQ(second_basis.width, Degree{3});
}

TEST(BasisPackTests, NestedAnnotationsPreserveExistingTag)
{
    auto const basis_data = TensorBasisData(2, 3);
    auto const annotated_basis = rpp::basis::in(rpp::basis::idx<1>(basis_data.basis));

    static_assert(std::is_same_v<
        typename decltype(annotated_basis)::BasisTag,
        rpp::InputBasisTag<rpp::IndexedBasisTag<1, rpp::TensorBasisTag>>
    >);

    auto const pack = rpp::make_basis_pack(annotated_basis);
    auto const& basis = rpp::get_basis(
        rpp::InputBasisTag<rpp::IndexedBasisTag<1, rpp::TensorBasisTag>>{},
        pack
    );

    EXPECT_EQ(basis.width, Degree{2});
    EXPECT_EQ(basis.depth, Degree{3});
}

TEST(BasisPackTests, BoundBasisBypassesPackLookup)
{
    auto const basis_data = TensorBasisData(4, 1);

    auto const& basis = rpp::get_basis(rpp::InputBasisTag<rpp::TensorBasisTag>{}, basis_data.basis);

    EXPECT_EQ(basis.width, Degree{4});
    EXPECT_EQ(basis.depth, Degree{1});
}

TEST(BasisPackTests, UniquenessCheckAllowsDistinctTagsAndRejectsDuplicates)
{
    EXPECT_TRUE((rpp::detail::check_unique(rpp::TensorBasisTag{}, rpp::LieBasisTag{})));
    EXPECT_TRUE((rpp::detail::check_unique(
        rpp::InputBasisTag<rpp::TensorBasisTag>{},
        rpp::OutputBasisTag<rpp::TensorBasisTag>{},
        rpp::IndexedBasisTag<1, rpp::TensorBasisTag>{}
    )));
    EXPECT_FALSE((rpp::detail::check_unique(
        rpp::IndexedBasisTag<0, rpp::TensorBasisTag>{},
        rpp::IndexedBasisTag<0, rpp::TensorBasisTag>{}
    )));
}

} // namespace
