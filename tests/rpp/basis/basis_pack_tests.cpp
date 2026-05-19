#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/basis/basis_pack.hpp>
#include <rpp/basis/lie_basis.hpp>
#include <rpp/basis/tensor_basis.hpp>

namespace {

using Degree = rpp::basis::StandardTensorBasis::Degree;
using Index = rpp::basis::StandardTensorBasis::Index;
using TensorBasis = rpp::basis::StandardTensorBasis;

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
    auto const pack = rpp::basis::make_basis_pack(tensor_basis_data.basis);

    auto const& basis = rpp::basis::get_basis(rpp::basis::TensorBasisTag{}, pack);

    EXPECT_EQ(basis.width, Degree{2});
    EXPECT_EQ(basis.depth, Degree{3});
    EXPECT_EQ(basis.degree_begin, tensor_basis_data.basis.degree_begin);
}

TEST(BasisPackTests, RetrievesRoleAnnotatedBases)
{
    auto const input_basis_data = TensorBasisData(2, 2);
    auto const output_basis_data = TensorBasisData(3, 1);

    auto const pack = rpp::basis::make_basis_pack(
        rpp::basis::in(input_basis_data.basis),
        rpp::basis::out(output_basis_data.basis)
    );

    auto const& input_basis = rpp::basis::get_basis(rpp::basis::InputBasisTag<rpp::basis::TensorBasisTag>{}, pack);
    auto const& output_basis = rpp::basis::get_basis(rpp::basis::OutputBasisTag<rpp::basis::TensorBasisTag>{}, pack);

    EXPECT_EQ(input_basis.width, Degree{2});
    EXPECT_EQ(input_basis.depth, Degree{2});
    EXPECT_EQ(output_basis.width, Degree{3});
    EXPECT_EQ(output_basis.depth, Degree{1});
}

TEST(BasisPackTests, RetrievesIndexedBasesOfSameType)
{
    auto const first_basis_data = TensorBasisData(2, 2);
    auto const second_basis_data = TensorBasisData(3, 2);

    auto const pack = rpp::basis::make_basis_pack(
        rpp::basis::idx<0>(first_basis_data.basis),
        rpp::basis::idx<1>(second_basis_data.basis)
    );

    auto const& first_basis = rpp::basis::get_basis(rpp::basis::IndexedBasisTag<0, rpp::basis::TensorBasisTag>{}, pack);
    auto const& second_basis = rpp::basis::get_basis(rpp::basis::IndexedBasisTag<1, rpp::basis::TensorBasisTag>{}, pack);

    EXPECT_EQ(first_basis.width, Degree{2});
    EXPECT_EQ(second_basis.width, Degree{3});
}

TEST(BasisPackTests, NestedAnnotationsPreserveExistingTag)
{
    auto const basis_data = TensorBasisData(2, 3);
    auto const annotated_basis = rpp::basis::in(rpp::basis::idx<1>(basis_data.basis));

    static_assert(std::is_same_v<
        typename decltype(annotated_basis)::BasisTag,
        rpp::basis::InputBasisTag<rpp::basis::IndexedBasisTag<1, rpp::basis::TensorBasisTag>>
    >);

    auto const pack = rpp::basis::make_basis_pack(annotated_basis);
    auto const& basis = rpp::basis::get_basis(
        rpp::basis::InputBasisTag<rpp::basis::IndexedBasisTag<1, rpp::basis::TensorBasisTag>>{},
        pack
    );

    EXPECT_EQ(basis.width, Degree{2});
    EXPECT_EQ(basis.depth, Degree{3});
}

TEST(BasisPackTests, BoundBasisBypassesPackLookup)
{
    auto const basis_data = TensorBasisData(4, 1);

    auto const& basis = rpp::basis::get_basis(rpp::basis::InputBasisTag<rpp::basis::TensorBasisTag>{}, basis_data.basis);

    EXPECT_EQ(basis.width, Degree{4});
    EXPECT_EQ(basis.depth, Degree{1});
}

TEST(BasisPackTests, UniquenessCheckAllowsDistinctTagsAndRejectsDuplicates)
{
    EXPECT_TRUE((rpp::basis::detail::check_unique(rpp::basis::TensorBasisTag{}, rpp::basis::LieBasisTag{})));
    EXPECT_TRUE((rpp::basis::detail::check_unique(
        rpp::basis::InputBasisTag<rpp::basis::TensorBasisTag>{},
        rpp::basis::OutputBasisTag<rpp::basis::TensorBasisTag>{},
        rpp::basis::IndexedBasisTag<1, rpp::basis::TensorBasisTag>{}
    )));
    EXPECT_FALSE((rpp::basis::detail::check_unique(
        rpp::basis::IndexedBasisTag<0, rpp::basis::TensorBasisTag>{},
        rpp::basis::IndexedBasisTag<0, rpp::basis::TensorBasisTag>{}
    )));
}

} // namespace
