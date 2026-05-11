#ifndef RPP_TESTS_CPU_OPS_SINGLE_THREAD_POLYNOMIAL_TENSOR_HELPER_HPP
#define RPP_TESTS_CPU_OPS_SINGLE_THREAD_POLYNOMIAL_TENSOR_HELPER_HPP

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <utility>
#include <vector>

#include <libalgebra_lite/coefficients.h>
#include <libalgebra_lite/polynomial.h>

#include <rpp/basis/tensor_basis.hpp>
#include <rpp/cpu/strategies.hpp>
#include <rpp/dense/views.hpp>

namespace rpp::tests {

struct PolynomialTensorHelper {
    using Scalar = typename lal::polynomial_ring::scalar_type;
    using Monomial = typename Scalar::key_type;
    using Indeterminate = typename Monomial::letter_type;
    using Rational = typename Scalar::scalar_type;

    using Basis = StandardTensorBasis;
    using Degree = typename Basis::Degree;
    using Index = typename Basis::Index;

    struct TestArchitecture {
        using Degree = PolynomialTensorHelper::Degree;
        using Index = PolynomialTensorHelper::Index;
        using Letter = std::uint8_t;
        using Bitmask = std::uint32_t;

        static constexpr unsigned max_depth = 16;
    };

    using Strategy = cpu::strategies::SingleThreadStrategy<Scalar, TestArchitecture>;
    using Context = typename Strategy::Context;

    template <typename Ptr>
    using VectorView = dense::DenseGradedVectorView<Ptr, Basis>;

    template <typename Ptr>
    using TensorView = dense::DenseTensorView<Ptr, Basis>;

    struct BasisData {
        std::vector<Index> degree_begin;
        Basis basis;

        BasisData(Degree width, Degree depth)
            : degree_begin(make_degree_begin(width, depth)),
              basis(width, depth, degree_begin.data())
        {}
    };

    [[nodiscard]] static Context make_context() noexcept
    {
        return Strategy::make_context(nullptr);
    }

    [[nodiscard]] static Monomial make_monomial(char letter, std::size_t position)
    {
        return Monomial(Indeterminate(letter, position));
    }

    [[nodiscard]] static Scalar make_scalar(
        std::initializer_list<
            std::tuple<
                std::initializer_list<std::pair<char, std::size_t>>,
                std::int32_t,
                std::int32_t
            >
        > coeffs
    )
    {
        Scalar result;

        for (auto&& [markers, numerator, denominator] : coeffs) {
            Monomial monomial;
            for (auto&& [marker, position] : markers) {
                monomial *= make_monomial(marker, position);
            }

            result[monomial] = Rational(numerator, denominator);
        }

        return result;
    }

    [[nodiscard]] static std::vector<Index> make_degree_begin(Degree width, Degree depth)
    {
        std::vector<Index> result(static_cast<std::size_t>(depth + 2));
        for (Degree degree = 1; degree <= depth + 1; ++degree) {
            result[degree] = 1 + width * result[degree - 1];
        }
        return result;
    }

    [[nodiscard]] static std::size_t symbol_index(
        Basis const& basis,
        Degree degree,
        Index level_index
    ) noexcept
    {
        return static_cast<std::size_t>(basis.start_of_degree(degree) + level_index);
    }

    [[nodiscard]] static Scalar symbol(
        char marker,
        Basis const& basis,
        Degree degree,
        Index level_index
    )
    {
        return make_scalar({
            {{{marker, symbol_index(basis, degree, level_index)}}, 1, 1}
        });
    }

    [[nodiscard]] static std::vector<Scalar> make_tensor(char marker, Basis const& basis)
    {
        std::vector<Scalar> result;
        result.reserve(static_cast<std::size_t>(basis.size()));

        for_each_index(
            basis,
            [&result, marker, &basis](Degree degree, Index level_index) {
                result.emplace_back(symbol(marker, basis, degree, level_index));
            }
        );

        return result;
    }

    template <typename F>
    static void for_each_index(Basis const& basis, F&& fn)
    {
        for (Degree degree = 0; degree <= basis.depth; ++degree) {
            const auto level_size = basis.size_of_degree(degree);
            for (Index index = 0; index < level_size; ++index) {
                fn(degree, index);
            }
        }
    }

    [[nodiscard]] static std::vector<std::size_t> unpack_level_index(
        Basis const& basis,
        Degree degree,
        Index level_index
    )
    {
        std::vector<std::size_t> word(static_cast<std::size_t>(degree));
        for (Degree pos = degree; pos > 0; --pos) {
            word[static_cast<std::size_t>(pos - 1)] =
                static_cast<std::size_t>(level_index % basis.width);
            level_index /= basis.width;
        }
        return word;
    }

    template <typename It>
    [[nodiscard]] static Index pack_word(Basis const& basis, It begin, It end)
    {
        Index index = 0;
        for (auto it = begin; it != end; ++it) {
            index *= basis.width;
            index += static_cast<Index>(*it);
        }
        return index;
    }

    [[nodiscard]] static Index reverse_index(Basis const& basis, Degree degree, Index index)
    {
        auto word = unpack_level_index(basis, degree, index);
        std::reverse(word.begin(), word.end());
        return pack_word(basis, word.begin(), word.end());
    }
};

} // namespace rpp::tests

#endif // RPP_TESTS_CPU_OPS_SINGLE_THREAD_POLYNOMIAL_TENSOR_HELPER_HPP
