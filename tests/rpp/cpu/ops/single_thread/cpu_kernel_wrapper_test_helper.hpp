#ifndef RPP_TESTS_CPU_OPS_SINGLE_THREAD_CPU_KERNEL_WRAPPER_TEST_HELPER_HPP
#define RPP_TESTS_CPU_OPS_SINGLE_THREAD_CPU_KERNEL_WRAPPER_TEST_HELPER_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include <rpp/dense/batch.hpp>

#include "polynomial_tensor_helper.hpp"

namespace rpp::tests {

struct CpuKernelWrapperTestHelper : PolynomialTensorHelper {
    static constexpr Degree width = 2;
    static constexpr Degree depth = 3;
    static constexpr Index tensor_count = 2;

    [[nodiscard]] static std::vector<Scalar> make_batch(char marker, Basis const& basis)
    {
        std::vector<Scalar> result(
            static_cast<std::size_t>(tensor_count * basis.size())
        );

        for (Index tensor_idx = 0; tensor_idx < tensor_count; ++tensor_idx) {
            auto tensor = make_tensor(
                static_cast<char>(marker + tensor_idx),
                basis
            );
            auto const offset = static_cast<std::size_t>(tensor_idx * basis.size());
            std::copy(tensor.begin(), tensor.end(), result.begin() + offset);
        }

        return result;
    }

    [[nodiscard]] static TensorView<Scalar*> tensor_view(
        std::vector<Scalar>& data,
        Basis const& basis,
        Index tensor_idx
    )
    {
        return {data.data() + tensor_idx * basis.size(), basis};
    }

    [[nodiscard]] static TensorView<Scalar const*> tensor_view(
        std::vector<Scalar> const& data,
        Basis const& basis,
        Index tensor_idx
    )
    {
        return {data.data() + tensor_idx * basis.size(), basis};
    }

    [[nodiscard]] static VectorView<Scalar*> vector_view(
        std::vector<Scalar>& data,
        Basis const& basis,
        Index tensor_idx
    )
    {
        return {data.data() + tensor_idx * basis.size(), basis};
    }

    [[nodiscard]] static VectorView<Scalar const*> vector_view(
        std::vector<Scalar> const& data,
        Basis const& basis,
        Index tensor_idx
    )
    {
        return {data.data() + tensor_idx * basis.size(), basis};
    }

    [[nodiscard]] static auto tensor_batch(
        std::vector<Scalar>& data,
        Basis const& basis
    )
    {
        return dense::make_tensor_batch(
            data.data(),
            basis.size(),
            Degree{0},
            basis.depth
        );
    }

    [[nodiscard]] static auto tensor_batch(
        std::vector<Scalar> const& data,
        Basis const& basis
    )
    {
        return dense::make_tensor_batch(
            data.data(),
            basis.size(),
            Degree{0},
            basis.depth
        );
    }

    [[nodiscard]] static auto vector_batch(
        std::vector<Scalar>& data,
        Basis const& basis
    )
    {
        return dense::make_vector_batch(
            data.data(),
            basis.size(),
            Degree{0},
            basis.depth,
            typename Basis::Tag{}
        );
    }

    [[nodiscard]] static auto vector_batch(
        std::vector<Scalar> const& data,
        Basis const& basis
    )
    {
        return dense::make_vector_batch(
            data.data(),
            basis.size(),
            Degree{0},
            basis.depth,
            typename Basis::Tag{}
        );
    }

    template <typename Op, typename Fn>
    static void apply_direct(Basis const& basis, Fn&& fn)
    {
        auto const strategy = Strategy{};
        auto const scratch_bytes = Op::scratch_space_size(strategy, basis);
        std::vector<std::byte> scratch(scratch_bytes);
        auto const ctx = Strategy::make_context(scratch.data());

        Op::init_scratch_space(ctx, basis);
        Op op;
        for (Index tensor_idx = 0; tensor_idx < tensor_count; ++tensor_idx) {
            fn(op, ctx, tensor_idx);
        }
        Op::destroy_scratch_space(ctx, basis);
    }
};

} // namespace rpp::tests

#endif // RPP_TESTS_CPU_OPS_SINGLE_THREAD_CPU_KERNEL_WRAPPER_TEST_HELPER_HPP
