#ifndef RPP_GPU_OPS_BLOCK_DETAIL_ANTIPODE_HPP
#define RPP_GPU_OPS_BLOCK_DETAIL_ANTIPODE_HPP

#include <rpp/config.h>

namespace rpp::gpu::block {

enum class AntipodeSigningPolicy {
    NoSigning,
    SignByDegree
};

template <AntipodeSigningPolicy Policy>
struct AntipodeSigningPolicyTag {
    static constexpr auto policy = Policy;
};

using DefaultSigningPolicy = AntipodeSigningPolicyTag<AntipodeSigningPolicy::SignByDegree>;
using NoSigningPolicy = AntipodeSigningPolicyTag<AntipodeSigningPolicy::NoSigning>;

template <typename Context, typename TensorOut, typename TensorArg, AntipodeSigningPolicy Policy>
RPP_DEVICE void generalised_antipode(
    Context const& ctx,
    TensorOut& out,
    TensorArg const& arg,
    AntipodeSigningPolicyTag<Policy>&& sign_policy RPP_MAYBE_UNUSED
    ) noexcept {
    auto const& basis = out.basis();

    for (auto elt_idx=ctx.thread_rank(); elt_idx<arg.size(); elt_idx += ctx.num_threads()) {
        const auto degree = basis.degree(elt_idx);
        const auto degree_begin = basis.start_of_degree(degree);
        const auto rev_idx = basis.reverse_index(elt_idx - degree_begin, degree);


        if constexpr (Policy == AntipodeSigningPolicy::SignByDegree) {
            out[rev_idx + degree_begin] = arg[elt_idx] * (degree % 2 == 0 ? 1 : -1);
        } else {
            out[rev_idx + degree_begin] = arg[elt_idx];
        }
    }


}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPS_BLOCK_DETAIL_ANTIPODE_HPP
