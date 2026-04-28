#ifndef RPP_CPU_OPS_SINGLE_THREAD_DETAIL_GENERALISED_ANTIPODE_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_DETAIL_GENERALISED_ANTIPODE_HPP

#include <rpp/config.h>

namespace rpp::cpu::single_thread {
enum class AntipodeSigningPolicy {
    NoSigning,
    SignByDegree
};

template<AntipodeSigningPolicy Policy>
struct AntipodeSigningPolicyTag {
    static constexpr auto policy = Policy;
};

using DefaultSigningPolicy = AntipodeSigningPolicyTag<AntipodeSigningPolicy::SignByDegree>;
using NoSigningPolicy = AntipodeSigningPolicyTag<AntipodeSigningPolicy::NoSigning>;


template<typename Context, typename TensorOut, typename TensorArg, AntipodeSigningPolicy Policy>
void generalised_antipode(
    Context const &ctx,
    TensorOut &out,
    TensorArg const &arg,
    AntipodeSigningPolicyTag<Policy> policy_tag RPP_MAYBE_UNUSED
) noexcept {
    using Index = typename Context::Strategy::Index;
    const auto min_degree = std::max(out.min_degree(), arg.min_degree());
    const auto max_degree = std::min(out.max_degree(), arg.max_degree());

    if (min_degree == 0) {
        out[0] = arg[0];
    }

    if (min_degree <= 1) {
        auto out_view = out.degree_view(1);
        const auto arg_view = arg.degree_view(1);

        const auto size = arg_view.size();
        for (Index i = 0; i < size; ++i) {
            if constexpr (Policy == AntipodeSigningPolicy::SignByDegree) {
                out_view[i] = -arg_view[i];
            } else {
                out_view[i] = arg_view[i];
            }
        }
    }

    for (auto degree = std::max(2, min_degree); degree <= max_degree; ++degree) {
        auto out_view = out.degree_view(degree);
        const auto arg_view = arg.degree_view(degree);

        for (Index i = 0; i < arg_view.size(); ++i) {
            if constexpr (Policy == AntipodeSigningPolicy::SignByDegree) {
                out_view[i] = arg_view[i] * (degree % 2 == 0 ? 1 : -1) ;
            } else {
                out_view[i] = arg_view[i];
            }
        }
    }
}
}


#endif //RPP_CPU_OPS_SINGLE_THREAD_DETAIL_GENERALISED_ANTIPODE_HPP
