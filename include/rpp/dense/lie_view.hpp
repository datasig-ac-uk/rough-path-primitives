#ifndef RPP_DENSE_LIE_VIEW_HPP
#define RPP_DENSE_LIE_VIEW_HPP

#include <algorithm>

#include <rpp/config.h>

#include <rpp/dense/graded_vector_view.hpp>

namespace rpp::dense {

template <typename It, typename Basis>
class DenseLieView : public DenseGradedVectorView<It, Basis> {
    using Base = DenseGradedVectorView<It, Basis>;

public:
    using Base::Base;
    using typename Base::Degree;

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr DenseLieView truncate(Degree min_degree, Degree max_degree) const noexcept {
        return {
            this->data(),
            this->basis(),
            std::max(min_degree, this->min_degree()),
            std::min(max_degree, this->max_degree())
        };
    }
};

} // namespace rpp::dense

#endif // RPP_DENSE_LIE_VIEW_HPP
