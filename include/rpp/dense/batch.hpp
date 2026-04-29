#ifndef RPP_DENSE_BATCH_HPP
#define RPP_DENSE_BATCH_HPP

#include <tuple>

#include <rpp/config.h>

#include <rpp/architecture.hpp>
#include <rpp/dense/views.hpp>

namespace rpp::dense {
template<typename It_, typename Stride, typename MinDegree, typename MaxDegree>
class VectorBatch {
    std::tuple<It_, Stride, MinDegree, MaxDegree> data_;

public:

    RPP_HOST_DEVICE
    constexpr VectorBatch(It_ it, Stride stride, MinDegree min, MaxDegree max) noexcept
        : data_(it, stride, min, max)
    {
    }

    RPP_HOST_DEVICE constexpr decltype(auto) data() const noexcept {
        return std::get<0>(data_);
    }

    RPP_HOST_DEVICE constexpr decltype(auto) stride() const noexcept {
        return std::get<1>(data_);
    }

    RPP_HOST_DEVICE constexpr decltype(auto) min_degree() const noexcept {
        return std::get<2>(data_);
    }

    RPP_HOST_DEVICE constexpr decltype(auto) max_degree() const noexcept {
        return std::get<3>(data_);
    }

    template<template<typename, typename> class ViewT, typename Index, typename Basis>
    RPP_HOST_DEVICE constexpr auto view_as(Index index, Basis const &basis) const noexcept {
        using View = ViewT<It_, Basis>;
        using Degree = typename View::Degree;
        return View(data() + index * stride(), basis, static_cast<Degree>(min_degree()),
                    static_cast<Degree>(max_degree()));
    }

    template<template<typename, typename> class ViewT, typename Index, typename Basis>
    RPP_HOST_DEVICE constexpr auto view(Index index, Basis const &basis) const noexcept {
        return view_as<ViewT>(index, basis);
    }

    template<typename Index, typename Basis>
    RPP_HOST_DEVICE constexpr auto view(Index index, Basis const &basis) const noexcept {
        return view_as<DenseVectorView>(index, basis);
    }
};

template <typename It, typename Stride, typename MinDegree, typename MaxDegree>
RPP_HOST_DEVICE
constexpr auto make_vector_batch(It it, Stride stride, MinDegree min, MaxDegree max) noexcept {
    return VectorBatch<It, Stride, MinDegree, MaxDegree>{it, stride, min, max};
}

template <typename It_, typename Stride, typename MinDegree, typename MaxDegree>
class TensorBatch : public VectorBatch<It_, Stride, MinDegree, MaxDegree> {
    using Base = VectorBatch<It_, Stride, MinDegree, MaxDegree>;
public:

    using Base::Base;

    template <typename Index, typename Basis>
    RPP_HOST_DEVICE constexpr auto view(Index index, Basis const &basis) const noexcept {
        return this->template view_as<DenseTensorView>(index, basis);
    }
};


template <typename It, typename Stride, typename MinDegree, typename MaxDegree>
RPP_HOST_DEVICE
constexpr auto make_tensor_batch(It it, Stride stride, MinDegree min, MaxDegree max) noexcept {
    return TensorBatch<It, Stride, MinDegree, MaxDegree>{it, stride, min, max};
}




template <typename It_, typename Stride, typename MinDegree, typename MaxDegree>
class LieBatch : public VectorBatch<It_, Stride, MinDegree, MaxDegree> {
    using Base = VectorBatch<It_, Stride, MinDegree, MaxDegree>;
public:

    using Base::Base;

    template <typename Index, typename Basis>
    RPP_HOST_DEVICE constexpr auto view(Index index, Basis const &basis) const noexcept {
        return this->template view_as<DenseLieView>(index, basis);
    }
};

template <typename It, typename Stride, typename MinDegree, typename MaxDegree>
RPP_HOST_DEVICE
constexpr auto make_lie_batch(It it, Stride stride, MinDegree min, MaxDegree max) noexcept {
    return LieBatch<It, Stride, MinDegree, MaxDegree>{it, stride, min, max};
}

} // namespace rpp::dense


#endif //RPP_DENSE_BATCH_HPP
