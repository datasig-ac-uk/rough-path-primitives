#ifndef RPP_BASIS_GRADED_BASIS_HPP
#define RPP_BASIS_GRADED_BASIS_HPP

#include <rpp/architecture.hpp>
#include <rpp/config.h>
#include <rpp/support/tagged_pointer.hpp>

namespace rpp::basis {

template <typename Architecture_, typename Tag_>
struct GradedBasis {
    using Architecture = Architecture_;
    using Tag = Tag_;

    using Degree = typename Architecture::Degree;
    using Index = typename Architecture::Index;
    using DBPtr = typename Architecture::template Ptr<Index const>;

    Degree width;
    Degree depth;
    DBPtr degree_begin;

    GradedBasis(Degree width_,
                Degree depth_,
                Index const* degree_begin_) noexcept
        : width(width_), depth(depth_), degree_begin(degree_begin_) {}

    GradedBasis(Degree width_, Degree depth_, DBPtr degree_begin_) noexcept
        : width(width_), depth(depth_), degree_begin(degree_begin_) {}

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Index size() const noexcept {
        return degree_begin[depth + 1];
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Index true_size() const noexcept {
        return size();
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Index
    start_of_degree(Degree d) const noexcept {
        return degree_begin[d];
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Index
    end_of_degree(Degree d) const noexcept {
        return degree_begin[d + 1];
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Index
    size_of_degree(Degree d) const noexcept {
        return degree_begin[d + 1] - degree_begin[d];
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Degree
    degree(Index idx) const noexcept {
        Degree diff = this->depth + 1;
        Degree pos = 0;
        while (diff > 0) {
            const Degree half = diff / 2;
            const Degree new_pos = pos + half;

            if (this->degree_begin[new_pos] <= idx) {
                pos = new_pos + 1;
                diff -= half + 1;
            }
            else {
                diff = half;
            }
        }
        return pos - 1;
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Degree
    degree_linear(Index idx) const noexcept {
        Degree result = 0;
        while (result <= depth && degree_begin[result] <= idx) {
            ++result;
        }
        return result - 1;
    }

    template <typename DataMapper>
    RPP_NODISCARD friend typename DataMapper::template Result<
        GradedBasis<typename DataMapper::Architecture, Tag_>>
    map_data(GradedBasis const& basis, DataMapper& mapper) noexcept {
        if constexpr (std::is_same_v<Architecture_,
                                     typename DataMapper::Architecture>) {
            return basis;
        }
        else {
            using TgtIndex = typename DataMapper::Architecture::Index;

            auto mapped_db = mapper.template copy_n<TgtIndex>(
                basis.degree_begin, basis.depth + 2);
            if (!mapped_db) {
                return std::move(mapped_db).error();
            }

            return GradedBasis<typename DataMapper::Architecture, Tag_>{
                mapped_db.width, mapped_db.depth, mapped_db.value()};
        }
    }
};

} // namespace rpp::basis

#endif // RPP_BASIS_GRADED_BASIS_HPP
