#ifndef RPP_BASIS_HALL_BASIS_HPP
#define RPP_BASIS_HALL_BASIS_HPP

#include <vector>
#include <algorithm>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/architecture.hpp>
#include <rpp/basis/lie_basis.hpp>


namespace rpp {

template <typename Architecture=arch::NativeArchitecture>
class HallBasis {
public:
    using Degree = typename Architecture::Degree;
    using Index = typename Architecture::Index;

private:
    std::vector<Index> data_;
    std::vector<Index> degree_begin_;
    Degree width_;
    Degree depth_;

    void grow();
    void emplace_back(Index left, Index right) {
        data_.emplace_back(left);
        data_.emplace_back(right);
    }
public:

    HallBasis(Degree width, Degree depth)
        : width_(width), depth_(depth) {
        degree_begin_ = {0, 1};
        data_.reserve(2*(1+width_));
        data_.emplace_back(0);
        data_.emplace_back(0);

        if (depth > 0) {
            for (Index letter = 1; letter <= width_; ++letter) {
                data_.emplace_back(0);
                data_.emplace_back(letter);
            }
            degree_begin_.emplace_back(static_cast<Index>(1 + width_));
            grow();
        }
    }

    constexpr LieBasis<Architecture> to_lie_basis() const noexcept {
        return LieBasis<Architecture>{width_, depth_, degree_begin_.data(), data_.data() };
    }

    constexpr auto operator[](Index index) const noexcept {
        return std::tie(data_[2*index], data_[2*index+1]);
    }

    Index size() const noexcept {
        return degree_begin_.back();
    }

};


template<typename Architecture>
void HallBasis<Architecture>::grow() {
    auto size = degree_begin_[2];
    for (Degree degree=2; degree<=depth_; ++degree) {
        for (Degree lhs_degree=1; 2*lhs_degree<=degree; ++lhs_degree) {
            const auto right_degree = degree - lhs_degree;
            const auto lbegin = degree_begin_[lhs_degree];
            const auto lend = degree_begin_[lhs_degree+1];
            const auto rbegin = degree_begin_[right_degree];
            const auto rend = degree_begin_[right_degree+1];

            for (auto left_idx=lbegin; left_idx<lend; ++left_idx) {
                for (auto right_idx=std::max(left_idx+1, rbegin); right_idx<rend; ++right_idx) {
                    if (data_[2*right_idx] <= left_idx) {
                        emplace_back(left_idx, right_idx);
                        ++size;
                    }
                }
            }
        }
        degree_begin_.emplace_back(size);
    }
}


} // namespace

#endif //RPP_BASIS_HALL_BASIS_HPP
