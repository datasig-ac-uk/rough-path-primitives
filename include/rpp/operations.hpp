#ifndef RPP_OPERATIONS_HPP
#define RPP_OPERATIONS_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/utility.hpp>

namespace rpp::ops {


/*****************************************************************************
 *                           Vector Operations                               *
 *****************************************************************************/

template <typename Strategy>
class VectorSetConstant {
    using Context = typename Strategy::Context;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename Vector, typename Value>
    void operator()(Context const& ctx, Vector& vec, Value const& value) const noexcept;
};

template <typename Strategy>
class VectorSetZero {
    using Context = typename Strategy::Context;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename Vector>
    void operator()(Context const& ctx, Vector& vec) const noexcept;
};

template <typename Strategy>
class VectorAssign {
    using Context = typename Strategy::Context;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename VectorOut, typename VectorArg>
    void operator()(Context const& ctx, VectorOut& out, VectorArg const& arg) const noexcept;
};

template <typename Strategy>
class VectorInplaceAdd {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename VectorLhs, typename VectorRhs>
    void operator()(Context const& ctx, VectorLhs& lhs, VectorRhs const& rhs, Accum alpha=Accum{1}) const noexcept;


};


/*****************************************************************************
 *                        Elementary Tensor Operations                       *
 *****************************************************************************/

template <typename Strategy>
class TensorAddIdentity {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename Tensor>
    void operator()(Context const& ctx, Tensor& tensor, Accum scalar=Accum{1}) const noexcept;
};

template <typename Strategy>
class TensorSetIdentity {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename Tensor>
    void operator()(Context const& ctx, Tensor& tensor, Accum scalar=Accum{1}) const noexcept;
};





/*****************************************************************************
 *                        Basic Algebra Operations                           *
 *****************************************************************************/


template <typename Strategy>
class TensorAntipode {
    using Context = typename Strategy::Context;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorArg>
    void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept;
};

template <typename Strategy>
class TensorReflect {
    using Context = typename Strategy::Context;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorArg>
    void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept;
};


enum class InplaceFMAType {
    AEqualsBCPlusA, // a <- b*c + a
    AEqualsABPlusC, // a <- a*b + c
    AEqualsBAPlusC  // a <- b*a + c
};


template <typename Strategy, InplaceFMAType FMAType>
class FTInplaceFma {
    using Context = typename Strategy::Context;

        using Accum = typename Strategy::Accum;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    void operator()(Context const& ctx, TensorA& a, TensorB const& b, TensorC const& c, Accum alpha=Accum{1}, Accum beta=Accum{1}) const noexcept;
};

template <typename Strategy>
using FTInplaceFma231 = FTInplaceFma<Strategy, InplaceFMAType::AEqualsBCPlusA>;
template <typename Strategy>
using FTInplaceFma123 = FTInplaceFma<Strategy, InplaceFMAType::AEqualsABPlusC>;
template <typename Strategy>
using FTInplaceFma213 = FTInplaceFma<Strategy, InplaceFMAType::AEqualsABPlusC>;

template <typename Strategy>
class FTFma {
    using Context = typename Strategy::Context;

        using Accum = typename Strategy::Accum;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorA, typename TensorB, typename TensorC>
    void operator()(Context const& ctx, TensorOut& out, TensorA const& a, TensorB const& b, TensorC const& c, Accum alpha=Accum{1}, Accum beta=Accum{1}) const noexcept;

};

template <typename Strategy>
class FTMul {
    using Context = typename Strategy::Context;

        using Accum = typename Strategy::Accum;

    using FMA = FTInplaceFma231<Strategy>;

    FMA fma;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        return FMA::scratch_space_size(config, basis);
    }

    template <typename TensorOut, typename TensorLhs, typename TensorRhs>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, TensorOut& out, TensorLhs const& lhs, TensorRhs& rhs, Accum beta=Accum{1}) const noexcept {
        fma(ctx, out, lhs. rhs, Accum{0}, beta);
    }

};

template <typename Strategy>
class FTInplaceMul {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorLhs, typename TensorRhs>
    void operator()(Context const& ctx, TensorLhs& lhs, TensorRhs const& rhs, Accum beta=Accum{1}) const noexcept;
};

template <typename Strategy>
class FTAdjLMul {
    using Context = typename Strategy::Context;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    void operator()(Context const& ctx, TensorOut& out, TensorOp const& op, TensorArg const& arg) const noexcept;
};

template <typename Strategy>
class FTAdjRMul {
    using Context = typename Strategy::Context;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    void operator()(Context const& ctx, TensorOut& out, TensorOp const& op, TensorArg const& arg) const noexcept;
};


template <typename Strategy> class STFma {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorA, typename TensorB, typename TensorC>
    void operator()(Context const& ctx, TensorOut& out, TensorA const& a, TensorB const& b, TensorC const& c, Accum alpha=Accum{1}, Accum beta=Accum{1}) const noexcept;
};

template <typename Strategy> class STInplaceFma {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    void operator()(Context const& ctx, TensorA& a, TensorB const& b, TensorC const& c, Accum alpha=Accum{1}, Accum beta=Accum{1}) const noexcept;
};

template <typename Strategy>
class STMul {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    using InplaceFMA = STInplaceFma<Strategy>;
    InplaceFMA fma;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        return InplaceFMA::scratch_space_size(config, basis);
    }

    template <typename TensorOut, typename TensorLhs, typename TensorRhs>
    void operator()(Context const& ctx, TensorOut& out, TensorLhs const& lhs, TensorRhs const& rhs, Accum beta=Accum{1}) const noexcept {
        return fma(ctx, out, lhs, rhs, Accum { 0 }, beta);
    }
};



template <typename Strategy>
class STAdjMul {
    using Context = typename Strategy::Context;
public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    void operator()(Context const& ctx, TensorOut& out, TensorOp const& op, TensorArg const& arg) const noexcept;
};


/*****************************************************************************
 *                    Intermediate Algebra Operations                        *
 *****************************************************************************/

template <typename Strategy>
class FTExp {
    using Context = typename Strategy::Context;
        using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;

    using InplaceMul = FTInplaceMul<Strategy>;
    using SetIdentity = TensorSetIdentity<Strategy>;
    using AddIdentity = TensorAddIdentity<Strategy>;

    InplaceMul inplace_mul;
    SetIdentity set_identity;
    AddIdentity add_identity;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        return std::max(InplaceMul::scratch_space_size(config, basis),
            std::max(SetIdentity::scratch_space_size(config, basis),
                        AddIdentity::scratch_space_size(config, basis)));
    }

    template <typename TensorOut, typename TensorArg>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        constexpr Accum one { 1 };

        set_identity(ctx, out);

        for (Degree d = basis.depth; d > 0; --d) {
            const auto max_degree = basis.depth - d + 1;
            const Accum divisor = one / d;

            ctx.sync();

            inplace_mul(ctx, out, arg.truncate(1, max_degree), divisor);

            ctx.sync();
            add_identity(ctx, out);
        }
    }
};

template <typename Strategy>
class FTFMExp {
    using Context = typename Strategy::Context;

        using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;


    using InplaceFMA123 = FTInplaceFma<Strategy, InplaceFMAType::AEqualsABPlusC>;
    using Assign = VectorAssign<Strategy>;

    InplaceFMA123 inplace_fma123;
    Assign assign;
public:

    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        return std::max(
            InplaceFMA123 ::scratch_space_size(config, basis),
            Assign::scratch_space_size(config, basis));
    }

    template <typename TensorOut, typename TensorMultiplier, typename TensorExponent>
    void operator()(Context const& ctx, TensorOut& out, TensorMultiplier const& multiplier, TensorExponent const& exponent) const noexcept {
        auto const& basis = out.basis();
        constexpr Accum one { 1 };

        assign(ctx, out, multiplier);

        for (Degree d =basis.depth; d > 0; --d) {
            const auto max_degree = basis.depth - d + 1;
            const Accum divisor = one / d;

            ctx.sync();

            inplace_fma123(ctx, out, exponent.truncate(1, max_degree), multiplier, one, divisor);
        }
    }
};

template <typename Strategy>
class FTLog {
    using Context = typename Strategy::Context;

        using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;

    using SetZero =VectorSetZero<Strategy>;
    using InplaceMul = FTInplaceMul<Strategy>;
    using AddIdentity = TensorAddIdentity<Strategy>;

    SetZero set_zero;
    InplaceMul inplace_mul;
    AddIdentity add_identity;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        return std::max(
            SetZero::scratch_space_size(config, basis),
            std::max(SetZero::scratch_space_size(config, basis),
                AddIdentity::scratch_space_size(config, basis))
            );

    }

    template <typename TensorOut, typename TensorArg>
    void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        constexpr Accum one { 1 };

        set_zero(ctx, out);

        for (Degree d=basis.depth; d > 0; --d) {
            const Accum val = (d % 2 == 0 ? -one : one) / d;

            add_identity(ctx, out, val);

            ctx.sync();

            inplace_mul(ctx, out, arg.truncate(1, d), val);
        }
    }
};


} // namespace rpp


#endif // RPP_OPERATIONS_HPP
