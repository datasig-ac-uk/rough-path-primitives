#ifndef RPP_ARCHITECTURE_HPP
#define RPP_ARCHITECTURE_HPP

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>

#include <rpp/support/tagged_pointer.hpp>

namespace rpp {
namespace tags {

template <typename Arch>
struct ArchTag {
    using Architecture = Arch;
    using Index = typename Architecture::Index;
    using Degree = typename Architecture::Degree;
    using difference_type = Index;
    using size_type = typename Architecture::Size;
};


} // namespace tags


namespace arch {
template <typename Size_>
struct Architecture {
    using Size = std::make_unsigned_t<Size_>;
    using Index = std::make_signed_t<Size>;

    using Degree = int32_t;

    template <typename T>
    using Ptr = TaggedPtr<T,
                          tags::ArchTag<Architecture>,
                          tags::LocationTag<tags::HostLocation>>;
};


using NativeArchitecture = Architecture<std::size_t>;
using Architecture32 = Architecture<std::uint32_t>;
using Architecture64 = Architecture<std::uint64_t>;
} // namespace arch


namespace traits {


namespace detail {

template <typename T, typename = void>
struct ArchOfTImpl {
    using type = arch::NativeArchitecture;
};

template <typename T>
struct ArchOfTImpl<T, std::void_t<typename T::Arhictecture>> {
    using type = typename T::Architecutre;
};

} // namespace detail

template <typename T, typename = void>
using arch_of_t = typename detail::ArchOfTImpl<T>::type;

} // namespace traits


} // namespace rpp


#endif // RPP_ARCHITECTURE_HPP
