#ifndef INCLUDE_RPP_SUPPORT_ERROR_HPP
#define INCLUDE_RPP_SUPPORT_ERROR_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <type_traits>
#include <variant>

#include <rpp/config.h>

namespace rpp {
/**
 * @brief Enumerates the error codes that identify distinct failure conditions.
 *
 * The enumeration values correspond to specific situations that arise during
 * operation execution. They are intentionally compact, using an @c
 * `unsigned char` underlying type to minimize footprint.
 *
 * The `Ok` code indicates successful completion, while other entries represent
 * various failure modes such as cancellation, timeouts, or contract violations.
 */
enum class ErrorCode : unsigned char {
    Ok = 0,
    Cancelled = 1,
    Unknown = 2,
    InvalidArgument = 3,
    Timeout = 4,
    OutOfResources = 5,
    ContractViolation = 6,
    OutOfBounds = 7,
    NotImplemented = 8,
    Internal = 9
};

namespace error_detail {

template <typename Payload>
struct PayloadHolder;

template <>
struct PayloadHolder<void> {
    static constexpr bool has_payload = false;
};

template <>
struct PayloadHolder<std::string> {
    static constexpr bool has_payload = true;
    std::string message;
};

template <>
struct PayloadHolder<const char*> {
    static constexpr bool has_payload = true;
    const char* message;
};

/**
 * @brief Returns a default human‑readable message for the specified error code.
 *
 * @param code The error code whose default message is desired.
 * @return A string view containing the corresponding message.
 */
RPP_NODISCARD
constexpr std::string_view default_message(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::Ok:
            return "Success";
        case ErrorCode::Cancelled:
            return "Operation cancelled";
        case ErrorCode::Unknown:
            return "Unknown error";
        case ErrorCode::InvalidArgument:
            return "Invalid argument";
        case ErrorCode::Timeout:
            return "Timeout expired";
        case ErrorCode::OutOfResources:
            return "Out of resources";
        case ErrorCode::ContractViolation:
            return "Contract violation";
        case ErrorCode::OutOfBounds:
            return "Out of bounds";
        case ErrorCode::NotImplemented:
            return "Not implemented";
        case ErrorCode::Internal:
            return "Internal error";
    }
    return "Unknown error";
}

} // namespace error_detail

/**
 * @brief Represents an error with an associated @ref ErrorCode and an optional payload.
 *
 * The class template can be instantiated with a @p Payload type that determines
 * whether the error carries additional information used to compose a custom
 * message. When @p Payload is @c void (or otherwise lacks an associated message),
 * the error falls back to a default human‑readable message retrieved from
 * @ref error_detail::default_message. If a payload is provided and supports
 * message extraction, that message is returned instead.
 *
 * Construction can occur with just an @ref ErrorCode or with both a code and a
 * payload. The class provides implicit conversion to @c bool to indicate success,
 * logical NOT to test for the @c Ok state, and access to the stored code and
 * message.
 *
 * @tparam Payload The type of optional payload carried by the error. Defaults to
 *                 @c void when no payload is needed.
 *
 * @param code The @ref ErrorCode that identifies the specific error condition.
 *
 * @note The type is annotated with @ref RPP_NODISCARD to enforce explicit handling
 *       of returned error instances, and it inherits from @ref error_detail::PayloadHolder
 *       to manage payload storage and default‑message fallback.
 */
template <typename Payload = void>
class RPP_NODISCARD Error : error_detail::PayloadHolder<Payload> {
    using Holder = error_detail::PayloadHolder<Payload>;
    ErrorCode code_;

public:
    explicit Error(ErrorCode code) : code_(code) {}

    template <typename... Args>
    explicit Error(ErrorCode code, Args&&... args)
        : Holder{std::forward<Args>(args)...}, code_(code) {}

    constexpr explicit operator bool() const noexcept { return code_ != ErrorCode::Ok; }

    constexpr bool operator!() const noexcept { return code_ == ErrorCode::Ok; }

    RPP_NODISCARD
    constexpr ErrorCode code() const noexcept { return code_; }

    // Returns payload message if present, otherwise uses default_message.
    RPP_NODISCARD
    constexpr std::string_view message() const noexcept {
        if constexpr (Holder::has_payload) {
            return this->Holder::message;
        } else {
            return error_detail::default_message(code_);
        }
    }
};

/**
 * @brief Result – a wrapper type that represents either a successful
 *        computation producing a value of type T or an error represented
 *        by a type E.
 *
 * The template facilitates handling operations that may fail without using
 * exceptions. It provides mechanisms to access the stored value, transform
 * it, or convert it into other types while preserving the underlying
 * success/failure state.
 *
 * @tparam T The type of the successful result value.
 * @tparam E The type used to represent an error condition.
 */
template <typename T, typename E = Error<>>
class Result {
    std::variant<T, E> stored_;
public:

    constexpr Result() noexcept : stored_(std::in_place_type<void>) {}
    constexpr Result(Result const& other) : stored_(other.stored_) {}
    constexpr Result(Result&& other) noexcept : stored_(std::move(other.stored_)) {}

    template <typename U, typename=std::enable_if_t<!std::is_same_v<std::remove_cv_t<U>, Result>>>
    // ReSharper disable once CppNonExplicitConvertingConstructor
    constexpr Result(U&& v) noexcept : stored_(std::in_place_type<U>, std::forward<U>(v)) {}

    // ReSharper disable once CppNonExplicitConvertingConstructor
    constexpr Result(E err) noexcept : stored_(std::in_place_type<E>, std::move(err)) {}

    constexpr Result& operator=(Result const& other) noexcept {
        if (&other != this) {
            stored_ = other.stored_;
        }
        return *this;
    }
    constexpr Result& operator=(Result&& other) noexcept {
        if (&other != this) {
            stored_ = std::move(other.stored_);
        }
        return *this;
    }

    RPP_NODISCARD
    constexpr bool ok() const noexcept { return std::holds_alternative<T>(stored_); }
    constexpr explicit operator bool() const noexcept { return ok(); }
    constexpr bool operator!() const noexcept { return !ok(); }

    constexpr T const& value() const& { return std::get<T>(stored_); }
    constexpr T& value() & { return std::get<T>(stored_); }
    constexpr T&& value() && { return std::get<T>(std::move(stored_)); }

    constexpr E const& error() const& { return std::get<E>(stored_); }
    constexpr E& error() & { return std::get<E>(stored_); }
    constexpr E&& error() && { return std::get<E>(std::move(stored_)); }
};




} // namespace rpp

#endif // INCLUDE_RPP_SUPPORT_ERROR_HPP