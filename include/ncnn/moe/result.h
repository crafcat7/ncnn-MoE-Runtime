#ifndef NCNN_MOE_RESULT_H
#define NCNN_MOE_RESULT_H

#include <cassert>
#include <string>
#include <utility>
#include <variant>

namespace ncnn {
namespace moe {

enum class ErrorCode
{
    InvalidArgument,
    IoError,
    InvalidModel,
    UnsupportedModel,
    InternalError
};

struct Error
{
    ErrorCode code = ErrorCode::InternalError;
    std::string message;
};

template<typename T>
class Result
{
public:
    Result(T value)
        : storage(std::move(value))
    {
    }
    Result(Error error)
        : storage(std::move(error))
    {
    }

    explicit operator bool() const noexcept
    {
        return std::holds_alternative<T>(storage);
    }

    T& value() &
    {
        assert(std::holds_alternative<T>(storage));
        return std::get<T>(storage);
    }

    const T& value() const&
    {
        assert(std::holds_alternative<T>(storage));
        return std::get<T>(storage);
    }

    T&& value() &&
    {
        assert(std::holds_alternative<T>(storage));
        return std::move(std::get<T>(storage));
    }

    const Error& error() const
    {
        assert(std::holds_alternative<Error>(storage));
        return std::get<Error>(storage);
    }

private:
    std::variant<T, Error> storage;
};

template<>
class Result<void>
{
public:
    Result() = default;
    Result(Error error)
        : has_error(true), failure(std::move(error))
    {
    }

    explicit operator bool() const noexcept
    {
        return !has_error;
    }

    const Error& error() const
    {
        assert(has_error);
        return failure;
    }

private:
    bool has_error = false;
    Error failure;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_RESULT_H
