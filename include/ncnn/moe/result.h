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
    Result(T value) : storage_(std::move(value))
    {
    }
    Result(Error error) : storage_(std::move(error))
    {
    }

    explicit operator bool() const noexcept
    {
        return std::holds_alternative<T>(storage_);
    }

    T& value() &
    {
        assert(std::holds_alternative<T>(storage_));
        return std::get<T>(storage_);
    }

    const T& value() const&
    {
        assert(std::holds_alternative<T>(storage_));
        return std::get<T>(storage_);
    }

    T&& value() &&
    {
        assert(std::holds_alternative<T>(storage_));
        return std::move(std::get<T>(storage_));
    }

    const Error& error() const
    {
        assert(std::holds_alternative<Error>(storage_));
        return std::get<Error>(storage_);
    }

private:
    std::variant<T, Error> storage_;
};

template<>
class Result<void>
{
public:
    Result() = default;
    Result(Error error) : has_error_(true), error_(std::move(error))
    {
    }

    explicit operator bool() const noexcept
    {
        return !has_error_;
    }

    const Error& error() const
    {
        assert(has_error_);
        return error_;
    }

private:
    bool has_error_ = false;
    Error error_;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_RESULT_H
