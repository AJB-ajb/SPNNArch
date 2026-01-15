"""
Simple interval arithmetic implementation for Julia.

This module provides a basic implementation of interval arithmetic
"""

# Convenience imports and public API exports for the SimpleIntervals file
using Base: isnan, isfinite, isinf, zero, NaN, Inf, iseven, convert

# Public API
export Interval, interval
export isempty, isfinite, width, midpoint, radius
export in, issubset, intersect, clamp, hull
export +, -, *, /, ^, sqrt, exp, log
export bisect, convert, float, eval_on_boundary

struct Interval{T, S}
    lo::T
    hi::S
end

# Type-unified interval constructor
function interval(lo::T, hi::S) where {T, S}
    if isnan(lo) || isnan(hi)
        U = promote_type(T, S)
        return Interval(U(NaN), U(NaN))
    end
    if lo > hi
        U = promote_type(T, S)
        return Interval(U(NaN), U(NaN))  # Empty interval
    end
    return Interval(lo, hi)
end

# Single-type constructors for backward compatibility
interval(lo::T, hi::T) where T = Interval(lo, hi)

# Outer constructors
interval(x::T) where T = interval(x, x)  # Point interval

# Display
Base.show(io::IO, x::Interval) = print(io, "[$(x.lo), $(x.hi)]")

# Basic properties
Base.isempty(x::Interval) = isnan(x.lo) || isnan(x.hi) || x.lo > x.hi
Base.isfinite(x::Interval) = isfinite(x.lo) && isfinite(x.hi)
width(x::Interval{T, S}) where {T, S} = isempty(x) ? promote_type(T, S)(NaN) : x.hi - x.lo
midpoint(x::Interval{T, S}) where {T, S} = isempty(x) ? promote_type(T, S)(NaN) : (x.lo + x.hi) / 2
radius(x::Interval) = width(x) / 2

# Set operations
Base.in(y::Real, x::Interval) = !isempty(x) && x.lo ≤ y ≤ x.hi
Base.issubset(x::Interval, y::Interval) = isempty(x) || (!isempty(y) && y.lo ≤ x.lo && x.hi ≤ y.hi)

function Base.intersect(x::Interval{T1, S1}, y::Interval{T2, S2}) where {T1, S1, T2, S2}
    if isempty(x) || isempty(y)
        U = promote_type(T1, S1, T2, S2)
        return Interval(U(NaN), U(NaN))
    end
    lo = max(x.lo, y.lo)
    hi = min(x.hi, y.hi)
    return Interval(lo, hi)
end

function Base.clamp(x::Interval, a::Real, b::Real)
    return intersect(x, Interval(a, b))
end

function hull(x::Interval{T1, S1}, y::Interval{T2, S2}) where {T1, S1, T2, S2}
    if isempty(x)
        return y
    elseif isempty(y)
        return x
    else
        return Interval(min(x.lo, y.lo), max(x.hi, y.hi))
    end
end

# Arithmetic operations with proper rounding
function Base.:+(x::Interval{T1, S1}, y::Interval{T2, S2}) where {T1, S1, T2, S2}
    if isempty(x) || isempty(y)
        U = promote_type(T1, S1, T2, S2)
        return Interval(U(NaN), U(NaN))
    end
    return Interval(x.lo + y.lo, x.hi + y.hi)
end

function Base.:-(x::Interval{T1, S1}, y::Interval{T2, S2}) where {T1, S1, T2, S2}
    if isempty(x) || isempty(y)
        U = promote_type(T1, S1, T2, S2)
        return Interval(U(NaN), U(NaN))
    end
    return Interval(x.lo - y.hi, x.hi - y.lo)
end

Base.:-(x::Interval) = isempty(x) ? x : Interval(-x.hi, -x.lo)

function Base.:*(x::Interval{T1, S1}, y::Interval{T2, S2}) where {T1, S1, T2, S2}
    if isempty(x) || isempty(y)
        U = promote_type(T1, S1, T2, S2)
        return Interval(U(NaN), U(NaN))
    end
    
    # Compute all four products
    p1, p2, p3, p4 = x.lo * y.lo, x.lo * y.hi, x.hi * y.lo, x.hi * y.hi
    return Interval(min(p1, p2, p3, p4), max(p1, p2, p3, p4))
end

function Base.:/(x::Interval{T1, S1}, y::Interval{T2, S2}) where {T1, S1, T2, S2}
    if isempty(x) || isempty(y) || 0 ∈ y
        U = promote_type(T1, S1, T2, S2)
        return Interval(U(NaN), U(NaN))
    end
    
    # Multiplication by reciprocal
    return x * Interval(1/y.hi, 1/y.lo)
end

# Scalar operations
Base.:+(x::Interval, c::Real) = Interval(x.lo + c, x.hi + c)
Base.:+(c::Real, x::Interval) = x + c
Base.:-(x::Interval, c::Real) = Interval(x.lo - c, x.hi - c)
Base.:-(c::Real, x::Interval) = Interval(c - x.hi, c - x.lo)
Base.:*(x::Interval, c::Real) = c ≥ 0 ? Interval(x.lo * c, x.hi * c) : Interval(x.hi * c, x.lo * c)
Base.:*(c::Real, x::Interval) = x * c
function Base.:/(x::Interval, c::Real)
    if c > 0
        return Interval(x.lo / c, x.hi / c)
    elseif c < 0
        return Interval(x.hi / c, x.lo / c)
    else
        T = promote_type(typeof(x.lo), typeof(x.hi))
        return Interval(T(NaN), T(NaN))
    end
end

# Power function
function Base.:^(x::Interval, n::Integer)
    if isempty(x)
        return x
    end
    
    if n == 0
        T = promote_type(typeof(x.lo), typeof(x.hi))
        return Interval(T(1), T(1))
    elseif n == 1
        return x
    elseif n < 0
        T = promote_type(typeof(x.lo), typeof(x.hi))
        return Interval(T(1), T(1)) / (x^(-n))
    elseif iseven(n)
        if x.hi ≤ 0
            return Interval(x.hi^n, x.lo^n)
        elseif x.lo ≥ 0
            return Interval(x.lo^n, x.hi^n)
        else  # 0 ∈ x
            T = promote_type(typeof(x.lo), typeof(x.hi))
            return Interval(T(0), max(x.lo^n, x.hi^n))
        end
    else  # odd n > 0
        return Interval(x.lo^n, x.hi^n)
    end
end

# Elementary functions
function Base.sqrt(x::Interval)
    if isempty(x) || x.hi < 0
        T = promote_type(typeof(x.lo), typeof(x.hi))
        return Interval(T(NaN), T(NaN))
    end
    T = promote_type(typeof(x.lo), typeof(x.hi))
    lo = x.lo ≤ 0 ? T(0) : sqrt(x.lo)
    return Interval(lo, sqrt(x.hi))
end

function Base.exp(x::Interval)
    isempty(x) ? x : Interval(exp(x.lo), exp(x.hi))
end

function Base.log(x::Interval)
    if isempty(x) || x.hi ≤ 0
        T = promote_type(typeof(x.lo), typeof(x.hi))
        return Interval(T(NaN), T(NaN))
    end
    T = promote_type(typeof(x.lo), typeof(x.hi))
    lo = x.lo ≤ 0 ? T(-Inf) : log(x.lo)
    return Interval(lo, log(x.hi))
end

# Comparison operations
Base.:(==)(x::Interval, y::Interval) = !isempty(x) && !isempty(y) && x.lo == y.lo && x.hi == y.hi

# Utility functions
function bisect(x::Interval)
    if isempty(x)
        return x, x
    end
    mid = midpoint(x)
    return Interval(x.lo, mid), Interval(mid, x.hi)
end

# Convert to/from other types
Base.convert(::Type{Interval{T, S}}, x::Interval{U, V}) where {T, S, U, V} = Interval{T, S}(T(x.lo), S(x.hi))
Base.convert(::Type{Interval{T}}, x::Interval{U, V}) where {T, U, V} = Interval{T, T}(T(x.lo), T(x.hi))  # For backward compatibility
Base.float(x::Interval{T, S}) where {T, S} = Interval(float(x.lo), float(x.hi))

"Return f(b) - f(a)"
eval_on_boundary(f, x::Interval) = isempty(x) ? zero(f(x.lo)) : (f(x.hi) - f(x.lo))