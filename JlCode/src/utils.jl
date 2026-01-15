using DataFrames
using PrettyTables
using Crayons
using Random
using Colors
using ColorSchemes
using Printf
using Statistics

export display_df_colorbg, display_df_colortxt, NumberFormatter


struct NumberFormatter
    float_format_str::String # Format string for floats in Printf format
    int_format_str::String   # Format string for integers in Printf format
end

NumberFormatter(; num_float_digits = 3) = NumberFormatter("%.$(num_float_digits)f", "%d")

"""
    number_equivalent(formatter, value)

Trait-like function to extract a numeric value for coloring from any value.
Returns a number (for coloring), or nothing if the value should not be colored.
Default: numbers return themselves, numeric arrays return mean, others return nothing.
"""
number_equivalent(::NumberFormatter, value::Integer) = value
number_equivalent(::NumberFormatter, value::AbstractFloat) = value
number_equivalent(::NumberFormatter, value::AbstractArray) =
    all(x -> isa(x, Number) && !ismissing(x), value) && !isempty(value) ? mean(skipmissing(value)) : nothing
number_equivalent(::NumberFormatter, value) = nothing  # fallback


function (formatter::NumberFormatter)(value, i, j)
    if isa(value, Integer) && !(isa(value, Bool))
        return Printf.format(Printf.Format(formatter.int_format_str), value)
    elseif isa(value, AbstractFloat)
        return Printf.format(Printf.Format(formatter.float_format_str), value)
    elseif isa(value, AbstractArray)
        # Only handle arrays of numbers
        if isempty(value)
            return "[]"
        elseif all(x -> isa(x, Number) && !ismissing(x), value)
            arr = collect(skipmissing(value))
            m = mean(arr)
            s = std(arr)
            mn, mx = extrema(arr)
            fmt = formatter.float_format_str
            mean_str = Printf.format(Printf.Format(fmt), m)
            std_str = Printf.format(Printf.Format(fmt), s)
            min_str = Printf.format(Printf.Format(fmt), mn)
            max_str = Printf.format(Printf.Format(fmt), mx)
            return "$(mean_str)±$(std_str) [$(min_str), $(max_str)]"
        else
            # Non-numeric or mixed array
            return string(value)
        end
    else
        return string(value)  # Default for other types
    end
end


"""
    display_df_colorbg(df::DataFrame; steps=20, colorscheme=ColorSchemes.coolwarm, formatter=NumberFormatter())

Print DataFrame with continuous background color gradient using standardized colorschemes.
Only colors numerical values, ignores strings. Extrema calculated per column.

# Arguments
- `df::DataFrame`: The DataFrame to display
- `steps::Int=20`: Number of color steps for the gradient  
- `colorscheme=ColorSchemes.coolwarm`: ColorScheme from ColorSchemes.jl
- `formatter=NumberFormatter()`: Formatter for numbers (default: integers as-is, floats with 3 decimals)

# Examples
```julia
display_df_colorbg(df)  # Default formatting
display_df_colorbg(df, formatter=NumberFormatter(num_float_digits=2))  # 2 decimal places
display_df_colorbg(df, formatter=(v,i,j) -> my_custom_format(v))  # Custom formatter function
```
"""
function display_df_colorbg(df::DataFrame; steps=20, colorscheme=ColorSchemes.coolwarm, formatter=NumberFormatter())
    # Calculate min/max for each column using number_equivalent
    col_ranges = Dict{Int, Tuple{Any, Any}}()

    for (col_idx, col) in enumerate(eachcol(df))
        num_equiv_values = [number_equivalent(formatter, val) for val in col]
        # Filter out nothing and missing
        num_equiv_values = [v for v in num_equiv_values if v !== nothing && !ismissing(v)]
        if !isempty(num_equiv_values)
            col_ranges[col_idx] = extrema(num_equiv_values)
        end
    end

    if isempty(col_ranges)
        pretty_table(df, formatters = (v,i,j) -> formatter(v, i, j), crop = :none)
        return
    end
    
    highlighters = [
        Highlighter(
            (data, i, j) -> begin
                val = data[i, j]
                numval = number_equivalent(formatter, val)
                if numval === nothing || ismissing(numval) || !haskey(col_ranges, j)
                    return false
                end
                min_val, max_val = col_ranges[j]
                if min_val == max_val  # Handle constant columns
                    return k == 1
                end
                lower = min_val + (k-1)*(max_val-min_val)/steps
                upper = min_val + k*(max_val-min_val)/steps
                k == steps ? lower <= numval <= upper : lower <= numval < upper
            end,
            let t = (k-1)/(steps-1)
                color = get(colorscheme, t)
                r, g, b = round.(Int, (red(color), green(color), blue(color)) .* 255)
                Crayon(background = (r, g, b), foreground = :black)
            end
        ) for k in 1:steps
    ]
    
    pretty_table(df, highlighters = Tuple(highlighters), formatters = (v,i,j) -> formatter(v, i, j), crop = :none)
end

"""
    display_df_colortxt(df::DataFrame; colorscheme=ColorSchemes.coolwarm, formatter=NumberFormatter())

Print DataFrame with colored text using standardized colorschemes.
Only colors numerical values, ignores strings. Extrema calculated per column.
Default: coolwarm (blue=cold/min, red=warm/max)

# Arguments
- `df::DataFrame`: The DataFrame to display
- `colorscheme=ColorSchemes.coolwarm`: ColorScheme from ColorSchemes.jl
- `formatter=NumberFormatter()`: Formatter for numbers (default: integers as-is, floats with 3 decimals)

# Examples
```julia
display_df_colortxt(df)  # Default formatting
display_df_colortxt(df, formatter=NumberFormatter(num_float_digits=1))  # 1 decimal place
display_df_colortxt(df, formatter=(v,i,j) -> my_custom_format(v))  # Custom formatter function
```
"""
function display_df_colortxt(df::DataFrame; colorscheme=ColorSchemes.coolwarm, formatter=NumberFormatter())
    # Calculate min/max for each column separately
    col_ranges = Dict{Int, Tuple{Float64, Float64}}()
    
    for (col_idx, col) in enumerate(eachcol(df))
        numeric_values = [val for val in col if isa(val, Number) && !ismissing(val)]
        if !isempty(numeric_values)
            col_ranges[col_idx] = extrema(numeric_values)
        end
    end
    
    if isempty(col_ranges)
        pretty_table(df, formatters = (v,i,j) -> formatter(v, i, j), crop = :none)
        return
    end
    
    # Create 5 discrete color levels from the colorscheme
    colors = [get(colorscheme, t) for t in [0.0, 0.25, 0.5, 0.75, 1.0]]
    crayons = [Crayon(foreground = (round.(Int, (red(c), green(c), blue(c)) .* 255)...,)) for c in colors]
    
    highlighters = (
        Highlighter((data, i, j) -> begin
            val = data[i, j]
            numval = number_equivalent(formatter, val)
            if numval === nothing || ismissing(numval) || !haskey(col_ranges, j)
                return false
            end
            min_val, max_val = col_ranges[j]
            numval == min_val
        end, Crayon(foreground = (round.(Int, (red(colors[1]), green(colors[1]), blue(colors[1])) .* 255)...,), bold = true)),
        
        Highlighter((data, i, j) -> begin
            val = data[i, j]
            numval = number_equivalent(formatter, val)
            if numval === nothing || ismissing(numval) || !haskey(col_ranges, j)
                return false
            end
            min_val, max_val = col_ranges[j]
            numval == max_val && min_val != max_val
        end, Crayon(foreground = (round.(Int, (red(colors[5]), green(colors[5]), blue(colors[5])) .* 255)...,), bold = true)),
        
        Highlighter((data, i, j) -> begin
            val = data[i, j]
            numval = number_equivalent(formatter, val)
            if numval === nothing || ismissing(numval) || !haskey(col_ranges, j)
                return false
            end
            min_val, max_val = col_ranges[j]
            min_val != max_val && (numval - min_val) / (max_val - min_val) < 0.33
        end, crayons[2]),
        
        Highlighter((data, i, j) -> begin
            val = data[i, j]
            numval = number_equivalent(formatter, val)
            if numval === nothing || ismissing(numval) || !haskey(col_ranges, j)
                return false
            end
            min_val, max_val = col_ranges[j]
            min_val != max_val && 0.33 <= (numval - min_val) / (max_val - min_val) < 0.67
        end, crayons[3]),
        
        Highlighter((data, i, j) -> begin
            val = data[i, j]
            numval = number_equivalent(formatter, val)
            if numval === nothing || ismissing(numval) || !haskey(col_ranges, j)
                return false
            end
            min_val, max_val = col_ranges[j]
            min_val != max_val && 0.67 <= (numval - min_val) / (max_val - min_val) < 1.0
        end, crayons[4])
    )
    
    pretty_table(df, highlighters = highlighters, formatters = (v,i,j) -> formatter(v, i, j), crop = :none)
end