using CategoricalArrays
using Plots
using LaTeXStrings

export line_connected_to_center, circle_points

"""
    line_connected_to_center(points)

Connects a list of points to the center (0, 0) and adds NaN to separate segments.
Use for example for plotting polygons in a star-like way.
"""
function line_connected_to_center(points)
    _zero = zero(points[1])
    _nan_vec = _zero .+ NaN
    points = cat(([_zero, point, _nan_vec] for point in points)...; dims=1)
    x = first.(points)
    y = last.(points)
    return x, y
end

function circle_points(radius=1., n_points=100)
    return radius * [cos(2π * i / n_points) for i in 0:n_points-1], radius * [sin(2π * i / n_points) for i in 0:n_points-1]
end

#! doesn't yet seem to work for lines (currently shows scatter plots); 
# deprecated
function plot_with_markers(x, y; markers = nothing, styles = nothing, colors = nothing, 
                          marker_title = nothing, style_title = nothing, color_title = nothing, kwargs...)
    """
    Plots the data in `x` and `y` where the markers, styles and colors are automatically chosen based on the markers, styles, and colors input arguments provided.
    Legend shows each visual property independently rather than all combinations.
    
    # Arguments
    - `x`: x coordinates of the data points
    - `y`: y coordinates of the data points
    - `markers`: Optional. Vector indicating marker grouping for data points
    - `styles`: Optional. Vector indicating line style grouping for data points
    - `colors`: Optional. Vector indicating color grouping for data points
    - `kwargs...`: Additional keyword arguments passed to the plot function
    """
    
    # Default plot object
    p = plot(; kwargs...)
    
    # Define available visual properties (similar to seaborn)
    available_markers = [:circle, :square, :diamond, :utriangle, :dtriangle, :cross, :xcross, :star5]
    available_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    available_colors = [:blue, :red, :green, :purple, :orange, :cyan, :magenta, :yellow, :brown, :pink]
    
    # Process categorical data for markers, styles, and colors
    if markers !== nothing
        cat_markers = categorical(markers)
        unique_markers = levels(cat_markers)
        marker_map = Dict(m => available_markers[i % length(available_markers) + 1] 
                         for (i, m) in enumerate(unique_markers))
    else
        cat_markers = categorical(fill(1, length(x)))
        marker_map = Dict(1 => available_markers[1])
    end
    
    if styles !== nothing
        cat_styles = categorical(styles)
        unique_styles = levels(cat_styles)
        style_map = Dict(s => available_styles[i % length(available_styles) + 1] 
                        for (i, s) in enumerate(unique_styles))
    else
        cat_styles = categorical(fill(1, length(x)))
        style_map = Dict(1 => available_styles[1])
    end
    
    if colors !== nothing
        cat_colors = categorical(colors)
        unique_colors = levels(cat_colors)
        color_map = Dict(c => available_colors[i % length(available_colors) + 1] 
                        for (i, c) in enumerate(unique_colors))
    else
        cat_colors = categorical(fill(1, length(x)))
        color_map = Dict(1 => available_colors[1])
    end
    
    # Generate all unique combinations of markers, styles, and colors that appear in the data
    combinations = Set(zip(cat_markers, cat_styles, cat_colors))
    
    # Plot each combination separately without labels (for the actual data)
    for (m, s, c) in combinations
        # Find indices where this combination appears
        indices = findall((cat_markers .== m) .& (cat_styles .== s) .& (cat_colors .== c))
        
        # Extract corresponding x and y values
        x_subset = x[indices]
        y_subset = y[indices]
        
        # Add to plot without labels
        plot!(p, x_subset, y_subset, 
             marker=marker_map[m],
             linestyle=style_map[s],
             color=color_map[c],
             label=nothing)  # No label for the actual data points
    end


    function plot_heading_entry(heading_text)
        # Add invisible entry to the legend for the section heading (plain text only)
        # Use LaTeXStrings for formatting if desired
        heading_text === nothing && return
        
        plot!(p, Float64[], Float64[], 
            label = heading_text,
            color = :transparent,
            linealpha = 0)
    end
    
    # Add invisible entries to the legend for each marker type
    if markers !== nothing && length(unique(cat_markers)) > 1
        # Add customizable section heading
        plot_heading_entry(marker_title)
        
        for m in levels(cat_markers)
            plot!(p, Float64[], Float64[], 
                marker=marker_map[m],
                markersize=6, 
                markerstrokewidth=1,
                linealpha=0,
                color=:black,
                linestyle=:solid,
                label="$(m)")
        end
    end
    
    # Add invisible entries to the legend for each line style
    if styles !== nothing && length(unique(cat_styles)) > 1
        # Add customizable section heading
        plot_heading_entry(style_title)
        
        for s in levels(cat_styles)
            plot!(p, Float64[], Float64[], 
                marker=:none,  # No marker
                linestyle=style_map[s], 
                linewidth=2,
                color=:black,  # Consistent color
                label="$(s)")
        end
    end
    
    # Add invisible entries to the legend for each color
    if colors !== nothing && length(unique(cat_colors)) > 1
        # Add customizable section heading
        plot_heading_entry(color_title)

        for c in levels(cat_colors)
            plot!(p, Float64[], Float64[], 
                marker=:none,  # No marker
                linestyle=:solid,  # Consistent linestyle
                linewidth=2,
                color=color_map[c],
                label="$(c)")
        end
    end
    
    return p
end

# @deprecate plot_with_markers(x, y; markers = nothing, styles = nothing, colors = nothing, 
#                          marker_title = nothing, style_title = nothing, color_title = nothing, kwargs...) "Use # `plot_with_markers` instead."