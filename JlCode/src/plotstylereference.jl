# Plot style reference for Julia Plots.jl / StatsPlots.jl

# Plot conventions:
# xlabels, ylabels and labels are capitalized. 
# No title; We usually add a figure description in the text.
# Examples: "Feature Probability", "Represented Features"
# All plots should be readable in black and white, so we use different line styles and markers to distinguish between different curves.
# We use the same naming conventions for the dataframes used for the plots, i.e., the dataframe column names should be capitalized as well

using Plots, StatsPlots

export lineplot_kwargs, errorbar_kwargs
export linestyle_palette, color_palette

# Set default plot attributes

default(
    # fontfamily = "Computer Modern",
    guidefontsize = 10,      # x and y labels
    tickfontsize = 10,       # tick labels
    legendfontsize = 10,     # legend
    titlefontsize = 12,      # subplot titles (though we typically don't use titles)
    linewidth = 3,
    markersize = 6,
#   markerstrokewidth = 0,
#   grid = false,
#   framestyle = :box,
#   dpi = 300,
    #size = (400, 300)
)

color_palette = palette(:Set1_9)  
# Default plotting kwargs for line plots
lineplot_kwargs = (
    linewidth = 3,
    markersize = 10,
    palette = color_palette,  # Use Seaborn-like colors
)


# For error bands (using ribbons in Plots.jl)
errorbar_kwargs = (
    fillalpha = 0.05,  # equivalent to alpha in Python
    ribbon_style = :fill,  # for error bands
)

# Example usage for typical plots:
# plot(x, y; xlabel="Feature Probability", ylabel="Represented Features (mean ± sd)", 
#      xscale=:log10, lineplot_kwargs...)
# 
# For error bands:
# plot(x, y; ribbon=yerr, xlabel="Feature Probability", ylabel="Represented Features", 
#      fillalpha=0.05, lineplot_kwargs...)

linestyle_palette = [:solid, :dash, :dashdot, :dot]
