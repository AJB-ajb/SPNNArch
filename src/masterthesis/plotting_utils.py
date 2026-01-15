import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from IPython.display import display

# ---- dataframe utilities ----

def disp_color_df(df, cmap="coolwarm"):
    """
    Prints a DataFrame with numerical values colorized using the specified colormap.
    
    Args:
        df: pandas DataFrame
        cmap: Colormap name (default: "coolwarm")
    """
    styled_df = df.style.background_gradient(cmap=cmap)
    display(styled_df)


# ------ plotting utilities ----


def add_ribbon(fig, x, y_mean, y_std, name, color='rgba(0,100,80,0.2)'):
    y_upper = y_mean + y_std
    y_lower = y_mean - y_std
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself',
        fillcolor=color,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name=name
    ))

def plot_line_with_ribbon(fig, x, y_mean, y_std, name, rgb_color=(0,100,80), alpha=0.2, line_kwargs=None):
    """
    Plots a line and a ribbon (shaded std band) in the same color on the given figure.
    Args:
        fig: plotly.graph_objects.Figure
        x: x values
        y_mean: mean line values
        y_std: std deviation values
        name: name for the line
        rgb_color: tuple of (R,G,B)
        alpha: float, transparency for the ribbon
        line_kwargs: dict, extra kwargs for the line trace
    """
    if line_kwargs is None:
        line_kwargs = {}
    rgb_str = f"rgb{rgb_color}"
    rgba_str = f"rgba{rgb_color + (alpha,)}"
    y_upper = y_mean + y_std
    y_lower = y_mean - y_std
    # Ribbon
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself',
        fillcolor=rgba_str,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name=name+" (ribbon)"
    ))
    # Overlay line
    fig.add_trace(go.Scatter(
        x=x,
        y=y_mean,
        mode='lines+markers',
        name=name,
        line=dict(color=rgb_str, **line_kwargs)
    ))

def add_axis_scale_toggles(fig):
    """
    Adds two sets of toggle-style buttons to a Plotly figure:
    one for the x-axis scale, one for the y-axis scale.
    """
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="X: Linear",
                         method="relayout",
                         args=[{"xaxis.type": "linear"}]),
                    dict(label="X: Log",
                         method="relayout",
                         args=[{"xaxis.type": "log"}]),
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                x=0,  # move left-right
                y=1.15,  # move up-down
                xanchor="left",
                yanchor="top",
                showactive=True,
            ),
            dict(
                type="buttons",
                buttons=[
                    dict(label="Y: Linear",
                         method="relayout",
                         args=[{"yaxis.type": "linear"}]),
                    dict(label="Y: Log",
                         method="relayout",
                         args=[{"yaxis.type": "log"}]),
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                x=0.4,
                y=1.15,
                xanchor="left",
                yanchor="top",
                showactive=True,
            )
        ],
        margin=dict(t=100)  # slightly larger top margin if needed
    )

def create_comparison_plot(
    x_data, 
    series_data, 
    title, 
    xaxis_title="X-axis", 
    yaxis_title="Y-axis",
    xaxis_type="log",
    showlegend=True,
    name_suffix=" (mean ± std)",
    plot_name=None,
    **layout_kwargs
):
    """
    Create a standardized plotly comparison plot with multiple series, each with line + ribbon.
    
    Args:
        x_data: Common x-axis data (array-like)
        series_data: List of tuples (label, color, mean_values, std_values)
                    where color is RGB tuple and mean/std are array-like
        title: Plot title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label  
        xaxis_type: "log" or "linear"
        showlegend: Whether to show legend
        name_suffix: Suffix added to series names (e.g., " (mean ± std)")
        plot_name: Optional name attribute for the figure
        **layout_kwargs: Additional layout parameters
        
    Returns:
        plotly.graph_objects.Figure
    """
    from masterthesis.utils import to_np
    
    fig = go.Figure()
    
    # Set default layout
    layout_config = {
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "title": title,
        "xaxis_type": xaxis_type,
        "showlegend": showlegend
    }
    
    # Update with any custom layout kwargs
    layout_config.update(layout_kwargs)
    fig.update_layout(**layout_config)
    
    # Add each series as line + ribbon
    for (label, color, mean_values, std_values) in series_data:
        plot_line_with_ribbon(
            fig,
            x_data,
            to_np(mean_values),
            to_np(std_values),
            name=label + name_suffix,
            rgb_color=color
        )
    
    # Set plot name if provided
    if plot_name:
        # Store the name as a custom attribute that won't conflict with Plotly
        fig._custom_name = plot_name
        
    return fig

def get_color_palette(n_colors, palette_name="default"):
    """
    Generate a color palette with n_colors for plotting.
    
    Args:
        n_colors: Number of colors needed
        palette_name: Name of the color palette to use
        
    Returns:
        List of RGB tuples: [(R, G, B), ...]
    """
    palettes = {
        "default": [
            (0, 100, 80),      # Teal
            (100, 20, 80),     # Purple 
            (200, 100, 20),    # Orange
            (20, 100, 200),    # Blue
            (120, 120, 120),   # Gray
            (180, 60, 60),     # Red
            (60, 180, 60),     # Green
            (150, 75, 150),    # Magenta
            (75, 150, 75),     # Light Green
            (150, 150, 75),    # Yellow
        ],
        "colorbrewer_set1": [
            (228, 26, 28),     # Red
            (55, 126, 184),    # Blue  
            (77, 175, 74),     # Green
            (152, 78, 163),    # Purple
            (255, 127, 0),     # Orange
            (255, 255, 51),    # Yellow
            (166, 86, 40),     # Brown
            (247, 129, 191),   # Pink
        ],
        "colorbrewer_dark2": [
            (27, 158, 119),    # Teal
            (217, 95, 2),      # Orange
            (117, 112, 179),   # Purple
            (231, 41, 138),    # Magenta
            (102, 166, 30),    # Green
            (230, 171, 2),     # Yellow
            (166, 118, 29),    # Brown
            (102, 102, 102),   # Gray
        ]
    }
    
    if palette_name not in palettes:
        available = ", ".join(palettes.keys())
        raise ValueError(f"Unknown palette '{palette_name}'. Available: {available}")
    
    palette = palettes[palette_name]
    
    # If we need more colors than available in the palette, cycle through them
    if n_colors <= len(palette):
        return palette[:n_colors]
    else:
        # Cycle through the palette to get enough colors
        colors = []
        for i in range(n_colors):
            colors.append(palette[i % len(palette)])
        return colors

def assign_colors_to_series(series_labels, palette_name="default"):
    """
    Assign colors to a list of series labels.
    
    Args:
        series_labels: List of string labels for the series
        palette_name: Name of the color palette to use
        
    Returns:
        Dict mapping label -> RGB color tuple
    """
    colors = get_color_palette(len(series_labels), palette_name)
    return dict(zip(series_labels, colors))

def plot_line_with_ribbon_mpl(x, y_mean, y_std, label, ax, rgb_color=(0,100,80), alpha=0.2):
    """Plot line with error ribbon using matplotlib."""
    color = [c/255 for c in rgb_color]  # Normalize to 0-1
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=alpha)
    ax.plot(x, y_mean, label=label, color=color)

def create_comparison_plot_mpl(x_data, series_data, title, xaxis_title="X-axis", 
                              yaxis_title="Y-axis", xaxis_type="log", showlegend=True,
                              name_suffix=" (mean ± std)", figsize=(10, 6), **kwargs):
    """Create matplotlib comparison plot with error ribbons."""
    import masterthesis.utils as utils
    
    # Set seaborn style for better aesthetics
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each series
    for label, color, mean_vals, std_vals in series_data:
        plot_line_with_ribbon_mpl(x_data, utils.to_np(mean_vals), utils.to_np(std_vals),
                                  label + name_suffix, ax, color)
    
    # Configure plot
    if xaxis_type == "log":
        ax.set_xscale("log")
    ax.set(xlabel=xaxis_title, ylabel=yaxis_title, title=title)
    if showlegend:
        ax.legend()
    
    return fig, ax

def save_plot(fig, filename, dpi=300, **kwargs):
    """Save Plotly or Matplotlib figure."""
    if isinstance(fig, go.Figure):
        fig.write_image(filename, width=800, height=600)
    else:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)

def create_comparison_plot_mpl_errorbar(x_values, results, title, xaxis_title, yaxis_title):
    """Create clean matplotlib plot with error bars - colorful with line styles for accessibility."""
    import matplotlib.pyplot as plt
    from masterthesis.utils import to_np
    
    # Color-friendly with line style fallback for accessibility
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    markers = ['o', 's', '^', 'v', 'D', '<', '>']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (label, color, μ, σ) in enumerate(results):
        # Use the original color from results, convert RGB tuple to hex if needed
        if isinstance(color, tuple) and len(color) == 3:
            plot_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        else:
            plot_color = color
        
        ax.errorbar(x_values, to_np(μ), yerr=to_np(σ),
                   label=label, color=plot_color,
                   linestyle=line_styles[i % len(line_styles)],
                   marker=markers[i % len(markers)],
                   markersize=5, capsize=4, capthick=1.2, linewidth=1.8)
    
    ax.set_xscale('log')
    ax.set_xlabel(xaxis_title)
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    return fig, ax

def create_publication_ribbon_plot(x, y_mean, y_std, title, xaxis_title, yaxis_title, 
                                 label="Data", color=None, figsize=(8, 6), 
                                 xaxis_type="log", alpha=0.3, linewidth=2.5,
                                 style="whitegrid", context="paper", font_scale=1.2):
    """
    Create a publication-quality seaborn ribbon plot with error bands.
    
    Args:
        x: x-axis values
        y_mean: mean values for y-axis
        y_std: standard deviation values for error band
        title: plot title
        xaxis_title: x-axis label
        yaxis_title: y-axis label
        label: legend label for the data series
        color: color for the plot (default: seaborn blue)
        figsize: figure size tuple
        xaxis_type: "log" or "linear"
        alpha: transparency for error band
        linewidth: width of the main line
        style: seaborn style
        context: seaborn context (paper, notebook, talk, poster)
        font_scale: scaling factor for fonts
        
    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from masterthesis.utils import to_np
    
    # Set seaborn style and context for publication
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to numpy if needed
    x_np = to_np(x)
    y_mean_np = to_np(y_mean)
    y_std_np = to_np(y_std)
    
    # Use seaborn color palette if no color specified
    if color is None:
        color = sns.color_palette("deep")[0]  # Default seaborn blue
    elif isinstance(color, tuple) and len(color) == 3:
        # Convert RGB tuple (0-255) to (0-1) if needed
        if max(color) > 1:
            color = tuple(c/255 for c in color)
    
    # Plot the error band (ribbon)
    ax.fill_between(x_np, y_mean_np - y_std_np, y_mean_np + y_std_np, 
                    alpha=alpha, color=color, label=f'{label} ± std')
    
    # Plot the main line
    ax.plot(x_np, y_mean_np, color=color, linewidth=linewidth, label=label)
    
    # Set scale
    if xaxis_type == "log":
        ax.set_xscale("log")
    
    # Labels and title
    ax.set_xlabel(xaxis_title, fontweight='bold')
    ax.set_ylabel(yaxis_title, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(frameon=True, fancybox=False, shadow=False, 
              loc='best', fontsize='medium')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Tight layout
    plt.tight_layout()
    
    return fig, ax
