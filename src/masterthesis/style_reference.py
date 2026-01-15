# define the plot style reference

# Plot conventions:
# xlabels, ylabels and labels are title capitalized, i.e. only very short words, such as of are not capitalized. 
# No title; We usually add a figure description in the text.
# Examples: "Feature Probability", "Represented Features"
# All plots should be readable in black and white, so we use different line styles and markers to distinguish between different curves.
# We use the same naming conventions for the dataframes used for the plots, i.e., the dataframe column names should be capitalized as well

"""
# Plot Conventions
xlabels, ylabels, and labels are title capitalized, i.e., only very short words, such as "of", are not capitalized.
As examples: "Feature Probability", "Represented Features".

No title; we usually add a figure description in the text.
All plots should be readable in black and white, so we use different line styles and markers to distinguish between different curves.
We use the same naming conventions for the dataframes used for the plots, i.e., the dataframe column names should be capitalized as well.
We use the standard markers and line styles from seaborn, which are compatible with black and white printing.

# Plotting Code Conventions
We use the short syntax for plots, for example:
```
plt.gca().set(ylabel='Represented Features (mean ± sd)', xlabel='Feature Probability', xscale='log')
```
where the line is in only one line for brevity and oversight.
We use tight layout for all plots to ensure that labels and titles do not overlap.

Full example:
```python
plt.figure()
sns.lineplot(data=leaky_df, x='Feature Probability', y='Represented Features Weight', hue='Activation Function', style='Activation Function', 
             estimator='mean', errorbar='sd', legend='full', markers=True, **errorbar_kwargs, **lineplot_kwargs)
plt.gca().set(ylabel='Represented Features Weight (mean ± sd)', xlabel='Feature Probability', xscale='log')
plt.tight_layout()
logger.save_figure(plt.gcf(), "leaky_relu_variants_comparison_weight_based", show=True)

## Side-by-side plots
We use side-by-side plots for comparing different experiments or configurations.
The standard size is (16, 6) for two plots side by side.
If both plots have the same labels, we only show a legend for the left plot.
We do not manually synchronize colors or markers, as seaborn handles this automatically if one uses the same data in the same order.
```

"""
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10,           # General font size (in points)
    'axes.labelsize': 10,      # x and y labels
    'axes.titlesize': 12,      # subplot titles
    'xtick.labelsize': 12,     # x tick labels
    'ytick.labelsize': 12,     # y tick labels
    'legend.fontsize': 12,     # legend
    'figure.figsize': (10, 6), # default figure size
})

lineplot_kwargs = dict(
    linewidth = 3,
    markersize = 10,
    palette = "tab10",
)

errorbar_kwargs = dict(
    err_style = "band",
    err_kws = {"alpha": 0.05},
)