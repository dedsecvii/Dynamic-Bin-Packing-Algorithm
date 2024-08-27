import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Open in wide mode
st.set_page_config(layout="wide")

# Load CSV data
csv_path = 'results.csv'
df = pd.read_csv(csv_path)

# Sorting algorithms by the time taken (lower is better)
df_sorted = df.sort_values(by=['Case', 'Time (s)'])

# Convert metrics to float
df_sorted['Time (s)'] = df_sorted['Time (s)'].astype(float)
df_sorted['Bins Used'] = df_sorted['Bins Used'].astype(float)

# Create unique lists for dropdowns
cases = df_sorted['Case'].unique()
metrics = ['Time (s)','Time / Item (s)', 'Bins Used',"Bin Utilization", "Avg Items per Bin",'Efficiency','Bins Used per Item','Percent Items Departed']
metrics_lower_better=[True,True,True,False,False,False,True,False]
algorithms = df_sorted['Algorithm'].unique()

# Create discrete lists for sliders
items_values = sorted(df['nitems'].unique())
days_values = sorted(df['ndays'].unique())

# Sidebar controls for selecting case, metric, and algorithms
with st.sidebar:
    selected_case = st.selectbox("Select Case", cases)
    selected_metric = st.selectbox("Select Metric", metrics)
    selected_algorithms = st.multiselect("Select Algorithms", algorithms, default=algorithms,
                                         placeholder="Select one or more algorithms")
    items_range = st.select_slider("Select number of items range", options=items_values,
                                   value=(items_values[0], items_values[-1]))
    if selected_case in ['Time-aware items','Correlated Arrival and Departure']:
        days_range = st.select_slider("Select number of days range", options=days_values,
                                      value=(days_values[0], days_values[-1]))
    else:
        days_range = (-1, -1)


# Function to create the main bar plot
def create_bar_plot(case, metric, algorithms, items, days):
    # Filter and sort the dataframe
    filtered_df = df_sorted[(df_sorted['Case'] == case) & (df_sorted['Algorithm'].isin(algorithms))]
    filtered_df = filtered_df[(filtered_df['nitems'] >= items[0]) & (filtered_df['nitems'] <= items[1]) & (
            filtered_df['ndays'] >= days[0]) & (filtered_df['ndays'] <= days[1])]
    filtered_df = filtered_df.sort_values(by=[metric])

    if filtered_df.empty:
        st.error("No data available for the selected case, algorithms, or range of items/days")
        st.stop()

    # Group by Algorithm and calculate the mean of the selected metric

    grouped_df = filtered_df.groupby('Algorithm').mean(numeric_only=True).sort_values(by=metric)

    # Determine the color for the bars
    colors = ['#636EFA'] * len(grouped_df)

    # Create the bar plot
    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped_df.index,
                y=grouped_df[metric],
                name=metric,
                marker_color=colors
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title=f'Algorithm Ranking for {case} ({metric})',
        xaxis_title='Algorithm',
        yaxis_title="Mean " + metric,
        height=800,
        width=1400,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig


# Function to create a line chart of the selected metric vs. number of items
def create_line_plot(case, metric, algorithms, days):
    filtered_df = df_sorted[(df_sorted['Case'] == case) & (df_sorted['Algorithm'].isin(algorithms))]
    filtered_df = filtered_df[(filtered_df['ndays'] >= days[0]) & (filtered_df['ndays'] <= days[1])]
    #group by nitems and calculate the mean of the selected metric
    filtered_df = filtered_df.groupby(['Algorithm', 'nitems']).mean(numeric_only=True).reset_index()
    if filtered_df.empty:
        st.error("No data available for the selected case, algorithms, or range of items/days")
        st.stop()

    fig = go.Figure()
    for algo in algorithms:
        algo_df = filtered_df[filtered_df['Algorithm'] == algo]
        fig.add_trace(go.Scatter(x=algo_df['nitems'], y=algo_df[metric], mode='lines+markers', name=algo))

    fig.update_layout(
        title=f'{metric} vs. Number of Items for {case}',
        xaxis_title='Number of Items',
        yaxis_title=metric,
        height=800,
        width=1400,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig


# Function to create a grouped bar chart with ranks
def create_grouped_bar_chart(metric, algorithms):
    filtered_df = df_sorted[df_sorted['Algorithm'].isin(algorithms)]

    if filtered_df.empty:
        st.error("No data available for the selected algorithms")
        st.stop()

    # Aggregate the data
    grouped_df = filtered_df.groupby(['Algorithm', 'Case']).agg({metric: 'mean'}).reset_index()
    grouped_df.columns = ['Algorithm', 'Case', 'Mean']

    # Calculate the rank within each case
    grouped_df['Rank'] = grouped_df.groupby('Case')['Mean'].rank("dense", ascending=metrics_lower_better[metrics.index(metric)])

    # Create the grouped bar chart
    fig = go.Figure()
    for case in grouped_df['Case'].unique():
        case_df = grouped_df[grouped_df['Case'] == case]
        fig.add_trace(go.Bar(
            x=case_df['Algorithm'],
            y=case_df['Rank'],
            name=f'{case} (Rank)',
            hovertext=[f"Mean {metric}: {mean_val:.2f}" for mean_val in case_df['Mean']],
            marker=dict(line=dict(width=1))
        ))

    fig.update_layout(
        barmode='group',
        title=f'Algorithm Ranks by Case ({metric})',
        xaxis_title='Algorithm',
        yaxis_title='Rank (Lower is Better)',
        height=800,
        width=1400,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig


if not selected_algorithms:
    st.error(" 👈 Please select at least one algorithm from the sidebar")
    st.stop()

# Create and display the main bar plot
bar_plot_fig = create_bar_plot(selected_case, selected_metric, selected_algorithms, items_range, days_range)
st.plotly_chart(bar_plot_fig, use_container_width=True)

# Create and display the line plot of the selected metric vs. number of items
line_plot_fig = create_line_plot(selected_case, selected_metric, selected_algorithms, days_range)
st.plotly_chart(line_plot_fig, use_container_width=True)

# Create and display the grouped bar chart with ranks
grouped_bar_fig = create_grouped_bar_chart(selected_metric, selected_algorithms)
st.plotly_chart(grouped_bar_fig, use_container_width=True)

# Optionally, save the figures as HTML files
st.markdown("### Save the plots")
if st.button("Save All Plots as HTML"):
    bar_plot_fig.write_html("bar_plot.html")
    line_plot_fig.write_html("line_plot.html")
    grouped_bar_fig.write_html("grouped_bar.html")
    st.success("Plots saved as bar_plot.html, line_plot.html, and grouped_bar.html")
