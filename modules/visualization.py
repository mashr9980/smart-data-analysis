import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.api import call_llama, extract_json_from_response
from config.settings import DEFAULT_COLORS

def get_visualization_recommendation(query, df):
    """
    Get a recommendation for the best visualization type for the given query and data.
    
    Args:
        query (str): User's data query
        df (pd.DataFrame): The dataframe to visualize
    
    Returns:
        dict: Visualization recommendation details
    """
    data_sample = df.head(5).to_string()
    data_description = f"""
    Data Summary:
    - Shape: {df.shape[0]} rows, {df.shape[1]} columns
    - Columns: {', '.join(df.columns.tolist())}
    - Data types: {', '.join([f"{col}: {df[col].dtype}" for col in df.columns])}
    - Sample data:
    {data_sample}
    """
    
    viz_recommendation_prompt = f"""
    Given this query: "{query}"
    
    And this dataset:
    {data_description}
    
    As a data visualization expert:
    1. What is the BEST visualization to answer this query with this data?
    2. Explain WHY this visualization type is most appropriate.
    3. Provide detailed specifications for creating this visualization:
       - Which columns should be on which axes
       - Any grouping, aggregation, or transformations needed
       - Color scheme recommendations (use only valid matplotlib color names like 'blue', 'green', 'red', 'purple', 'orange')
       - Title and axis labels
       - Any annotations or highlights
    
    Format your response as JSON with these exact keys:
    {{
        "recommended_viz": "visualization type",
        "rationale": "explanation why",
        "specifications": {{
            "x_axis": "column name or transformation",
            "y_axis": "column name or transformation",
            "grouping": "column for grouping if applicable",
            "aggregation": "sum, mean, count, etc. if applicable",
            "title": "recommended title",
            "x_label": "x-axis label",
            "y_label": "y-axis label",
            "annotations": ["any key points to highlight"],
            "color_scheme": "blue"
        }},
        "insights": ["1-3 key insights this visualization will show"]
    }}
    
    IMPORTANT: For color_scheme, use only single color names that matplotlib supports (like 'blue', 'green', 'red', etc.) - do not use color codes or multiple colors.
    
    Just provide the JSON, no other text.
    """
    
    try:
        recommendation_response = call_llama(viz_recommendation_prompt)
        recommendation = extract_json_from_response(recommendation_response)
        
        if not recommendation:
            recommendation = create_default_visualization_recommendation(df)
    except Exception as e:
        recommendation = create_default_visualization_recommendation(df)
    
    # Ensure color_scheme is valid
    if "specifications" in recommendation and "color_scheme" in recommendation["specifications"]:
        # Check if color_scheme is a valid matplotlib color
        if recommendation["specifications"]["color_scheme"] not in DEFAULT_COLORS:
            recommendation["specifications"]["color_scheme"] = "blue"
    
    return recommendation

def create_default_visualization_recommendation(df):
    """
    Create a default visualization recommendation when recommendation generation fails.
    
    Args:
        df (pd.DataFrame): The dataframe to visualize
    
    Returns:
        dict: Default visualization recommendation
    """
    return {
        "recommended_viz": "bar chart",
        "rationale": "Simple comparison of values",
        "specifications": {
            "x_axis": df.columns[0],
            "y_axis": df.columns[1] if len(df.columns) > 1 else df.columns[0],
            "grouping": None,
            "aggregation": "sum",
            "title": "Data Visualization",
            "x_label": df.columns[0],
            "y_label": df.columns[1] if len(df.columns) > 1 else df.columns[0],
            "annotations": [],
            "color_scheme": "blue"
        },
        "insights": ["Data pattern visualization"]
    }

def create_visualization(df, recommendation):
    """
    Create a visualization based on the recommendation.
    
    Args:
        df (pd.DataFrame): The dataframe to visualize
        recommendation (dict): Visualization recommendation details
    
    Returns:
        tuple: (figure, insights_text)
    """
    if not isinstance(recommendation, dict):
        recommendation = create_default_visualization_recommendation(df)
    
    viz_type = recommendation.get("recommended_viz", "bar chart").lower()
    
    if not isinstance(recommendation.get("specifications"), dict):
        specs = {
            "x_axis": df.columns[0],
            "y_axis": df.columns[1] if len(df.columns) > 1 else df.columns[0],
            "grouping": None,
            "aggregation": None,
            "title": "Data Visualization",
            "x_label": df.columns[0],
            "y_label": df.columns[1] if len(df.columns) > 1 else df.columns[0],
            "annotations": [],
            "color_scheme": "blue"
        }
    else:
        specs = recommendation["specifications"]
    
    # Create figure with compact size
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if "x_axis" not in specs or specs["x_axis"] not in df.columns:
        x_col = df.columns[0]
    else:
        x_col = specs["x_axis"]
        
    if "y_axis" not in specs or specs["y_axis"] not in df.columns:
        y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    else:
        y_col = specs["y_axis"]
    
    try:
        plot_df = prepare_data_for_visualization(df, specs, x_col, y_col)
    except Exception as e:
        plot_df = df.copy()
    
    try:
        color = specs.get("color_scheme", "blue")
        
        create_appropriate_plot(ax, plot_df, viz_type, x_col, y_col, color)
            
    except Exception as e:
        # Fallback to simple visualization
        ax.clear()
        try:
            first_col = df.columns[0]
            second_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            ax.bar(df[first_col].head(10), df[second_col].head(10), color='blue')
            ax.set_title("Fallback Visualization")
            ax.set_xlabel(first_col)
            ax.set_ylabel(second_col)
        except:
            ax.bar(['A', 'B', 'C'], [1, 2, 3], color='blue')
            ax.set_title("Sample Data")
            ax.set_xlabel("Category")
            ax.set_ylabel("Value")
    
    try:
        apply_visualization_styling(ax, plt, plot_df, specs, x_col)
    except Exception as e:
        pass
    
    insights = recommendation.get("insights", ["Data visualization"])
    if not isinstance(insights, list):
        insights = [str(insights)]
        
    insights_text = "\n".join([f"📊 {insight}" for insight in insights if insight])
    
    try:
        plt.tight_layout(pad=1.1)
    except:
        pass
    
    return fig, insights_text

def prepare_data_for_visualization(df, specs, x_col, y_col):
    """
    Prepare data for visualization, including any grouping or aggregation.
    
    Args:
        df (pd.DataFrame): The dataframe to visualize
        specs (dict): Visualization specifications
        x_col (str): X-axis column name
        y_col (str): Y-axis column name
    
    Returns:
        pd.DataFrame: Prepared dataframe for plotting
    """
    if specs.get("grouping") and specs.get("aggregation") and specs["grouping"] in df.columns:
        if specs["aggregation"] == "sum":
            agg_data = df.groupby(specs["grouping"])[y_col].sum().reset_index()
        elif specs["aggregation"] == "mean":
            agg_data = df.groupby(specs["grouping"])[y_col].mean().reset_index()
        elif specs["aggregation"] == "count":
            agg_data = df.groupby(specs["grouping"])[y_col].count().reset_index()
        else:
            agg_data = df  # Default to no aggregation
        
        plot_df = agg_data
        return plot_df
    else:
        return df.copy()

def create_appropriate_plot(ax, plot_df, viz_type, x_col, y_col, color):
    """
    Create the appropriate plot based on the visualization type.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to plot on
        plot_df (pd.DataFrame): The dataframe to visualize
        viz_type (str): Type of visualization
        x_col (str): X-axis column name
        y_col (str): Y-axis column name
        color (str): Color for the plot
    """
    if "bar" in viz_type:
        ax.bar(plot_df[x_col], plot_df[y_col], color=color)
        
    elif "line" in viz_type:
        if pd.api.types.is_datetime64_any_dtype(plot_df[x_col]):
            plot_df = plot_df.sort_values(by=x_col)
        ax.plot(plot_df[x_col], plot_df[y_col], marker='o', linestyle='-', color=color)
        
    elif "scatter" in viz_type:
        ax.scatter(plot_df[x_col], plot_df[y_col], color=color)
        
    elif "pie" in viz_type:
        positive_values = plot_df[y_col].apply(lambda x: max(0, x))
        if positive_values.sum() > 0:
            ax.pie(positive_values, labels=plot_df[x_col], autopct='%1.1f%%')
        else:
            ax.pie([1, 1, 1], labels=["A", "B", "C"], autopct='%1.1f%%')
        
    elif "histogram" in viz_type:
        ax.hist(plot_df[y_col], bins=10, color=color)
        
    elif "heatmap" in viz_type:
        create_heatmap(ax, plot_df, x_col, y_col)
        
    else:
        # Default to bar chart
        ax.bar(plot_df[x_col], plot_df[y_col], color=color)

def create_heatmap(ax, df, x_col, y_col, grouping_col=None, aggregation="mean"):
    """
    Create a heatmap visualization.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to plot on
        df (pd.DataFrame): The dataframe to visualize
        x_col (str): X-axis column name
        y_col (str): Y-axis column name
        grouping_col (str, optional): Column to group by. Defaults to None.
        aggregation (str, optional): Aggregation function. Defaults to "mean".
    """
    if grouping_col and grouping_col in df.columns:
        try:
            pivot_df = pd.pivot_table(
                df, 
                values=y_col, 
                index=x_col,
                columns=grouping_col,
                aggfunc=aggregation
            )
            
            if not pivot_df.empty and pivot_df.size > 0:
                im = ax.imshow(pivot_df, cmap='viridis')
                ax.set_xticks(range(len(pivot_df.columns)))
                ax.set_yticks(range(len(pivot_df.index)))
                ax.set_xticklabels(pivot_df.columns)
                ax.set_yticklabels(pivot_df.index)
                plt.colorbar(im, ax=ax)
            else:
                ax.bar(df[x_col], df[y_col], color="blue")
        except Exception as e:
            ax.bar(df[x_col], df[y_col], color="blue")
    else:
        ax.bar(df[x_col], df[y_col], color="blue")

def apply_visualization_styling(ax, plt, plot_df, specs, x_col):
    """
    Apply styling to the visualization.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to style
        plt (matplotlib.pyplot): Matplotlib pyplot object
        plot_df (pd.DataFrame): The dataframe being visualized
        specs (dict): Visualization specifications
        x_col (str): X-axis column name
    """
    title = specs.get("title", "Data Visualization")
    x_label = specs.get("x_label", x_col)
    y_label = specs.get("y_label")
    
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    
    # Make tick labels smaller
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Rotate x-axis labels if they're long
    plt.xticks(rotation=45 if len(str(plot_df[x_col].iloc[0])) > 5 else 0, ha='right')
    
    if pd.api.types.is_datetime64_any_dtype(plot_df[x_col]):
        plt.gcf().autofmt_xdate()
    
    for annotation in specs.get("annotations", []):
        if annotation:
            ax.annotate(annotation, xy=(0.5, 0.9), xycoords='axes fraction', 
                        ha='center', va='center', fontsize=8, color='red')