from utils.api import call_llama

def get_expert_analysis(query, df, data_summary, viz_recommendation):
    """
    Generate expert analysis based on the data and visualization.
    
    Args:
        query (str): User's data query
        df (pd.DataFrame): The dataframe being analyzed
        data_summary (str): Summary of the data
        viz_recommendation (dict): Visualization recommendation details
    
    Returns:
        str: Expert analysis of the data
    """
    analysis_prompt = f"""
    Query: "{query}"
    
    I've analyzed data with these characteristics:
    {data_summary}
    
    Visualization used: {viz_recommendation.get('recommended_viz', 'bar chart')}
    
    Based on this data and visualization, provide a concise expert analysis (3-5 sentences) that:
    1. Directly answers the user's query
    2. Highlights the most significant patterns or findings
    3. Offers one actionable insight or recommendation
    
    Keep your response conversational and avoid technical jargon.
    """
    
    try:
        final_analysis = call_llama(analysis_prompt)
    except Exception as e:
        # Fallback response if analysis generation fails
        insights = viz_recommendation.get('insights', ['The data shows interesting patterns'])
        insight = insights[0] if isinstance(insights, list) and len(insights) > 0 else "The data shows interesting patterns"
        final_analysis = f"Based on the visualization, we can see key patterns in the data that address your query. {insight}"
    
    return final_analysis