import streamlit as st
from modules.data_retrieval import get_data_for_query
from modules.visualization import get_visualization_recommendation, create_visualization
from modules.analysis import get_expert_analysis
from utils.streamlit_utils import set_streamlit_page_config
import matplotlib.pyplot as plt
import pandas as pd

def main():
    set_streamlit_page_config()
    
    st.title("🧠 Smart Data Analysis with LLaMA 3.1")
    st.markdown("Ask me anything about data, and I'll find or generate relevant data, recommend the best visualization, and create insights.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "I'm your data analysis expert. What data would you like to explore today?"}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask anything (e.g., 'Show me Tesla stock price trends')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            error_log = []
            success = True
            
            message_placeholder.markdown("🔍 Analyzing your request...")
            
            try:
                with st.spinner("Finding relevant data..."):
                    data_df, source_info, is_synthetic = get_data_for_query(prompt)
                    
                    st.info(source_info)
                    
                    with st.expander("View Data Sample"):
                        st.dataframe(data_df.head(10))
            except Exception as e:
                st.error(f"Error getting data: {e}")
                error_log.append(f"Data acquisition error: {e}")
                
                try:
                    data_df = pd.DataFrame({
                        'Category': ['A', 'B', 'C', 'D', 'E'],
                        'Value': [10, 24, 15, 32, 18]
                    })
                    source_info = "Using sample data for demonstration"
                    is_synthetic = True
                    
                    st.warning("Using sample data instead")
                    with st.expander("View Sample Data"):
                        st.dataframe(data_df)
                except Exception as e2:
                    st.error(f"Critical error creating fallback data: {e2}")
                    success = False
            
            if success:
                try:
                    with st.spinner("Determining the best visualization..."):
                        viz_recommendation = get_visualization_recommendation(prompt, data_df)
                        
                        explanation = (
                            f"**Visualization Selected:** {viz_recommendation['recommended_viz']}\n\n"
                            f"**Why this works best:** {viz_recommendation['rationale']}"
                        )
                        st.markdown(explanation)
                except Exception as e:
                    st.error(f"Error getting visualization recommendation: {e}")
                    error_log.append(f"Viz recommendation error: {e}")
                    
                    viz_recommendation = {
                        "recommended_viz": "bar chart",
                        "rationale": "Simple comparison of values",
                        "specifications": {
                            "x_axis": data_df.columns[0],
                            "y_axis": data_df.columns[1] if len(data_df.columns) > 1 else data_df.columns[0],
                            "grouping": None,
                            "aggregation": None,
                            "title": "Data Visualization",
                            "x_label": data_df.columns[0],
                            "y_label": data_df.columns[1] if len(data_df.columns) > 1 else data_df.columns[0],
                            "annotations": [],
                            "color_scheme": "blue"
                        },
                        "insights": ["Data visualization"]
                    }
                    st.warning("Using default visualization settings")
                
                try:
                    with st.spinner("Creating visualization..."):
                        fig, insights_text = create_visualization(data_df, viz_recommendation)
                        
                        # Create a centered column with constrained width for the visualization
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col2:
                            st.pyplot(fig)
                        
                        st.markdown("### Key Insights:")
                        st.markdown(insights_text)
                except Exception as e:
                    st.error(f"Error creating visualization: {e}")
                    error_log.append(f"Visualization creation error: {e}")
                    
                    try:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.bar(data_df.iloc[:5, 0], data_df.iloc[:5, 1] if len(data_df.columns) > 1 else range(5))
                        ax.set_title("Simple Data Overview")
                        
                        # Create a centered column with constrained width for the visualization
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col2:
                            st.pyplot(fig)
                            
                        st.warning("⚠️ Showing simplified visualization")
                    except:
                        st.error("Unable to create visualization")
                
                try:
                    with st.spinner("Generating expert analysis..."):
                        data_summary = f"Data shape: {data_df.shape[0]} rows, {data_df.shape[1]} columns. Columns: {', '.join(data_df.columns.tolist())}"
                        
                        final_analysis = get_expert_analysis(prompt, data_df, data_summary, viz_recommendation)
                        
                        full_response = f"""
                        {final_analysis}
                        
                        Would you like to explore a different aspect of this data or analyze something else?
                        """
                        
                        message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Error generating final analysis: {e}")
                    error_log.append(f"Final analysis error: {e}")
                    
                    fallback_message = """
                    I've analyzed the data and created a visualization based on your request.
                    
                    The visualization shows the key patterns in the data. While I encountered some issues generating a detailed analysis, you can see the main trends in the chart above.
                    
                    Would you like to explore a different aspect of this data or analyze something else?
                    """
                    message_placeholder.markdown(fallback_message)
                    
                response_content = full_response if 'full_response' in locals() else fallback_message
                st.session_state.messages.append({"role": "assistant", "content": response_content})
            
            else:
                error_message = """
                I'm sorry, but I encountered a critical error processing your request. Here's what you can try:
                
                1. Rephrase your question to be more specific about the data you're looking for
                2. Ask about a different type of data
                3. Try a simpler query about commonly available data
                
                For example: "Show me monthly sales data for a retail store" or "Compare population growth of major cities"
                """
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                
            if error_log:
                with st.expander("Debug Information", expanded=False):
                    st.write("The system encountered some issues while processing your request:")
                    for i, error in enumerate(error_log, 1):
                        st.write(f"{i}. {error}")

if __name__ == "__main__":
    main()