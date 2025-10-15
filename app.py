import pandas as pd
import numpy as np
import io
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Academic Stress Dashboard")

# --- Data Loading and Cleaning ---
# Define the file path
DATA_FILE_PATH = "academic Stress level - maintainance 1.csv"

# Load the data directly from the CSV file
try:
    df = pd.read_csv(DATA_FILE_PATH)
except FileNotFoundError:
    st.error(f"Error: The file '{DATA_FILE_PATH}' was not found. Please ensure it is in the same directory as 'app.py'.")
    st.stop()
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# 1. Rename/Simplify Columns
new_cols = {
    'Your Academic Stage': 'Academic_Stage',
    'Peer pressure': 'Peer_Pressure',
    'Academic pressure from your home': 'Home_Pressure',
    'Study Environment': 'Study_Environment',
    'What coping strategy you use as a student?': 'Coping_Strategy',
    'Do you have any bad habits like smoking, drinking on a daily basis?': 'Bad_Habits',
    'What would you rate the academic  competition in your student life': 'Competition_Rating',
    'Rate your academic stress index ': 'Stress_Index'
}
df.rename(columns=new_cols, inplace=True)

# 2. Standardize Text
df['Coping_Strategy'] = df['Coping_Strategy'].str.replace(r'\(.*\)', '', regex=True).str.strip().replace('', 'Unknown')

# 3. Ensure Numeric Types
numeric_cols = ['Peer_Pressure', 'Home_Pressure', 'Competition_Rating', 'Stress_Index']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaN in key columns for clean analysis
df.dropna(subset=numeric_cols + ['Academic_Stage', 'Coping_Strategy', 'Study_Environment'], inplace=True)


# --- NEW: Mock LLM Function for General/Conversational Questions ---

def mock_general_llm_response(query):
    query = query.lower()
    
    if "hello" in query or "hi" in query or "who are you" in query or "what do you do" in query:
        return ("Hello! 👋 I'm your **Academic Stress Data Coach**. I analyze the student survey data you see on the dashboard to provide tailored, data-driven advice on managing stress.")
    elif "study" in query or "focus" in query or "concentration" in query:
        return ("To boost your focus, try the **Pomodoro Technique** (short bursts of focused work followed by breaks) or minimizing digital distractions. Also, check the dashboard's **Study Environment** chart for peer insights!")
    elif "thank" in query or "bye" in query:
        return "You're very welcome! Feel free to ask more questions anytime. Remember to take a break when you need one! 🎓"
    elif "sleep" in query or "routine" in query:
        return ("Prioritizing sleep hygiene is essential for mental clarity and stress reduction. Aim for consistent sleep times to regulate your body clock and improve concentration.")
    else:
        return ("That's an interesting question! As a **Data Coach**, I specialize in analyzing the student dataset. Try asking me about the **best coping strategy** for your peers or the **top risk factor** for stress in this group.")


# --- Reworked Chatbot Logic (Now includes conversational fallback) ---

def chatbot_response(query, current_df):
    query = query.lower()
    
    # Check if data is sufficient for data-driven response
    if current_df.shape[0] < 5:
        # Fallback if the data filter is too restrictive
        return "I need a minimum of 5 data points in the current filtered view to provide a reliable data-driven recommendation. Try broadening your filter."

    # --- PRIORITY 1: DATA-DRIVEN RESPONSES ---
    if "coping" in query or "strategy" in query or "recommendation" in query:
        current_coping_stress = current_df.groupby('Coping_Strategy')['Stress_Index'].mean().sort_values(ascending=True)
        
        if current_coping_stress.empty:
            return "No coping strategy data available in the current filter."

        best_strategy = current_coping_stress.index[0]
        best_avg_stress = current_coping_stress.iloc[0]
        
        return (f"📊 **Data Insight:** Based on this filtered group (Avg Stress: {current_df['Stress_Index'].mean():.2f}):\n"
                f"The **most effective coping strategy** is **'{best_strategy}'** (Avg. Stress: {best_avg_stress:.2f}).\n"
                f"Recommendation: For maximum stress relief, try incorporating **'{best_strategy}'** into your routine.")

    elif "risk" in query or "factor" in query or "pressure" in query:
        correlations = current_df[['Peer_Pressure', 'Home_Pressure', 'Competition_Rating', 'Stress_Index']].corr()['Stress_Index'].drop('Stress_Index').abs().sort_values(ascending=False)
        top_risk_factor = correlations.index[0]
        
        return (f"⚠️ **Data Insight:** The strongest factor correlated with high stress in this group is **{top_risk_factor.replace('_', ' ')}** (Absolute Correlation: {correlations.iloc[0]:.2f}).\n"
                f"This suggests that interventions targeting **{top_risk_factor.replace('_', ' ')}** may provide the best relief for this cohort.")
    
    elif "average" in query or "mean" in query or "stress index" in query:
        avg = current_df['Stress_Index'].mean()
        return f"📈 **Data Insight:** The **Average Academic Stress Index** for the current dataset filter is **{avg:.2f}** (on a scale of 1-5)."

    # --- PRIORITY 2: CONVERSATIONAL FALLBACK ---
    else:
        return mock_general_llm_response(query)

# --- Initialize Session State for Chat History ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_input_key' not in st.session_state:
    st.session_state.chat_input_key = "" # For clearing the input field

# --- Dashboard Layout ---
st.title("📊 Academic Stress Level Analysis Dashboard")

tab1, tab2, tab3 = st.tabs(["🧑‍🎓 Tab 1: Student View (Personal Insights)", 
                           "👩‍🏫 Tab 2: Counselors View (Risk Factors)", 
                           "🔬 Tab 3: Researchers View (Statistical Trends)"])

#Global Chatbot Sidebar
with st.sidebar:
    st.header("🤖 Your Interactive Data Coach")
    st.markdown("Ask any question! The Coach provides **data-driven insights** based on the current filters or **general study advice**.")
    
    # --- Filter for Chatbot Context ---
    stages = ['All'] + sorted(df['Academic_Stage'].unique().tolist())
    selected_stage = st.selectbox("1. Filter Data Context for Chatbot", stages, key='student_stage_filter')
    
    st.markdown("---")

    # Determine the DataFrame to use for the chatbot's analysis
    chat_df = df.copy() 
    if st.session_state.student_stage_filter != 'All':
        chat_df = chat_df[chat_df['Academic_Stage'] == st.session_state.student_stage_filter]
        
    # --- Reworked Chat Interface using st.chat_input ---
    
    # st.chat_input handles the state and clearing automatically.
    # It must be called outside of any columns or containers that don't cover the full width, but st.sidebar is fine.
    chatbot_query = st.chat_input("Ask a question (e.g., 'best coping strategy' or 'How to focus')", key='sidebar_chat_input')
    
    if chatbot_query:
        with st.spinner('Thinking...'):
            response = chatbot_response(chatbot_query, chat_df)
            
            # Store the interaction
            st.session_state.chat_history.append({"user": chatbot_query, "assistant": response})
            # st.chat_input handles the rerun automatically
            
    st.markdown("---")
    st.subheader("Conversation History")
    
    # Display the chat history using the native chat message container
    # We display them in the order they were received (top to bottom)
    for message in st.session_state.chat_history: 
        with st.chat_message("user"):
            st.markdown(message["user"])
        with st.chat_message("assistant"):
            st.markdown(message["assistant"])


# --- Tab 1: Student View ---
with tab1:
    st.header("Personal Academic Stress Insights")
    st.markdown("Use this view to compare your experience to your peers and identify effective coping methods.")
    
    col1, col2 = st.columns([1, 4])
    
    # Re-filter student_df based on the stage selected in the sidebar (key='student_stage_filter')
    student_df = df.copy()
    if st.session_state.student_stage_filter != 'All':
        student_df = student_df[df['Academic_Stage'] == st.session_state.student_stage_filter]
    
    with col1:
        st.metric(label=f"Average Stress Index ({st.session_state.student_stage_filter})", 
                  value=f"{student_df['Stress_Index'].mean():.2f}", 
                  delta=f"{student_df['Stress_Index'].mean() - df['Stress_Index'].mean():.2f} vs All")
    
    with col2:
        # Key Visualization 1: Academic Stress Index Distribution
        avg_stress = student_df['Stress_Index'].mean()
        fig_dist = px.histogram(student_df, x="Stress_Index", nbins=5, 
                                title=f'Distribution of Academic Stress Index (Avg: {avg_stress:.2f})',
                                color_discrete_sequence=['#4682B4'])
        fig_dist.update_layout(xaxis_title="Stress Index (1=Low, 5=High)", yaxis_title="Count")
        fig_dist.add_vline(x=avg_stress, line_width=2, line_dash="dash", line_color="red", annotation_text="Average", annotation_position="top left")
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    
    # Key Visualization 2: Average Stress Index by Coping Strategy
    coping_stress = student_df.groupby('Coping_Strategy')['Stress_Index'].mean().reset_index().sort_values('Stress_Index', ascending=True)
    
    fig_coping = px.bar(coping_stress, x='Stress_Index', y='Coping_Strategy', orientation='h',
                        title='Average Stress Index by Coping Strategy (Lower is Better)',
                        color_discrete_sequence=['#228B22'])
    fig_coping.update_layout(yaxis_title="Coping Strategy", xaxis_title="Average Stress Index")
    st.plotly_chart(fig_coping, use_container_width=True)

# --- Tab 2: Counselors View ---
with tab2:
    st.header("Risk Factor Identification & Cohort Analysis")
    st.markdown("Filter to pinpoint high-risk cohorts and the factors contributing to their elevated stress.")
    
    # Interactive Filters
    filter_cols = st.columns(3)
    
    stages = sorted(df['Academic_Stage'].unique().tolist())
    selected_stages = filter_cols[0].multiselect("Filter Academic Stage", stages, default=stages, key='counselor_stage_filter')
    
    environments = sorted(df['Study_Environment'].unique().tolist())
    selected_environments = filter_cols[1].multiselect("Filter Study Environment", environments, default=environments, key='counselor_env_filter')
    
    pressure_range = filter_cols[2].slider("Filter Peer Pressure Range (1=Low, 5=High)", 
                                           min_value=1, max_value=5, value=(1, 5), step=1, key='counselor_pressure_slider')
    
    counselor_df = df[
        (df['Academic_Stage'].isin(selected_stages)) &
        (df['Study_Environment'].isin(selected_environments)) &
        (df['Peer_Pressure'].between(pressure_range[0], pressure_range[1]))
    ]
    
    st.metric(label=f"Average Stress Index for Filtered Cohort", 
              value=f"{counselor_df['Stress_Index'].mean():.2f}", 
              delta=f"{counselor_df['Stress_Index'].mean() - df['Stress_Index'].mean():.2f} vs All")
    
    st.markdown("---")
    
    # Key Visualization 1: Stress Index by Academic Stage (Box Plot)
    fig_box = px.box(counselor_df, x='Academic_Stage', y='Stress_Index', 
                     color='Academic_Stage',
                     title="Stress Index Distribution by Academic Stage (Overall Priority)",
                     color_discrete_sequence=px.colors.qualitative.Dark24)
    fig_box.update_layout(xaxis_title="Academic Stage", yaxis_title="Stress Index Rating")
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")

    # Key Visualization 2: Average Stress Index by Environmental Factors
    pressure_factors = ['Peer_Pressure', 'Home_Pressure', 'Competition_Rating']
    avg_pressure_stress = pd.concat([
        counselor_df.groupby('Study_Environment')['Stress_Index'].mean().rename('Avg_Stress').reset_index().assign(Factor='Study_Environment'),
        counselor_df[pressure_factors + ['Stress_Index']].melt(id_vars='Stress_Index', value_vars=pressure_factors, var_name='Factor', value_name='Rating').groupby(['Factor', 'Rating'])['Stress_Index'].mean().rename('Avg_Stress').reset_index()
    ], ignore_index=True)
    
    fig_bar = px.bar(avg_pressure_stress, x='Avg_Stress', y='Rating', color='Factor', orientation='h',
                     facet_row='Factor',
                     title='Average Stress Index by Key Contributing Factors (1-5 Rating)',
                     labels={'Avg_Stress': 'Average Stress Index', 'Rating': 'Rating/Environment'})
    fig_bar.update_layout(yaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig_bar, use_container_width=True)


# --- Tab 3: Researchers View ---
with tab3:
    st.header("Global Statistical Trends and Correlations")
    st.markdown("This view provides an overall, unfiltered statistical look at the dataset for macro-analysis.")
    
    # Key Visualization 1: Correlation Heatmap
    correlation_df = df[numeric_cols].corr()
    fig_corr = px.imshow(correlation_df, 
                         text_auto=".2f", 
                         color_continuous_scale=px.colors.sequential.RdBu,
                         aspect="auto",
                         title="Correlation Heatmap of Numerical Stress Factors")
    fig_corr.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")

    col_res1, col_res2 = st.columns(2)
    
    # Key Visualization 2: Distribution of Coping Strategies
    coping_counts = df['Coping_Strategy'].value_counts().reset_index()
    coping_counts.columns = ['Coping_Strategy', 'Count']
    fig_coping_dist = px.pie(coping_counts, values='Count', names='Coping_Strategy',
                            title='Distribution of Coping Strategies Used',
                            hole=0.3,
                            color_discrete_sequence=px.colors.qualitative.Vivid)
    col_res1.plotly_chart(fig_coping_dist, use_container_width=True)
    
    # Key Visualization 3: Distribution of Bad Habits
    habits_counts = df['Bad_Habits'].value_counts().reset_index()
    habits_counts.columns = ['Bad_Habits', 'Count']
    fig_habits_dist = px.bar(habits_counts, x='Bad_Habits', y='Count',
                            title='Distribution of Reported Bad Habits',
                            color='Bad_Habits',
                            color_discrete_sequence=['#FF6347', '#3CB371', '#808080'])
    col_res2.plotly_chart(fig_habits_dist, use_container_width=True)