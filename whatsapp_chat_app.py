import streamlit as st
import preprocessing_data
import functions
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import io
from PIL import Image
from wordcloud import WordCloud
import requests


st.markdown("""
<style>
    /* Dark background with subtle pattern */
    .stApp {
        background-color: #121212;
        background-image: 
            linear-gradient(rgba(18, 18, 18, 0.8), rgba(18, 18, 18, 0.8)),
            url('https://www.transparenttextures.com/patterns/cubes.png');
        color: #e0e0e0;
    }

    /* Sidebar styling */
    .stSidebar {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    /* Metric cards */
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        padding: 15px;
        text-align: center;
        color: #ffffff;
        border: 1px solid #333;
    }

    /* Text colors */
    .stMarkdown, .stDataFrame, .stMetric {
        color: #e0e0e0 !important;
    }

    /* Chart backgrounds */
    .plotly-chart {
        background-color: #1e1e1e;
        border-radius: 10px;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)
# Together.ai Chat Summarizer Function
TOGETHER_API_KEY = "e1eae2a0ac9d5686959df2c117dd839be0663ccc63d136d119bccd975a131ba4"  # Replace with your key

# def summarize_chat_with_together(chat_text):
#     prompt = f"""You are a smart assistant. Summarize the following WhatsApp conversation:\n\n{chat_text}\n\nSummary:"""
#     print("⏳ Sending request to Together API...")
#     print(f"Prompt length: {len(chat_text)}")
#     response = requests.post(
#         "https://api.together.xyz/inference",
#         headers={
#             "Authorization": f"Bearer {TOGETHER_API_KEY}",
#             "Content-Type": "application/json"
#         },
#         json={
#             "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
#             "prompt": prompt,
#             "max_tokens": 300,
#             "temperature": 0.7,
#             "stop": None,
#             "top_p": 0.95,
#             "top_k": 50
#         }
#     )
#     print("✅ Got response from API:")
#     print(response.status_code)
#     print(response.text)


#     try:
#         return response.json()['output']['choices'][0]['text']
#     except Exception as e:
#         return "❌ Error generating summary. Please try again."
# def summarize_chat_with_together(chat_text):
#     prompt = f"""Summarize this WhatsApp chat:\n\n{chat_text}\n\nSummary:"""

#     response = requests.post(
#         "https://api.together.xyz/inference",
#         headers={
#             "Authorization": f"Bearer {TOGETHER_API_KEY}",
#             "Content-Type": "application/json"
#         },
#         json={
#             "model": "mistralai/Mistral-7B-Instruct-v0.1",  # ✅ This works!
#             "prompt": prompt,
#             "max_tokens": 300,
#             "temperature": 0.7,
#             "top_p": 0.95,
#             "top_k": 50
#         }
#     )

#     try:
#         return response.json()['output']['choices'][0]['text']
#     except Exception as e:
#         print("API failed:", response.status_code, response.text)
#         return "❌ Could not generate summary. Please try again."
def summarize_chat_with_together(chat_text):
    prompt = f"""You are an AI assistant. Summarize the following WhatsApp chat between friends into clear bullet points or short sentences capturing the main topics, decisions, and any emotional tone:

Chat:
{chat_text}

Summary:"""

    response = requests.post(
        "https://api.together.xyz/inference",
        headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        }
    )

    try:
        return response.json()['output']['choices'][0]['text']
    except Exception as e:
        print("Error from Together API:", response.status_code, response.text)
        return "❌ Could not generate summary."




# Function to create WordCloud image
def create_wordcloud_image(selected_user, df):
    # Filter dataframe based on selected user
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Combine all messages
    text = " ".join(df['message'])

    # Generate WordCloud
    wordcloud = WordCloud(width=800,
                          height=400,
                          background_color='white',
                          min_font_size=10).generate(text)

    # Convert WordCloud to PIL Image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Open as PIL Image
    pil_image = Image.open(buf)

    return pil_image


# Main App
def main():
    st.title("🔍 Chat Analyser ")
    st.markdown("### Unlock the secrets of your WhatsApp conversations")

    # # Sidebar with a modern look
    # st.sidebar.image("https://via.placeholder.com/150", width=200)
    # st.sidebar.markdown("## 🔍 Conversation Analyzer")

    # File Upload
    uploaded_file = st.sidebar.file_uploader("📤 Upload Chat Export")

    if uploaded_file is not None:
        # Preprocessing
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessing_data.preprocess(data)
        with st.expander("🔍 Preview Chat Data", expanded=False):
            st.dataframe(df.head(40))

        # User Selection with a modern dropdown
        user_list = df['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.sidebar.selectbox("🧑 Select User", user_list,
                                             help="Choose a specific user or view overall stats")

        # Analysis Button with custom styling
        if st.sidebar.button("🚀 Analyze Conversation", type="primary"):
            # Create a container for metrics
            with st.container():
                st.markdown("## 📊 Conversation Metrics")

                # Fetch stats with Plotly cards
                num_messages, words, num_media_messages, num_links = functions.fetch_stats(selected_user, df)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Total Messages</h3>
                        <h1>{num_messages}</h1>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Total Words</h3>
                        <h1>{words}</h1>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Media Shared</h3>
                        <h1>{num_media_messages}</h1>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Links Shared</h3>
                        <h1>{num_links}</h1>
                    </div>
                    """, unsafe_allow_html=True)

            # Interactive Plotly Visualizations
            st.markdown("## 📈 Conversation Trends")

            # Monthly Timeline with Plotly
            timeline = functions.monthly_timeline(selected_user, df)
            fig_monthly = px.line(timeline, x='time', y='message',
                                  title='Monthly Message Trend',
                                  labels={'time': 'Month', 'message': 'Number of Messages'})
            st.plotly_chart(fig_monthly, use_container_width=True)

            # Daily Timeline
            daily_timeline = functions.daily_timeline(selected_user, df)
            fig_daily = px.line(daily_timeline, x='only_date', y='message',
                                title='Daily Message Trend',
                                labels={'only_date': 'Date', 'message': 'Number of Messages'})
            st.plotly_chart(fig_daily, use_container_width=True)

            # Activity Map
            st.markdown("## 🗓️ Activity Insights")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Most Busy Day")
                busy_day = functions.week_activity_map(selected_user, df)
                fig_busy_day = px.bar(x=busy_day.index, y=busy_day.values,
                                      title='Day-wise Activity',
                                      labels={'x': 'Day', 'y': 'Message Count'})
                st.plotly_chart(fig_busy_day)

            with col2:
                st.markdown("### Most Busy Month")
                busy_month = functions.month_activity_map(selected_user, df)
                fig_busy_month = px.bar(x=busy_month.index, y=busy_month.values,
                                        title='Month-wise Activity',
                                        labels={'x': 'Month', 'y': 'Message Count'})
                st.plotly_chart(fig_busy_month)
                
            # Summarize Chat Button
            st.markdown("## 🧠 AI Chat Summary")
            if st.button("📋 Generate Summary with AI"):
                with st.spinner("Summarizing conversation..."):
                     
                     full_chat = "\n".join(df.apply(lambda row: f"{row['user']}: {row['message']}", axis=1))
                     summary = summarize_chat_with_together(full_chat[:15000])  # Truncate if chat is long
                     st.success("Summary generated successfully!")
                     st.markdown("### 📝 Summary:")
                     st.write(summary)


            # Word Cloud
            st.markdown("## 💬 Conversation Highlights")
            df_wc = create_wordcloud_image(selected_user, df)
            st.image(df_wc, caption="Most Frequent Words", use_container_width=True)

            # Emoji Analysis
            emoji_df = functions.emoji_helper(selected_user, df)
            st.markdown("## 😀 Emoji Insights")

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig_emoji = px.pie(emoji_df, values=1, names=0,
                                   title='Emoji Distribution')
                st.plotly_chart(fig_emoji)


# Ensure the main function is called
if __name__ == "__main__":
    main()