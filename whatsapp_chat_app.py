# import streamlit as st
# import preprocessing_data
# import functions
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objs as go
# import io
# from PIL import Image
# from wordcloud import WordCloud
# import requests
#
#
# st.markdown("""
# <style>
#     /* Dark background with subtle pattern */
#     .stApp {
#         background-color: #121212;
#         background-image:
#             linear-gradient(rgba(18, 18, 18, 0.8), rgba(18, 18, 18, 0.8)),
#             url('https://www.transparenttextures.com/patterns/cubes.png');
#         color: #e0e0e0;
#     }
#
#     /* Sidebar styling */
#     .stSidebar {
#         background-color: #1e1e1e;
#         color: #ffffff;
#     }
#
#     /* Metric cards */
#     .metric-card {
#         background-color: #1e1e1e;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.5);
#         padding: 15px;
#         text-align: center;
#         color: #ffffff;
#         border: 1px solid #333;
#     }
#
#     /* Text colors */
#     .stMarkdown, .stDataFrame, .stMetric {
#         color: #e0e0e0 !important;
#     }
#
#     /* Chart backgrounds */
#     .plotly-chart {
#         background-color: #1e1e1e;
#         border-radius: 10px;
#     }
#
#     /* Headings */
#     h1, h2, h3, h4, h5, h6 {
#         color: #ffffff !important;
#     }
# </style>
# """, unsafe_allow_html=True)
# # Together.ai Chat Summarizer Function
# TOGETHER_API_KEY = "e1eae2a0ac9d5686959df2c117dd839be0663ccc63d136d119bccd975a131ba4"  # Replace with your key
#
#
# def summarize_chat_with_together(chat_text):
#     import streamlit as st
#     prompt = f"""You are an AI assistant chatbot . Summarize the following WhatsApp chat between friends into clear bullet points or short sentences capturing the main topics, decisions, and any emotional tone:
#
# Chat:
# {chat_text}
#
# Summary:"""
#
#     st.write("🔍 [DEBUG] Prompt length:", len(prompt))
#     st.write("🔍 [DEBUG] First 500 chars of prompt:", prompt[:500])
#
#     try:
#         response = requests.post(
#             "https://api.together.xyz/inference",
#             headers={
#                 "Authorization": f"Bearer {TOGETHER_API_KEY}",
#                 "Content-Type": "application/json"
#             },
#             json={
#                 "model": "mistralai/Mistral-7B-Instruct-v0.1",
#                 "prompt": prompt,
#                 "max_tokens": 300,
#                 "temperature": 0.7,
#                 "top_p": 0.9,
#                 "top_k": 50
#             }
#         )
#         st.write("🔍 [DEBUG] API called, status code:", response.status_code)
#         st.write("🔍 [DEBUG] API response text:", response.text[:500])
#         output = response.json()
#         st.write("🔍 [DEBUG] API response JSON:", output)
#         return output['output']['choices'][0]['text']
#     except Exception as e:
#         st.write("❌ [DEBUG] Error from Together API:", response.status_code if 'response' in locals() else None, str(e))
#         return "❌ Could not generate summary."
#
#
#
#
# # Function to create WordCloud image
# def create_wordcloud_image(selected_user, df):
#     # Filter dataframe based on selected user
#     if selected_user != 'Overall':
#         df = df[df['user'] == selected_user]
#
#     # Combine all messages
#     text = " ".join(df['message'])
#
#     # Generate WordCloud
#     wordcloud = WordCloud(width=800,
#                           height=400,
#                           background_color='white',
#                           min_font_size=10).generate(text)
#
#     # Convert WordCloud to PIL Image
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#
#     # Save to a buffer
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
#     buf.seek(0)
#
#     # Open as PIL Image
#     pil_image = Image.open(buf)
#
#     return pil_image
#
#
# # Main App
# def main():
#     st.title("🔍 Chat Analyser ")
#     st.markdown("### Unlock the secrets of your WhatsApp conversations")
#
#     # File Upload
#     uploaded_file = st.sidebar.file_uploader("📤 Upload Chat Export")
#
#     if uploaded_file is not None:
#         # Preprocessing
#         bytes_data = uploaded_file.getvalue()
#         data = bytes_data.decode("utf-8")
#         df = preprocessing_data.preprocess(data)
#         with st.expander("🔍 Preview Chat Data", expanded=False):
#             st.dataframe(df.head(40))
#
#         # User Selection with a modern dropdown
#         user_list = df['user'].unique().tolist()
#         if 'group_notification' in user_list:
#             user_list.remove('group_notification')
#         user_list.sort()
#         user_list.insert(0, "Overall")
#
#         selected_user = st.sidebar.selectbox("🧑 Select User", user_list,
#                                              help="Choose a specific user or view overall stats")
#
#         # Analysis Button with custom styling
#         if st.sidebar.button("🚀 Analyze Conversation", type="primary"):
#             # Create a container for metrics
#             with st.container():
#                 st.markdown("## 📊 Conversation Metrics")
#
#                 # Fetch stats with Plotly cards
#                 num_messages, words, num_media_messages, num_links = functions.fetch_stats(selected_user, df)
#
#                 col1, col2, col3, col4 = st.columns(4)
#
#                 with col1:
#                     st.markdown(f"""
#                     <div class='metric-card'>
#                         <h3>Total Messages</h3>
#                         <h1>{num_messages}</h1>
#                     </div>
#                     """, unsafe_allow_html=True)
#
#                 with col2:
#                     st.markdown(f"""
#                     <div class='metric-card'>
#                         <h3>Total Words</h3>
#                         <h1>{words}</h1>
#                     </div>
#                     """, unsafe_allow_html=True)
#
#                 with col3:
#                     st.markdown(f"""
#                     <div class='metric-card'>
#                         <h3>Media Shared</h3>
#                         <h1>{num_media_messages}</h1>
#                     </div>
#                     """, unsafe_allow_html=True)
#
#                 with col4:
#                     st.markdown(f"""
#                     <div class='metric-card'>
#                         <h3>Links Shared</h3>
#                         <h1>{num_links}</h1>
#                     </div>
#                     """, unsafe_allow_html=True)
#
#             # Interactive Plotly Visualizations
#             st.markdown("## 📈 Conversation Trends")
#
#             # Monthly Timeline with Plotly
#             timeline = functions.monthly_timeline(selected_user, df)
#             fig_monthly = px.line(timeline, x='time', y='message',
#                                   title='Monthly Message Trend',
#                                   labels={'time': 'Month', 'message': 'Number of Messages'})
#             st.plotly_chart(fig_monthly, use_container_width=True)
#
#             # Daily Timeline
#             daily_timeline = functions.daily_timeline(selected_user, df)
#             fig_daily = px.line(daily_timeline, x='only_date', y='message',
#                                 title='Daily Message Trend',
#                                 labels={'only_date': 'Date', 'message': 'Number of Messages'})
#             st.plotly_chart(fig_daily, use_container_width=True)
#
#             # Activity Map
#             st.markdown("## 🗓️ Activity Insights")
#             col1, col2 = st.columns(2)
#
#             with col1:
#                 st.markdown("### Most Busy Day")
#                 busy_day = functions.week_activity_map(selected_user, df)
#                 fig_busy_day = px.bar(x=busy_day.index, y=busy_day.values,
#                                       title='Day-wise Activity',
#                                       labels={'x': 'Day', 'y': 'Message Count'})
#                 st.plotly_chart(fig_busy_day)
#
#             with col2:
#                 st.markdown("### Most Busy Month")
#                 busy_month = functions.month_activity_map(selected_user, df)
#                 fig_busy_month = px.bar(x=busy_month.index, y=busy_month.values,
#                                         title='Month-wise Activity',
#                                         labels={'x': 'Month', 'y': 'Message Count'})
#                 st.plotly_chart(fig_busy_month)
#
#             # Summarize Chat Button
#             st.markdown("## 🧠 AI Chat Summary")
#             if st.button("📋 Generate Summary with AI"):
#                 with st.spinner("Summarizing conversation..."):
#                     chat_df = df if selected_user == "Overall" else df[df['user'] == selected_user]
#                     st.write("🔍 [DEBUG] chat_df shape:", chat_df.shape)
#                     st.write("🔍 [DEBUG] First 5 rows of chat_df:", chat_df.head())
#                     full_chat = "\n".join(chat_df.apply(lambda row: f"{row['user']}: {row['message']}", axis=1))
#                     st.write("🔍 [DEBUG] First 500 chars of full_chat:", full_chat[:500])
#                     summary = summarize_chat_with_together(full_chat[:15000])
#                     st.success("Summary generated successfully!")
#                     st.markdown("### 📝 Summary:")
#                     st.write(summary)
#
#
#             # Word Cloud
#             st.markdown("## 💬 Conversation Highlights")
#             df_wc = create_wordcloud_image(selected_user, df)
#             st.image(df_wc, caption="Most Frequent Words", use_container_width=True)
#
#             # Emoji Analysis
#             emoji_df = functions.emoji_helper(selected_user, df)
#             st.markdown("## 😀 Emoji Insights")
#
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.dataframe(emoji_df)
#             with col2:
#                 fig_emoji = px.pie(emoji_df, values=1, names=0,
#                                    title='Emoji Distribution')
#                 st.plotly_chart(fig_emoji)
#
#
# # Ensure the main function is called
# if __name__ == "__main__":
#     main()
# import streamlit as st
# import preprocessing_data
# import functions
# import matplotlib.pyplot as plt
# import io
# from PIL import Image
# from wordcloud import WordCloud
# import plotly.express as px
# import google.generativeai as genai
# import os
# import pandas as pd
# from dotenv import load_dotenv
#
# # --- Load Environment Variables ---
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#
# # --- Page Configuration and Styling ---
# st.set_page_config(layout="wide", page_title="Chat Analyser")
#
# st.markdown("""
# <style>
#     /* Your existing CSS styles */
#     .stApp { background-color: #121212; /* ... etc */ }
# </style>
# """, unsafe_allow_html=True)
#
#
# # --- AI Chatbot Functions ---
# def get_relevant_context(question, df, top_k=5):
#     """
#     Searches the DataFrame for messages relevant to the user's question.
#     """
#     # A simple keyword search. For better results, NLP techniques like embeddings could be used.
#     keywords = [word for word in question.lower().split() if len(word) > 3]
#     if not keywords:
#         return "No relevant context found."
#
#     # Find rows where the message contains any of the keywords
#     # This creates a boolean Series for each keyword, which are then combined with | (OR)
#     # and reduced with any() to see if any keyword matched.
#     try:
#         mask = df['message'].str.contains('|'.join(keywords), case=False, na=False)
#         relevant_df = df[mask]
#
#         if relevant_df.empty:
#             return "No relevant context found."
#
#         # Format the context with date, user, and message
#         context_str = ""
#         for index, row in relevant_df.head(top_k).iterrows():
#             timestamp = row['date'].strftime('%Y-%m-%d %H:%M')
#             context_str += f"On {timestamp}, {row['user']} said: {row['message']}\n"
#
#         return context_str.strip()
#     except Exception:
#         return "Could not search the chat for context."
#
#
# def get_ai_response(question, chat_df):
#     """
#     Generates a response from Gemini using context from the chat history.
#     """
#     if not GEMINI_API_KEY:
#         return "GEMINI_API_KEY not configured. Please set it up in your .env file."
#
#     genai.configure(api_key=GEMINI_API_KEY)
#     model = genai.GenerativeModel('gemini-pro')
#
#     # 1. Retrieve context relevant to the question
#     context = get_relevant_context(question, chat_df)
#
#     # 2. Create a prompt for the model
#     prompt = f"""You are a helpful AI assistant analyzing a WhatsApp chat history.
#     Your task is to answer the user's question based *only* on the provided context snippets from the chat.
#     Be concise and directly answer the question. If the context does not contain the answer, state that you couldn't find the information in the chat history.
#
#     **Provided Context from Chat:**
#     ---
#     {context}
#     ---
#
#     **User's Question:**
#     {question}
#
#     **Answer:**
#     """
#
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"❌ An error occurred with the AI model: {e}"
#
#
# # --- Visualization Functions (No changes needed) ---
# def create_wordcloud_image(selected_user, df):
#     # This function remains the same
#     pass  # (Your existing wordcloud code goes here)
#
#
# # --- Main App Logic ---
# def main():
#     st.title("🔍 WhatsApp Chat Analyser")
#
#     # --- Sidebar for Inputs ---
#     with st.sidebar:
#         st.title("Controls")
#         if not GEMINI_API_KEY:
#             st.warning("Gemini API Key not found. Please create a `.env` file with your key to use the AI features.")
#
#         uploaded_file = st.sidebar.file_uploader("📤 Upload Chat Export", type=['txt'])
#
#     if uploaded_file is None:
#         st.info("👋 Welcome! Please upload your WhatsApp chat export file using the sidebar to begin.")
#         return
#
#     # Process data once and store in session state
#     if 'processed_data' not in st.session_state or st.session_state.get('uploaded_filename') != uploaded_file.name:
#         try:
#             bytes_data = uploaded_file.getvalue()
#             data = bytes_data.decode("utf-8")
#             st.session_state.processed_data = preprocessing_data.preprocess(data)
#             st.session_state.uploaded_filename = uploaded_file.name
#         except Exception as e:
#             st.error(f"Failed to process file: {e}")
#             return
#
#     df = st.session_state.processed_data
#
#     # --- Sidebar User Selection ---
#     user_list = df['user'].unique().tolist()
#     if 'group_notification' in user_list:
#         user_list.remove('group_notification')
#     user_list.sort()
#     user_list.insert(0, "Overall")
#     selected_user = st.sidebar.selectbox("🧑‍🤝‍🧑 Analyze User", user_list)
#
#     # Filter dataframe based on selection for analysis
#     if selected_user == "Overall":
#         analysis_df = df
#     else:
#         analysis_df = df[df['user'] == selected_user]
#
#     # --- Main Page with Tabs ---
#     tab1, tab2 = st.tabs(["📊 Dashboard Analytics", "🤖 AI Chat Assistant"])
#
#     with tab1:
#         st.header(f"Analytics for: {selected_user}")
#
#         # --- Top Level Metrics ---
#         st.subheader("Key Metrics")
#         num_messages, words, num_media_messages, num_links = functions.fetch_stats(selected_user, df)
#         col1, col2, col3, col4 = st.columns(4)
#         col1.metric("Total Messages", num_messages)
#         col2.metric("Total Words", words)
#         col3.metric("Media Shared", num_media_messages)
#         col4.metric("Links Shared", num_links)
#
#         # --- Timeline Visualizations ---
#         st.subheader("Conversation Timelines")
#         col1, col2 = st.columns(2)
#         with col1:
#             timeline = functions.monthly_timeline(selected_user, df)
#             fig_monthly = px.line(timeline, x='time', y='message', title='Monthly Trend')
#             st.plotly_chart(fig_monthly, use_container_width=True)
#         with col2:
#             daily_timeline = functions.daily_timeline(selected_user, df)
#             fig_daily = px.line(daily_timeline, x='only_date', y='message', title='Daily Trend')
#             st.plotly_chart(fig_daily, use_container_width=True)
#
#         # ... (Add other charts like WordCloud, Emoji Analysis etc. here)
#
#     with tab2:
#         st.header("🤖 AI Chat Assistant")
#         st.markdown("Ask questions about your chat history! For example: `When did we talk about the Goa trip?`")
#
#         # Initialize chat history in session state
#         if "messages" not in st.session_state:
#             st.session_state.messages = [
#                 {"role": "assistant", "content": "Hello! How can I help you analyze this chat?"}]
#
#         # Display chat messages
#         for msg in st.session_state.messages:
#             with st.chat_message(msg["role"]):
#                 st.write(msg["content"])
#
#         # Chat input
#         if prompt := st.chat_input("Ask a question..."):
#             if not GEMINI_API_KEY:
#                 st.info("Please add your Gemini API Key in the `.env` file to use this feature.")
#                 return
#
#             # Add user message to session state and display it
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.write(prompt)
#
#             # Get and display AI response
#             with st.chat_message("assistant"):
#                 with st.spinner("Thinking..."):
#                     # Use the full dataframe for context searching, regardless of user selection
#                     response = get_ai_response(prompt, df)
#                     st.write(response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})
#
#
# if __name__ == "__main__":
#     main()
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import hashlib
import os
import abc  # Abstract Base Class
from datetime import datetime

# Import search libraries
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, DATETIME
from whoosh.qparser import MultifieldParser, QueryParser
from whoosh.query import Term, TermRange,And

# Import your custom preprocessor
import preprocessing_data

# --- Constants & Setup ---
CACHE_DIR = "cache"
INDEX_DIR = "index"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)


# --- 1. Data Processing & Caching Pipeline ---
@st.cache_data(show_spinner="Processing and Caching Chat File...")
def process_and_cache_data(uploaded_file):
    """
    Hashes the uploaded file, processes it into a DataFrame, and caches the result
    in a performant Parquet file.
    """
    bytes_data = uploaded_file.getvalue()
    file_hash = hashlib.sha256(bytes_data).hexdigest()
    parquet_path = os.path.join(CACHE_DIR, f"{file_hash}.parquet")

    if os.path.exists(parquet_path):
        st.sidebar.info("Loading from cache.")
        df = pd.read_parquet(parquet_path)
    else:
        st.sidebar.info("New file detected. Processing...")
        # Clean up old cache files to save space
        for f in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, f))

        data_str = bytes_data.decode("utf-8")
        df = preprocessing_data.preprocess(data_str)
        df.to_parquet(parquet_path)

    # Convert date to timezone-naive for compatibility
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = df['date'].dt.tz_localize(None)

    return df, file_hash


# --- 2. Information Retrieval System (Search) ---
# @st.cache_resource(show_spinner="Building Search Index...")
# def create_or_load_index(_file_hash):
#     """
#     Creates or loads a Whoosh search index for the chat data.
#     The file_hash is used to ensure the index corresponds to the current data.
#     """
#     index_path = os.path.join(INDEX_DIR, _file_hash)
#
#     if os.path.exists(index_path):
#         return open_dir(index_path)
#     else:
#         # Clean up old index files
#         for f in os.listdir(INDEX_DIR):
#             if os.path.isdir(os.path.join(INDEX_DIR, f)):
#                 import shutil
#                 shutil.rmtree(os.path.join(INDEX_DIR, f))
#
#         os.makedirs(index_path)
#         schema = Schema(
#             id=ID(stored=True, unique=True),
#             user=TEXT(stored=True),
#             message=TEXT(stored=True),
#             date=DATETIME(stored=True)
#         )
#         ix = create_in(index_path, schema)
#
#         df = st.session_state.df
#         writer = ix.writer()
#         for i, row in df.iterrows():
#             writer.add_document(
#                 id=str(i),
#                 user=str(row['user']),
#                 message=str(row['message']),
#                 date=row['date']
#             )
#         writer.commit()
#         return ix
@st.cache_resource(show_spinner="Building Search Index...")
def create_or_load_index(_file_hash):
    """
    Creates or loads a Whoosh search index for the chat data.
    The file_hash is used to ensure the index corresponds to the current data.
    """
    index_path = os.path.join(INDEX_DIR, _file_hash)

    if os.path.exists(index_path):
        return open_dir(index_path)
    else:
        # Clean up old index files
        for f in os.listdir(INDEX_DIR):
            if os.path.isdir(os.path.join(INDEX_DIR, f)):
                import shutil
                shutil.rmtree(os.path.join(INDEX_DIR, f))

        os.makedirs(index_path)
        schema = Schema(
            id=ID(stored=True, unique=True),
            user=TEXT(stored=True),
            message=TEXT(stored=True),
            date=DATETIME(stored=True)
        )
        ix = create_in(index_path, schema)

        df = st.session_state.df
        writer = ix.writer()
        for i, row in df.iterrows():
            # The fix is applied on the 'date' line below
            writer.add_document(
                id=str(i),
                user=str(row['user']),
                message=str(row['message']),
                date=row['date'].to_pydatetime()  # CONVERTED TO STANDARD PYTHON DATETIME
            )
        writer.commit()
        return ix


# def search_index(ix, query_str, search_user, date_range):
#     """Performs a search on the index with filters."""
#     with ix.searcher() as searcher:
#         parser = MultifieldParser(["user", "message"], schema=ix.schema)
#
#         # Base query
#         query = parser.parse(query_str)
#
#         # Filter query
#         filter_queries = []
#         if search_user and search_user != "Overall":
#             filter_queries.append(Term("user", search_user))
#
#         # Whoosh date range format
#         start_date_query = f"[{date_range[0].strftime('%Y%m%d')} TO]"
#         end_date_query = f"[TO {date_range[1].strftime('%Y%m%d%H%M')}]"
#
#         date_q_parser = QueryParser("date", ix.schema)
#         start_q = date_q_parser.parse(start_date_query)
#         end_q = date_q_parser.parse(end_date_query)
#
#         filter_queries.append(start_q)
#         filter_queries.append(end_q)
#
#         results = searcher.search(query, filterby=filter_queries, limit=100)
#
#         hits = []
#         for hit in results:
#             hits.append({
#                 "Score": hit.score,
#                 "Date": hit['date'].strftime('%Y-%m-%d %H:%M'),
#                 "User": hit['user'],
#                 "Message": hit['message']
#             })
#         return pd.DataFrame(hits)
# def search_index(ix, query_str, search_user, date_range):
#     """Performs a search on the index with filters."""
#     with ix.searcher() as searcher:
#         parser = MultifieldParser(["user", "message"], schema=ix.schema)
#
#         # Base query
#         query = parser.parse(query_str)
#
#         # Build a list of queries to use as a filter
#         filter_queries = []
#         if search_user and search_user != "Overall":
#             # Filter by user
#             filter_queries.append(Term("user", search_user))
#
#         # Filter by date range. Whoosh DATETIME fields are precise.
#         # We need to create datetime objects for the start and end of the day.
#         start_dt = datetime.combine(date_range[0], datetime.min.time())
#         end_dt = datetime.combine(date_range[1], datetime.max.time())
#
#         # The TermRange query is inclusive by default
#         date_filter = TermRange("date", start_dt, end_dt)
#         filter_queries.append(date_filter)
#
#         # The fix is changing 'filterby' to 'filter' in the line below
#         results = searcher.search(query, filter=filter_queries, limit=100)
#
#         hits = []
#         for hit in results:
#             hits.append({
#                 "Score": f"{hit.score:.2f}",
#                 "Date": hit['date'].strftime('%Y-%m-%d %H:%M'),
#                 "User": hit['user'],
#                 "Message": hit['message']
#             })
#         return pd.DataFrame(hits)
def search_index(ix, query_str, search_user, date_range):
    """Performs a search on the index with filters."""
    with ix.searcher() as searcher:
        parser = MultifieldParser(["user", "message"], schema=ix.schema)

        # Base query for the user's search string
        query = parser.parse(query_str)

        # Build a list of filter conditions that MUST be met
        filter_conditions = []
        if search_user and search_user != "Overall":
            # Condition 1: Filter by user
            filter_conditions.append(Term("user", search_user))

        # Condition 2: Filter by date range
        start_dt = datetime.combine(date_range[0], datetime.min.time())
        end_dt = datetime.combine(date_range[1], datetime.max.time())
        date_filter = TermRange("date", start_dt, end_dt)
        filter_conditions.append(date_filter)

        # Combine all filter conditions using the 'And' operator
        # This is the main fix
        combined_filter = And(filter_conditions) if filter_conditions else None

        results = searcher.search(query, filter=combined_filter, limit=100)

        hits = []
        for hit in results:
            hits.append({
                "Score": f"{hit.score:.2f}",
                "Date": hit['date'].strftime('%Y-%m-%d %H:%M'),
                "User": hit['user'],
                "Message": hit['message']
            })
        return pd.DataFrame(hits)

# --- 3. Modular Analysis Framework ---
class AbstractAnalyzer(abc.ABC):
    """Abstract Base Class for all analysis modules."""

    @abc.abstractmethod
    def render(self, df: pd.DataFrame):
        """Renders the analysis output in Streamlit."""
        pass


class KeyMetricsAnalyzer(AbstractAnalyzer):
    def render(self, df: pd.DataFrame):
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        num_messages = len(df)
        words = sum(df['message'].apply(lambda x: len(str(x).split())))
        media = df[df['message'] == '<Media omitted>\n'].shape[0]
        links = df['message'].str.contains(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+').sum()

        col1.metric("Total Messages", f"{num_messages:,}")
        col2.metric("Total Words", f"{words:,}")
        col3.metric("Media Shared", f"{media:,}")
        col4.metric("Links Shared", f"{links:,}")


class TimelineAnalyzer(AbstractAnalyzer):
    def render(self, df: pd.DataFrame):
        st.subheader("Message Timelines")
        col1, col2 = st.columns(2)

        with col1:
            df['month_year'] = df['date'].dt.to_period('M').astype(str)
            monthly_timeline = df.groupby('month_year').size().reset_index(name='message_count')
            fig = px.line(monthly_timeline, x='month_year', y='message_count', title="Monthly Messages", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            df['date_only'] = df['date'].dt.date
            daily_timeline = df.groupby('date_only').size().reset_index(name='message_count')
            fig = px.line(daily_timeline, x='date_only', y='message_count', title="Daily Messages")
            st.plotly_chart(fig, use_container_width=True)


class WordCloudAnalyzer(AbstractAnalyzer):
    def render(self, df: pd.DataFrame):
        st.subheader("Most Common Words")
        text = ' '.join(df['message'])

        if not text.strip():
            st.warning("Not enough text to generate a word cloud for the selected filters.")
            return

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)


# --- Main Application ---
def main():
    st.set_page_config(layout="wide", page_title="Professional Chat Analyzer")
    st.title(" WhatsApp Chat Analyzer ")

    # --- Sidebar for Upload and Global Filters ---
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload your WhatsApp chat export (.txt)", type="txt")

        if uploaded_file:
            # Module 1: Caching Pipeline
            df, file_hash = process_and_cache_data(uploaded_file)
            st.session_state.df = df
            st.session_state.file_hash = file_hash

            # Module 3: Indexing
            ix = create_or_load_index(file_hash)
            st.session_state.ix = ix

            st.success(f"Loaded {len(df)} messages.")
            st.markdown("---")
            st.header("Global Filters")

            # Global User Filter
            user_list = ["Overall"] + sorted(df['user'].unique().tolist())
            selected_user = st.selectbox("Filter by User", user_list)

            # Global Date Filter
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()
            date_range = st.date_input("Filter by Date Range", (min_date, max_date), min_value=min_date,
                                       max_value=max_date)
            if len(date_range) != 2:
                st.stop()  # Wait for user to select a valid date range

    # --- Main Content Area ---
    if 'df' not in st.session_state:
        st.info("Please upload a chat file to begin analysis.")
        return

    # Apply global filters to create a view for the dashboard
    df_main = st.session_state.df
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]).replace(hour=23, minute=59)

    filtered_df = df_main[
        (df_main['date'] >= start_date) &
        (df_main['date'] <= end_date)
        ]
    if selected_user != "Overall":
        filtered_df = filtered_df[filtered_df['user'] == selected_user]

    # --- Tabs for different functionalities ---
    tab1, tab2 = st.tabs(["📊 Analytical Dashboard", "🔍 Data Explorer & Search"])

    with tab1:
        st.header(f"Dashboard for: `{selected_user}`")
        st.markdown(f"**Date Range:** `{date_range[0]}` to `{date_range[1]}`")

        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
        else:
            # Module 2: Render Modular Analyzers
            KeyMetricsAnalyzer().render(filtered_df)
            st.markdown("---")
            TimelineAnalyzer().render(filtered_df)
            st.markdown("---")
            WordCloudAnalyzer().render(filtered_df)

    with tab2:
        st.header("Data Explorer & Search")

        # Search functionality
        with st.expander("Advanced Search", expanded=True):
            search_query = st.text_input("Search for keywords or phrases in the chat")
            if st.button("Search"):
                if search_query:
                    results_df = search_index(st.session_state.ix, search_query, selected_user, date_range)
                    st.write(f"Found `{len(results_df)}` results.")
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.warning("Please enter a search query.")

        # Data table view
        with st.expander("Explore Filtered Data"):
            st.dataframe(filtered_df, use_container_width=True)


if __name__ == "__main__":
    main()
