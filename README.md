#ChatMiner

An end-to-end data application engineered to perform high-speed, interactive analysis on large, unstructured WhatsApp chat histories. Deployed as a live web app on Streamlit Cloud.

<img width="2877" height="1027" alt="Screenshot 2025-07-22 200426" src="https://github.com/user-attachments/assets/35881ad6-4749-4578-bceb-5efa4e7104c8" />


<img width="2253" height="996" alt="Screenshot 2025-07-22 200450" src="https://github.com/user-attachments/assets/ab6f814d-eb23-4368-af1c-3dd8df65e949" />


<img width="2823" height="1529" alt="Screenshot 2025-07-22 200521" src="https://github.com/user-attachments/assets/d7696bb6-6bd8-4044-a52e-988d3ca0d7a5" />



<br><br>

🎯 The Problem
WhatsApp chats are a rich source of personal data—but they are unstructured, messy, and notoriously hard to search or analyze. Loading a chat with hundreds of thousands of messages into a script is painfully slow. Even worse, basic tools like Ctrl+F break down when you need specific, filtered insights.

ChatMiner solves this by providing a robust, production-grade platform for powerful, intelligent, and blazing-fast chat analysis.

🚀 Key Engineering Features
This isn’t just a data viz tool—it’s an engineered system built using SDE principles: modularity, scalability, fault-tolerance, and performance.

⚡ High-Performance Caching Pipeline
✅ Problem: Re-processing large chat files on every use was inefficient.

🛠️ Solution: Built a custom ETL pipeline that:

Hashes each uploaded file (SHA-256).

Caches the cleaned data as a Parquet file.

🚀 Impact: Reduced load times for cached files by >95%.

🧩 Modular, Extensible OOP Architecture
✅ Problem: Spaghetti code made future feature development hard.

🛠️ Solution: Developed a modular analysis framework using Python's OOP principles.

🔌 Impact: Plug-and-play design allows new features to be added with zero changes to core logic.

🔍 Advanced Information Retrieval Engine
✅ Problem: Ctrl+F and basic filtering fall short for big chats.

🛠️ Solution: Implemented a Whoosh-powered search engine:

Builds persistent indexes from the chat data.

Supports multi-filter queries (user, date range, text).

🔎 Impact: Converts unstructured text into a queryable system.

🏛️ System Architecture
The application's architecture is divided into two main processes:

🔄 Data Loading and Caching Pipeline
This pipeline ensures that even the largest chat files load efficiently:

File Upload: User uploads a WhatsApp chat.txt.

SHA256 Hashing: System computes a hash of file content.

Cache Check:

✅ Cache Hit: Loads .parquet file instantly.

❌ Cache Miss: Proceeds to:

Run a custom ETL pipeline (cleans, transforms).

Save result as .parquet (using hash as ID).

Dashboard Load: Cleaned data is sent to the dashboard.

🧠 Search Engine Flow
Powering deep and accurate information retrieval:

Indexing: During first-time ETL, the cleaned data is used to build a Whoosh index.

User Query: The user enters a query via dashboard.

Query Engine:

Searches the Whoosh index.

Applies filters (user, date, etc.).

Display: Filtered results are returned to the dashboard.


🛠️ Tech Stack
Category	Technologies
Frontend & Deployment	Streamlit, Streamlit Cloud, Plotly
Backend & Architecture	Python, OOP, Modular Design
Data Pipeline & Storage	Pandas, NumPy, Apache Parquet
Search & Retrieval	Whoosh (full-text search + indexing)

⚙️ Local Setup & Installation
bash
Copy
Edit
# 1. Clone the repo
git clone https://github.com/your-username/chatminer.git
cd chatminer

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
🔮 Future Enhancements
Asynchronous Processing: Offload heavy ETL and indexing using Celery + Redis.

NLP Analytics: Integrate Hugging Face models for:

Sentiment Analysis

Topic Modeling

Named Entity Recognition

Database Backend: Move from file-based caching to a PostgreSQL or MongoDB system for multi-user support.

Containerization: Use Docker for consistent deployments across environments.

Authentication: Add user login & saved sessions.




