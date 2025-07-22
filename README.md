#ChatMiner: A Production-Grade WhatsApp Chat Analysis Platform

An end-to-end data application engineered to perform high-speed, interactive analysis on large, unstructured WhatsApp chat histories. Deployed as a live web app on Streamlit Cloud.

<img width="2877" height="1027" alt="Screenshot 2025-07-22 200426" src="https://github.com/user-attachments/assets/35881ad6-4749-4578-bceb-5efa4e7104c8" />


<img width="2253" height="996" alt="Screenshot 2025-07-22 200450" src="https://github.com/user-attachments/assets/ab6f814d-eb23-4368-af1c-3dd8df65e949" />


<img width="2823" height="1529" alt="Screenshot 2025-07-22 200521" src="https://github.com/user-attachments/assets/d7696bb6-6bd8-4044-a52e-988d3ca0d7a5" />



<br><br>
🌟 The Problem

WhatsApp chats are a rich source of personal data, but notoriously difficult to analyze. Exported chat files are unstructured and inefficient to process. Searching large histories is painful with basic tools.

ChatMiner solves this with a scalable, fast, and intelligent system for interactive, deep analysis.

🚀 Key Engineering Features

High-Performance Caching Pipeline

Problem: Re-processing files every time is expensive and slow.

Solution: Files are hashed and stored as Parquet. If the same file is uploaded again, it's loaded from cache instantly.

Impact: Over 95% reduction in load time for returning users.

Extensible OOP Architecture

Problem: Monolithic scripts are hard to maintain.

Solution: Built a modular, class-based system with plug-and-play analyzers.

Impact: Easy addition of features without breaking existing code.

Advanced Information Retrieval System

Problem: Ctrl+F isn't powerful enough.

Solution: A full-text search engine built using Whoosh, supporting keyword, user, and date filters.

Impact: Turns raw chat logs into a queryable, filterable database.

🏫 System & Data Flow Architecture

The application's architecture is divided into two main processes:

✨ Data Loading and Caching Pipeline

File Upload: A user uploads their chat.txt file.

Hashing: A unique hash is created based on the file content.

Cache Check:

Cache Hit: If hash exists, the Parquet file is loaded instantly.

Cache Miss:

ETL script processes the file.

Structured data is saved as Parquet using the file hash.

Display: Data is visualized through the interactive dashboard.

🔍 Search Engine Flow

Indexing: After ETL, data is indexed using Whoosh.

User Query: A user enters a search term.

Query Engine: Applies filters and retrieves messages from the index.

Display: Relevant results shown in the dashboard.


🛠️ Tech Stack

Category

Technologies
Frontend & Deployment
Streamlit, Streamlit Cloud, Plotly
Backend & Architecture
Python, Object-Oriented Programming (OOP)
Data Pipeline & Storage
Pandas, NumPy, Apache Parquet, ETL
Search & Retrieval
Whoosh

📆 Local Installation & Setup

Clone the Repository

git clone https://github.com/your-username/chatminer.git
cd chatminer

Create and Activate a Virtual Environment

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

Run the Application

streamlit run app.py

🔮 Future Enhancements

Async Data Processing: Use Celery or Redis to process extremely large files in the background.

NLP Analysis: Add Hugging Face models for sentiment analysis, NER, topic modeling.

Database Integration: Switch from file-based cache to PostgreSQL or MongoDB for better multi-user support.

Containerization: Dockerize for consistent deployment and scaling.

