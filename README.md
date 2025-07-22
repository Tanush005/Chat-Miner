ChatMiner: A Production-Grade WhatsApp Chat Analysis Platform

An end-to-end data application engineered to perform high-speed, interactive analysis on large, unstructured WhatsApp chat histories. Deployed as a live web app on Streamlit Cloud.

<img width="2877" height="1027" alt="Screenshot 2025-07-22 200426" src="https://github.com/user-attachments/assets/35881ad6-4749-4578-bceb-5efa4e7104c8" />


<img width="2253" height="996" alt="Screenshot 2025-07-22 200450" src="https://github.com/user-attachments/assets/ab6f814d-eb23-4368-af1c-3dd8df65e949" />


<img width="2823" height="1529" alt="Screenshot 2025-07-22 200521" src="https://github.com/user-attachments/assets/d7696bb6-6bd8-4044-a52e-988d3ca0d7a5" />



<br><br>



🎯 The Problem

WhatsApp chats are a rich source of personal data, but they are notoriously difficult to analyze. Exported chat files are unstructured, and loading a large history with hundreds of thousands of messages into a standard script is slow and inefficient. Furthermore, finding specific information within these massive text files is nearly impossible with basic search tools.

ChatMiner was engineered to solve these problems by providing a production-grade platform that is fast, scalable, and powerful.

🚀 Key Engineering Features

This project was built from the ground up with a focus on software engineering principles, not just data analysis. The architecture prioritizes performance, modularity, and robust information retrieval.

High-Performance Caching Pipeline

Problem: Re-processing a large chat file on every visit is computationally expensive and leads to a poor user experience.

Solution: Architected a custom ETL pipeline that processes and hashes each file, caching structured output as a Parquet file.

Impact: Achieved >95% reduction in subsequent load times, enabling instant analysis for repeat users.

Extensible OOP Architecture

Problem: Monolithic scripts are difficult to maintain and extend.

Solution: Built a modular system using Object-Oriented Programming (OOP). Each analyzer is self-contained and inherits from a common base class.

Impact: New features can be added independently with zero code conflicts, ensuring maintainability.

Advanced Information Retrieval System

Problem: Basic Ctrl+F search is insufficient for complex queries.

Solution: Integrated the Whoosh search library to create a persistent, full-text index with support for multi-filter queries.

Impact: Enabled rich, user-defined searches with filters by user, date range, and content, transforming raw chat logs into a searchable knowledge base.

🏩 System Architecture

The application's architecture is divided into two main processes: the Data Loading and Caching Pipeline and the Search Engine Flow.

Data Loading and Caching Pipeline

This process ensures that chat files are loaded and processed efficiently.

File Upload: A user uploads their chat.txt file.

Hashing: The system generates a unique hash of the uploaded file's content.

Cache Check: It checks if a pre-processed Parquet file corresponding to this hash already exists in the cache.

Cache Hit (File exists): Load the data directly from the cache.

Cache Miss (File does not exist):

Run the ETL (Extract, Transform, Load) script.

Process, clean, and structure the raw text into a DataFrame.

Save the structured data as a Parquet file using the hash.

Display: The data is then passed to the interactive dashboard for user exploration.

Search Engine Flow

This process powers the advanced information retrieval capabilities.

Indexing: After processing a new file, the cleaned data is used to create or update a persistent search index using Whoosh.

User Query: The user enters a query in the dashboard.

Query Engine: The query is executed using the index, applying filters (user, date, etc.).

Display Results: Matching messages are sent back to the dashboard for display.

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

⚙️ Local Installation & Setup

Clone the Repository

git clone [Your GitHub Repository URL Here]
cd [your-repo-name]

Create and Activate a Virtual Environment

# For macOS/Linux
python -m venv venv
source venv/bin/activate

# For Windows
venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

Run the Application

streamlit run app.py

🔮 Future Enhancements

Asynchronous Processing: Move ETL and indexing to background workers using Celery + Redis.

Advanced NLP Analytics: Add sentiment analysis, NER, and topic modeling via Hugging Face Transformers.

Database Integration: Use PostgreSQL or a NoSQL store to manage user data and chat indexes for scalability.

Containerization: Dockerize the application for isolated deployment and CI/CD integration.
