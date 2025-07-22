# 🧠 ChatMiner: A Production-Grade WhatsApp Chat Analysis Platform

An end-to-end data application engineered to perform high-speed, interactive analysis on large, unstructured WhatsApp chat histories.  
Deployed as a live web app on **Streamlit Cloud**.

---
<img width="2877" height="1027" alt="Screenshot 2025-07-22 200426" src="https://github.com/user-attachments/assets/35881ad6-4749-4578-bceb-5efa4e7104c8" />


<img width="2253" height="996" alt="Screenshot 2025-07-22 200450" src="https://github.com/user-attachments/assets/ab6f814d-eb23-4368-af1c-3dd8df65e949" />


<img width="2823" height="1529" alt="Screenshot 2025-07-22 200521" src="https://github.com/user-attachments/assets/d7696bb6-6bd8-4044-a52e-988d3ca0d7a5" />



<br><br>


---

## 🎯 The Problem

WhatsApp chats are a rich source of personal data—but they are unstructured, messy, and notoriously hard to search or analyze. Loading a chat with hundreds of thousands of messages into a script is painfully slow. Even worse, basic tools like `Ctrl+F` break down when you need specific, filtered insights.

**ChatMiner** solves this by providing a robust, production-grade platform for powerful, intelligent, and blazing-fast chat analysis.

---

## 🚀 Key Engineering Features

This isn’t just a data viz tool—it’s an engineered system built using SDE principles: modularity, scalability, fault-tolerance, and performance.

### ⚡ High-Performance Caching Pipeline

- ✅ **Problem**: Re-processing large chat files on every use was inefficient.  
- 🛠️ **Solution**: Built a custom ETL pipeline that:
  - Hashes each uploaded file (SHA-256).
  - Caches the cleaned data as a Parquet file.
- 🚀 **Impact**: Reduced load times for cached files by **>95%**.

### 🧩 Modular, Extensible OOP Architecture

- ✅ **Problem**: Spaghetti code made future feature development hard.  
- 🛠️ **Solution**: Developed a modular analysis framework using Python's OOP principles.  
- 🔌 **Impact**: Plug-and-play design allows new features to be added with zero changes to core logic.

### 🔍 Advanced Information Retrieval Engine

- ✅ **Problem**: Basic search lacks precision.  
- 🛠️ **Solution**: Integrated a Whoosh-based full-text search engine:
  - Builds a persistent search index.
  - Supports keyword, user, and date filters.
- 🔎 **Impact**: Converts unstructured text into a queryable high-performance data system.

---

## 🏛️ System Architecture

The application is divided into two main flows:

### 🔄 Data Loading and Caching Pipeline

This pipeline ensures large chat files are loaded efficiently.

1. **File Upload**: User uploads a `chat.txt` file.
2. **Hashing**: SHA-256 hash is generated for caching logic.
3. **Cache Check**:
   - ✅ **Hit**: Load Parquet file directly (very fast).
   - ❌ **Miss**: 
     - Run custom ETL script to clean and structure data.
     - Save output as a Parquet file using the hash name.
4. **Display**: Structured data is loaded into the dashboard.

### 🧠 Search Engine Flow

Enables advanced, multi-filter full-text search.

1. **Indexing**: Cleaned data is indexed with Whoosh on first load.
2. **User Query**: Query is entered via dashboard UI.
3. **Query Engine**:
   - Uses Whoosh to retrieve filtered results (by user, date, content).
4. **Results**: Returned to and visualized in the dashboard.



---

## 🛠️ Tech Stack

| Category                | Technologies                                   |
|-------------------------|------------------------------------------------|
| Frontend & Deployment   | Streamlit, Streamlit Cloud, Plotly             |
| Backend Architecture    | Python, OOP, Modular Design                    |
| Data Pipeline & Storage | Pandas, NumPy, Apache Parquet                  |
| Search & Retrieval      | Whoosh (full-text indexing & filtering)        |

---

## ⚙️ Local Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/chatminer.git
cd chatminer

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py

















