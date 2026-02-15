# 🧠 Fake News Detection using Machine Learning & Sentiment Intelligence

<div align="center">

🚀 Comparative ML Study | 📊 Result-Based Evaluation | 🌐 Web Scraping | 💬 Sentiment Analysis  

</div>

---

## 🌍 Project Vision

In the age of digital misinformation, detecting fake news is no longer optional — it is essential.

This project presents a **comprehensive, result-driven comparative analysis** of six Machine Learning models for Fake News Detection, enhanced with real-time web scraping and sentiment intelligence.

It is designed not just as a classification system, but as a **research-ready analytical framework** for evaluating model performance, computational trade-offs, and emotional polarity patterns in misinformation.

---

# 🏗 System Architecture

```
Raw Dataset + Scraped News
            │
            ▼
   Data Preprocessing
            │
            ▼
      TF-IDF Vectorization
            │
            ▼
  Model Segmentation Strategy
   ├── Lightweight Models
   └── Heavyweight Models
            │
            ▼
   Evaluation & Visualization
            │
            ▼
   Sentiment Intelligence Layer
            │
            ▼
      Exportable Results
```

---

# 🧪 Models Implemented

## 🔹 Lightweight Models (Full Dataset)

Optimized for speed and scalability.

- ✅ Multinomial Naive Bayes  
- ✅ Logistic Regression (GridSearchCV tuned)  
- ✅ K-Nearest Neighbors  

These models are trained on the full dataset to observe generalization performance.

---

## 🔹 Heavyweight Models (Sampled Dataset)

Optimized for deeper decision boundaries and ensemble intelligence.

- ✅ Support Vector Machine (Linear Kernel, optimized runtime)  
- ✅ Random Forest (Controlled depth & estimators)  
- ✅ XGBoost (Parallel optimized)  

Heavy models are trained on a reduced dataset fraction to maintain computational efficiency while preserving analytical depth.

---

# 📊 Evaluation Framework

Each model is evaluated using:

- 🔹 Accuracy  
- 🔹 Precision  
- 🔹 Recall  
- 🔹 F1-Score  
- 🔹 Confusion Matrix  
- 🔹 ROC Curve (AUC)  
- 🔹 Precision-Recall Curve  

Additionally, a **Sentiment Intelligence Layer** provides:

- Average VADER Compound Sentiment Score  
- Average sentiment for predictions labeled Fake  
- Average sentiment for predictions labeled Real  

Results are exported as:

```
light_model_results.csv
heavy_model_results.csv
```

This enables direct inclusion into research papers or comparative studies.

---

# 💬 Sentiment Intelligence Layer

We integrate **VADER (Valence Aware Dictionary for Sentiment Reasoning)** to analyze the emotional polarity of news content.

Each news article receives:

- Compound sentiment score
- Polarity classification:
  - Positive
  - Neutral
  - Negative

This allows us to analyze:

> Do fake news articles exhibit stronger emotional polarity than real news?

This hybrid architecture makes the system not just predictive — but analytical.

---

# 🌐 Real-Time Web Scraping Integration

To enrich the dataset and maintain realism:

The system scrapes headlines from:

- BBC News  
- CNN World  
- Reuters  

Scraped articles are automatically labeled as Real (0) and appended before training.

This ensures dynamic dataset augmentation.

---

# 📂 Dataset Description

Main dataset file:

```
updated_news_dataset.csv
```

Columns:

| Column | Description |
|--------|-------------|
| text   | News content |
| label  | 1 = Fake, 0 = Real |

The dataset includes:

- Fake news articles  
- Verified real news articles  
- Live scraped headlines  

---

# ⚙️ Technical Stack

- Python  
- Scikit-learn  
- XGBoost  
- NLTK (VADER Sentiment Analyzer)  
- Pandas  
- Matplotlib  
- Seaborn  
- BeautifulSoup (Web Scraping)  
- Google Colab  
- Joblib (Model & Vectorizer Persistence)

---

# 🚀 How to Run

## 1️⃣ Mount Drive (Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2️⃣ Install Dependencies (if needed)

```bash
pip install xgboost nltk seaborn beautifulsoup4
```

## 3️⃣ Run Lightweight Notebook

- Uses full dataset  
- Trains NB, LR, KNN  
- Generates evaluation plots  
- Saves `light_model_results.csv`

## 4️⃣ Run Heavyweight Notebook

- Uses sampled dataset  
- Trains SVM, Random Forest, XGBoost  
- Generates evaluation plots  
- Saves `heavy_model_results.csv`

---

# 🧠 Key Engineering Decisions

✔ Segmented model architecture (Light vs Heavy)  
✔ Runtime optimization using dataset sampling  
✔ TF-IDF caching for faster re-runs  
✔ Decision function for SVM to reduce training time  
✔ Parallel training for ensemble models  
✔ Structured evaluation for research reproducibility  

---

# 📈 Research & Analytical Value

This project enables:

- Empirical comparison across 6 ML algorithms  
- Computational complexity vs performance analysis  
- Emotional polarity study in misinformation  
- ROC vs PR curve tradeoff evaluation  
- Exportable research-grade results  

It is suitable for:

- Academic Review Papers  
- Machine Learning Portfolios  
- Applied NLP Research  
- Interview Demonstrations  

---

# 📁 Project Structure

```
FakeNewsProject/
│
├── updated_news_dataset.csv
├── tfidf_vectorizer.pkl
│
├── lightweight_models.ipynb
├── heavyweight_models.ipynb
│
├── light_model_results.csv
├── heavy_model_results.csv
│
└── README.md
```

---

# 🔮 Future Enhancements

- Deep Learning models (LSTM / BERT)
- Transformer-based fake news classification
- Real-time API deployment
- Web dashboard visualization
- Explainable AI integration (SHAP / LIME)
- Automated hyperparameter optimization

---

# 👨‍💻 Author

Machine Learning Project  
Fake News Detection using Sentiment Analysis  
Comparative ML Evaluation Framework  

---

<div align="center">

⭐ If you found this project insightful, consider starring the repository!

</div>
