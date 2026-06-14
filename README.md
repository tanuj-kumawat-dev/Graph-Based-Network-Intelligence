# Delhivery Logistics Network Intelligence 🚚

Optimizing Delivery ETAs, identifying structural chokepoint hubs, and recommending FTL vs. Carting routes using Graph Algorithms and Machine Learning.

---

## 📌 Business Context & Challenge
Delhivery is India's largest fully-integrated logistics provider. The core operations rely on standard routing engines (OSRM) to estimate transit times. However, OSRM assumes clean traffic and shortest paths, leading to systematic underestimation of actual delivery times. 

This project models Delhivery's logistics network as a **directed weighted graph** to predict checkpoint-level ETAs with high accuracy, audit facility delay risks, and optimize route-type decisions.

---

## 📂 Project Structure

```text
Graph-Based Network Intelligence/
│
├── 📁 1. Code & Pipeline/
│   ├── build_graph.py            # Preprocesses scans & constructs NetworkX directed graph
│   ├── audit_network.py          # Performs hub centrality audit & saves visuals
│   ├── train_eta_model.py        # Generates Node2Vec embeddings & trains LightGBM models
│   └── route_type_framework.py    # Analyzes FTL vs. Carting and trains RF recommender
│
├── 📁 2. Consulting Reports/
│   ├── strategy_memo.md          # 1-2 page strategy memo (Markdown)
│   ├── strategy_memo.html        # Beautifully styled strategy memo (HTML)
│   ├── walkthrough.md            # Technical walkthrough & benchmark details (Markdown)
│   └── walkthrough.html          # Beautifully styled walkthrough (HTML)
│
├── 📁 3. Precomputed Artifacts & Visuals/
│   ├── network_bottlenecks.png   # Logistics network centrality visualization
│   ├── node_audit_results.csv    # Centrality & delay metrics per facility node
│   ├── corridor_audit_results.csv# Delays & SLA breach counts per corridor edge
│   └── ftl_vs_carting_recommendations.csv
│
├── 📁 4. Dashboard/
│   └── app.py                    # Streamlit Dashboard UI
│
├── 📁 5. Serialized Models & Graphs/
│   ├── logistics_graph.pkl       # NetworkX logistics graph object
│   ├── base_model.pkl            # Trained baseline OSRM regressor
│   ├── graph_model.pkl           # Trained graph-enhanced ETA regressor
│   └── route_type_classifier.pkl # Trained FTL vs. Carting recommender
│
└── 📄 delivery_data.csv          # Raw logistics tracking data
```

---

## ⚙️ Quick Start

### 1. Installation
Install the necessary python dependencies:
```bash
pip install pandas numpy networkx streamlit gensim scikit-learn lightgbm matplotlib seaborn markdown
```

### 2. Run the Data & ML Pipeline (In Order)
From the project root directory, run the pipeline commands:
```bash
# Compile logistics graph
python "1. Code & Pipeline/build_graph.py"

# Perform centrality audit & generate visuals
python "1. Code & Pipeline/audit_network.py"

# Train row-level models & print benchmarks
python "1. Code & Pipeline/train_eta_model.py"

# Train FTL vs. Carting decision recommender
python "1. Code & Pipeline/route_type_framework.py"
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run "4. Dashboard/app.py"
```
Open `http://localhost:8501` in your browser.

---

## 📊 Performance Benchmarks (Unseen Test Set)

We predict the actual remaining transit time at each checkpoint scan. By adding graph-based features (betweenness centrality, corridor delays, Node2Vec node embeddings) and training with log-target transformation, we achieve:

*   **Mean Absolute Error (MAE)**: Reduced from **46.23 minutes** (Baseline) to **35.81 minutes** (Graph Model) — a **22.54% reduction in error**.
*   **SLA Prediction Accuracy (within 15% of actual)**: Increased from **68.26%** to **75.69%** (a **+7.43% absolute increase**).

---

## 📍 Key Insights (Top 5 Bottlenecks)

By calculating structural betweenness centrality and historical delay minutes, we ranked the top bottleneck chokepoints:
1.  **Gurgaon Bilaspur HB (`IND000000ACB`)** — *Betweenness: 22.02% \| SLA Breach Rate: 95.8%*
2.  **Bangalore Nelamangala H (`IND562132AAA`)** — *Betweenness: 12.50% \| SLA Breach Rate: 94.9%*
3.  **Kolkata Dankuni HB (`IND712311AAA`)** — *Betweenness: 9.10% \| SLA Breach Rate: 100.0%*
4.  **Bhiwandi Mankoli HB (`IND421302AAG`)** — *Betweenness: 5.61% \| SLA Breach Rate: 99.8%*
5.  **Hyderabad Shamshabad H (`IND501359AAE`)** — *Betweenness: 9.04% \| SLA Breach Rate: 99.5%*

---

## 🛠️ FTL vs. Carting Recommendation
FTL is recommended over Carting for high-priority or long-distance corridors where transit times are highly volatile. For example, shifting route types on the **Agra Idgah to Delhi Gateway** corridor saves **49.5 minutes** on average.
