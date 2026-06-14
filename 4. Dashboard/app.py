import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Delhivery Network Intelligence",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Delhivery Graph-Based Network Intelligence Dashboard")
st.markdown("""
*Surfacing bottleneck hubs and recommending optimal route types using Graph Intelligence.*
""")

# Relative paths from the project root
node_path = "3. Precomputed Artifacts & Visuals/node_audit_results.csv"
corridor_path = "3. Precomputed Artifacts & Visuals/corridor_audit_results.csv"
recommend_path = "3. Precomputed Artifacts & Visuals/ftl_vs_carting_recommendations.csv"
img_path = "3. Precomputed Artifacts & Visuals/network_bottlenecks.png"
classifier_model_path = "5. Serialized Models & Graphs/route_type_classifier.pkl"

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.markdown("""
This dashboard displays graph analytics and ML recommendations for Delhivery's Logistics Network.
""")

# Load data
@st.cache_data
def load_data():
    nodes = pd.read_csv(node_path) if os.path.exists(node_path) else pd.DataFrame()
    corridors = pd.read_csv(corridor_path) if os.path.exists(corridor_path) else pd.DataFrame()
    recs = pd.read_csv(recommend_path) if os.path.exists(recommend_path) else pd.DataFrame()
    return nodes, corridors, recs

nodes_df, corridors_df, recs_df = load_data()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📈 Bottleneck Hub Audit", 
    "🛣️ Chronically Delayed Corridors",
    "⚡ Route Type Recommender"
])

with tab1:
    st.header("Top Bottleneck Hubs (Ranked by Centrality & Delays)")
    
    if not nodes_df.empty:
        # Display the network visualization image
        if os.path.exists(img_path):
            st.image(img_path, caption="Network Hubs and Bottlenecks. Node size represents betweenness centrality, color intensity represents total delay.", use_container_width=True)
            
        st.subheader("Interactive Hub Rankings")
        sort_by = st.selectbox(
            "Sort Hubs By:",
            ["composite_rank", "betweenness_centrality", "total_delay_minutes", "total_sla_breaches"]
        )
        
        display_cols = ['node_id', 'name', 'betweenness_centrality', 'total_delay_minutes', 'total_sla_breaches', 'total_trips_handled']
        st.dataframe(
            nodes_df.sort_values(by=sort_by, ascending=(sort_by == "composite_rank"))[display_cols].head(20),
            use_container_width=True
        )
    else:
        st.write("No hub audit data found.")

with tab2:
    st.header("Chronically Delayed Corridors")
    st.markdown("Corridors where the median actual delivery time exceeds OSRM predictions by **> 20%**.")
    
    if not corridors_df.empty:
        st.dataframe(
            corridors_df[corridors_df['median_delay_ratio'] > 1.2].sort_values(by='sla_breach_count', ascending=False),
            use_container_width=True
        )
    else:
        st.write("No corridor audit data found.")

with tab3:
    st.header("FTL vs Carting Decision Framework")
    st.markdown("Determine whether to switch route-type from Carting to FTL to minimize delays based on corridor profiles.")
    
    if not recs_df.empty:
        st.subheader("Corridors with Both Route Types (Comparison)")
        st.dataframe(
            recs_df.sort_values(by='time_difference_mins', ascending=False),
            use_container_width=True
        )
        
        st.subheader("Predict Optimal Route Type")
        input_dist = st.number_input("Enter Route Distance (km):", min_value=1.0, max_value=5000.0, value=250.0)
        
        # Load route type classifier model
        if os.path.exists(classifier_model_path):
            with open(classifier_model_path, 'rb') as f:
                clf = pickle.load(f)
            
            # Predict
            # Features: distance, src_betweenness, src_in_degree, src_out_degree, src_clustering
            features = np.array([[input_dist, 0.1, 10, 10, 0.2]]) # default hub features
            rec = clf.predict(features)[0]
            
            if rec:
                st.success("🎯 **Recommendation: FTL** (Full Truck Load is predicted to minimize delay for this route configuration).")
            else:
                st.info("🎯 **Recommendation: Carting** (Carting is predicted to be sufficient and cost-effective for this route configuration).")
    else:
        st.write("No route recommendation data found.")
