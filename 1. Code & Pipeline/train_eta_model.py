import pandas as pd
import numpy as np
import pickle
import os
import networkx as nx
from gensim.models import Word2Vec
import random
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_random_walks(G, num_walks=10, walk_length=15):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            curr = node
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(curr))
                if len(neighbors) == 0:
                    break
                weights = [G[curr][nbr].get('weight', 1.0) for nbr in neighbors]
                sum_w = sum(weights)
                probs = [w / sum_w for w in weights] if sum_w > 0 else None
                curr = np.random.choice(neighbors, p=probs)
                walk.append(curr)
            walks.append(walk)
    return walks

def learn_node_embeddings(G, embedding_dim=16):
    print("Generating random walks for Node2Vec...")
    walks = generate_random_walks(G, num_walks=15, walk_length=15)
    walks_str = [[str(node) for node in walk] for walk in walks]
    
    print("Training Word2Vec model...")
    model = Word2Vec(walks_str, vector_size=embedding_dim, window=5, min_count=1, sg=1, workers=4, epochs=10, seed=42)
    return model.wv

def build_row_features(data_path, node_metrics_path, corridor_metrics_path, wv=None, embedding_dim=16):
    print("Loading data and pre-computed metrics...")
    df = pd.read_csv(data_path)
    node_df = pd.read_csv(node_metrics_path).fillna(0)
    corridor_df = pd.read_csv(corridor_metrics_path).fillna(0)
    
    node_metrics = node_df.set_index('node_id').to_dict(orient='index')
    corridor_metrics = corridor_df.set_index(['source_center', 'destination_center']).to_dict(orient='index')
    
    print("Engineering row-level segment features...")
    rows = []
    for idx, row in df.iterrows():
        src = row['source_center']
        dest = row['destination_center']
        
        entry = {
            'data': row['data'],
            'route_type': row['route_type'],
            'osrm_time': row['osrm_time'],
            'osrm_distance': row['osrm_distance'],
            'actual_distance_to_destination': row['actual_distance_to_destination'],
            'segment_actual_time': row['segment_actual_time'],
            'segment_osrm_time': row['segment_osrm_time'],
            'segment_osrm_distance': row['segment_osrm_distance'],
            'target_actual_time': row['actual_time']
        }
        
        corr_feat = corridor_metrics.get((src, dest), {})
        entry['corr_median_delay_ratio'] = corr_feat.get('median_delay_ratio', 1.0)
        entry['corr_total_delay_minutes'] = corr_feat.get('total_delay_minutes', 0.0)
        
        src_feat = node_metrics.get(src, {})
        dest_feat = node_metrics.get(dest, {})
        for k in ['in_degree', 'out_degree', 'betweenness_centrality', 'clustering_coefficient', 'total_delay_minutes']:
            entry[f'src_{k}'] = src_feat.get(k, 0.0)
            entry[f'dest_{k}'] = dest_feat.get(k, 0.0)
            
        if wv is not None:
            src_emb = wv[src] if src in wv else np.zeros(embedding_dim)
            dest_emb = wv[dest] if dest in wv else np.zeros(embedding_dim)
            for i in range(embedding_dim):
                entry[f'src_emb_{i}'] = src_emb[i]
                entry[f'dest_emb_{i}'] = dest_emb[i]
                
        rows.append(entry)
        
    feat_df = pd.DataFrame(rows)
    print(f"Features built for {len(feat_df)} checkpoint scans.")
    return feat_df

def train_and_evaluate(feat_df, model_dir, embedding_dim=16):
    feat_df = pd.get_dummies(feat_df, columns=['route_type'], drop_first=True)
    
    train_df = feat_df[feat_df['data'] == 'training']
    test_df = feat_df[feat_df['data'] == 'test']
    
    baseline_cols = ['osrm_time', 'osrm_distance', 'actual_distance_to_destination', 
                     'segment_actual_time', 'segment_osrm_time', 'segment_osrm_distance']
    for col in feat_df.columns:
        if col.startswith('route_type_'):
            baseline_cols.append(col)
            
    graph_cols = baseline_cols.copy()
    graph_cols.extend(['corr_median_delay_ratio', 'corr_total_delay_minutes'])
    for k in ['in_degree', 'out_degree', 'betweenness_centrality', 'clustering_coefficient', 'total_delay_minutes']:
        graph_cols.extend([f'src_{k}', f'dest_{k}'])
        
    for i in range(embedding_dim):
        graph_cols.extend([f'src_emb_{i}', f'dest_emb_{i}'])
        
    X_train_base = train_df[baseline_cols]
    y_train = train_df['target_actual_time']
    X_test_base = test_df[baseline_cols]
    y_test = test_df['target_actual_time']
    
    X_train_graph = train_df[graph_cols]
    X_test_graph = test_df[graph_cols]
    
    y_train_log = np.log1p(y_train)
    
    print("\nTraining Baseline LightGBM Model (Log target)...")
    base_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    base_model.fit(X_train_base, y_train_log)
    
    print("Training Graph-Enhanced LightGBM Model (Log target)...")
    graph_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    graph_model.fit(X_train_graph, y_train_log)
    
    y_pred_base = np.expm1(base_model.predict(X_test_base))
    y_pred_graph = np.expm1(graph_model.predict(X_test_graph))
    
    y_pred_base = np.clip(y_pred_base, 5, None)
    y_pred_graph = np.clip(y_pred_graph, 5, None)
    
    mae_base = mean_absolute_error(y_test, y_pred_base)
    mae_graph = mean_absolute_error(y_test, y_pred_graph)
    
    acc_base = np.mean(np.abs(y_test - y_pred_base) / y_test <= 0.15) * 100
    acc_graph = np.mean(np.abs(y_test - y_pred_graph) / y_test <= 0.15) * 100
    
    print("\n=== Model Performance Comparison on Test Set ===")
    print(f"Baseline Model MAE: {mae_base:.2f} minutes")
    print(f"Graph-Enhanced Model MAE: {mae_graph:.2f} minutes")
    print(f"Baseline Model Accuracy (% within 15% actual): {acc_base:.2f}%")
    print(f"Graph-Enhanced Model Accuracy (% within 15% actual): {acc_graph:.2f}%")
    
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "base_model.pkl"), 'wb') as f:
        pickle.dump(base_model, f)
    with open(os.path.join(model_dir, "graph_model.pkl"), 'wb') as f:
        pickle.dump(graph_model, f)
    print("Models saved successfully.")
        
    return mae_base, mae_graph, acc_base, acc_graph

if __name__ == '__main__':
    graph_path = "5. Serialized Models & Graphs/logistics_graph.pkl"
    data_path = "delivery_data.csv"
    node_metrics_path = "3. Precomputed Artifacts & Visuals/node_audit_results.csv"
    corridor_metrics_path = "3. Precomputed Artifacts & Visuals/corridor_audit_results.csv"
    model_dir = "5. Serialized Models & Graphs"
    
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
        
    embedding_dim = 16
    wv = learn_node_embeddings(G, embedding_dim=embedding_dim)
    
    feat_df = build_row_features(data_path, node_metrics_path, corridor_metrics_path, wv=wv, embedding_dim=embedding_dim)
    train_and_evaluate(feat_df, model_dir, embedding_dim=embedding_dim)
