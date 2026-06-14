import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def analyze_ftl_vs_carting(od_data_path, node_metrics_path, output_dir):
    print("Loading data for FTL vs Carting framework...")
    od_df = pd.read_csv(od_data_path)
    node_df = pd.read_csv(node_metrics_path)
    
    train_df = od_df[od_df['data'] == 'training'].copy()
    corridors = train_df.groupby(['source_center', 'source_name', 'destination_center', 'destination_name'])
    
    corridor_comparisons = []
    for (src, src_name, dest, dest_name), group in corridors:
        ftl_group = group[group['route_type'] == 'FTL']
        cart_group = group[group['route_type'] == 'Carting']
        
        if len(ftl_group) >= 1 and len(cart_group) >= 1:
            ftl_delay = ftl_group['delay_ratio'].median()
            cart_delay = cart_group['delay_ratio'].median()
            ftl_time = ftl_group['od_actual_time'].median()
            cart_time = cart_group['od_actual_time'].median()
            ftl_dist = ftl_group['od_actual_distance'].median()
            
            corridor_comparisons.append({
                'source_center': src,
                'source_name': src_name,
                'destination_center': dest,
                'destination_name': dest_name,
                'distance': ftl_dist,
                'ftl_median_delay_ratio': ftl_delay,
                'cart_median_delay_ratio': cart_delay,
                'ftl_median_time': ftl_time,
                'cart_median_time': cart_time,
                'time_difference_mins': cart_time - ftl_time,
                'delay_ratio_difference': cart_delay - ftl_delay
            })
            
    comp_df = pd.DataFrame(corridor_comparisons)
    print(f"Found {len(comp_df)} corridors with both FTL and Carting trips.")
    
    if len(comp_df) == 0:
        return None
        
    node_df = node_df.set_index('node_id')
    comp_df = comp_df.join(node_df[['betweenness_centrality', 'in_degree', 'out_degree', 'clustering_coefficient']], on='source_center')
    comp_df = comp_df.rename(columns={
        'betweenness_centrality': 'src_betweenness',
        'in_degree': 'src_in_degree',
        'out_degree': 'src_out_degree',
        'clustering_coefficient': 'src_clustering'
    })
    
    comp_df['recommend_ftl'] = (comp_df['time_difference_mins'] > 10) | (comp_df['delay_ratio_difference'] > 0.1)
    
    print("\nSummary of Recommendations on overlap corridors:")
    print(comp_df['recommend_ftl'].value_counts())
    
    X = comp_df[['distance', 'src_betweenness', 'src_in_degree', 'src_out_degree', 'src_clustering']].fillna(0)
    y = comp_df['recommend_ftl']
    
    if y.value_counts().min() < 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("\nClassifier Performance Report:")
    print(classification_report(y_test, y_pred))
    
    comp_df.to_csv(os.path.join(output_dir, 'ftl_vs_carting_recommendations.csv'), index=False)
    
    with open("5. Serialized Models & Graphs/route_type_classifier.pkl", 'wb') as f:
        pickle.dump(clf, f)
    print("Saved FTL vs Carting recommendations and classifier.")
    
    print("\nExample Recommendations (Top 5 corridors where FTL is recommended over Carting):")
    print(comp_df[comp_df['recommend_ftl']].sort_values(by='time_difference_mins', ascending=False).head(5)[
        ['source_name', 'destination_name', 'distance', 'ftl_median_time', 'cart_median_time', 'time_difference_mins']
    ])
    
    return comp_df

if __name__ == '__main__':
    od_data_path = "3. Precomputed Artifacts & Visuals/od_data_processed.csv"
    node_metrics_path = "3. Precomputed Artifacts & Visuals/node_audit_results.csv"
    output_dir = "3. Precomputed Artifacts & Visuals"
    analyze_ftl_vs_carting(od_data_path, node_metrics_path, output_dir)
