import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_audit(graph_path, od_data_path, output_dir):
    print("Loading graph and OD data...")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    od_df = pd.read_csv(od_data_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Computing graph centrality metrics...")
    deg_in = dict(G.in_degree())
    deg_out = dict(G.out_degree())
    betweenness = nx.betweenness_centrality(G, weight=None)
    clustering = nx.clustering(G.to_undirected())
    
    node_metrics = []
    for node in G.nodes():
        node_metrics.append({
            'node_id': node,
            'name': G.nodes[node].get('name', 'Unknown'),
            'in_degree': deg_in.get(node, 0),
            'out_degree': deg_out.get(node, 0),
            'betweenness_centrality': betweenness.get(node, 0.0),
            'clustering_coefficient': clustering.get(node, 0.0)
        })
    node_df = pd.DataFrame(node_metrics)
    
    train_od = od_df[od_df['data'] == 'training'].copy()
    train_od['is_sla_breach'] = train_od['delay_ratio'] > 1.2
    train_od['delay_minutes'] = train_od['od_actual_time'] - train_od['od_osrm_time']
    
    src_delay = train_od.groupby('source_center').agg(
        src_delay_minutes=('delay_minutes', 'sum'),
        src_sla_breaches=('is_sla_breach', 'sum'),
        src_trips=('trip_uuid', 'count')
    ).reset_index()
    
    dest_delay = train_od.groupby('destination_center').agg(
        dest_delay_minutes=('delay_minutes', 'sum'),
        dest_sla_breaches=('is_sla_breach', 'sum'),
        dest_trips=('trip_uuid', 'count')
    ).reset_index()
    
    node_df = node_df.merge(src_delay, left_on='node_id', right_on='source_center', how='left')
    node_df = node_df.merge(dest_delay, left_on='node_id', right_on='destination_center', how='left')
    node_df = node_df.fillna(0)
    
    node_df['total_delay_minutes'] = node_df['src_delay_minutes'] + node_df['dest_delay_minutes']
    node_df['total_sla_breaches'] = node_df['src_sla_breaches'] + node_df['dest_sla_breaches']
    node_df['total_trips_handled'] = node_df['src_trips'] + node_df['dest_trips']
    
    node_df['betweenness_rank'] = node_df['betweenness_centrality'].rank(ascending=False)
    node_df['delay_rank'] = node_df['total_delay_minutes'].rank(ascending=False)
    node_df['composite_rank'] = (node_df['betweenness_rank'] + node_df['delay_rank']) / 2
    
    top_hubs = node_df.sort_values(by='composite_rank').head(10)
    
    print("\nTop 10 Bottleneck Hubs:")
    print(top_hubs[['node_id', 'name', 'betweenness_centrality', 'total_delay_minutes', 'total_sla_breaches', 'composite_rank']])
    
    node_df.to_csv(os.path.join(output_dir, 'node_audit_results.csv'), index=False)
    
    corridor_df = train_od.groupby(['source_center', 'source_name', 'destination_center', 'destination_name']).agg(
        median_delay_ratio=('delay_ratio', 'median'),
        total_delay_minutes=('delay_minutes', 'sum'),
        sla_breach_count=('is_sla_breach', 'sum'),
        total_trips=('trip_uuid', 'count')
    ).reset_index()
    
    chronic_corridors = corridor_df[corridor_df['median_delay_ratio'] > 1.2].sort_values(by='sla_breach_count', ascending=False)
    
    print("\nTop 10 Chronically Delayed Corridors (ranked by SLA breaches):")
    print(chronic_corridors.head(10)[['source_name', 'destination_name', 'median_delay_ratio', 'total_delay_minutes', 'sla_breach_count']])
    
    corridor_df.to_csv(os.path.join(output_dir, 'corridor_audit_results.csv'), index=False)
    
    print("Generating visualizations...")
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white")
    
    top_nodes = node_df.sort_values(by='total_trips_handled', ascending=False).head(50)['node_id'].tolist()
    subG = G.subgraph(top_nodes)
    
    pos = nx.spring_layout(subG, seed=42)
    sizes = [betweenness[node] * 5000 + 100 for node in subG.nodes()]
    delay_colors = [node_df[node_df['node_id'] == node]['total_delay_minutes'].values[0] for node in subG.nodes()]
    
    nx.draw_networkx_nodes(subG, pos, node_size=sizes, node_color=delay_colors, cmap=plt.cm.Oranges, alpha=0.9)
    nx.draw_networkx_edges(subG, pos, width=1.5, alpha=0.5, edge_color='gray', arrows=True)
    
    top_5_nodes = top_hubs.head(5)['node_id'].tolist()
    nx.draw_networkx_nodes(subG, pos, nodelist=[n for n in top_5_nodes if n in subG], 
                           node_size=[betweenness[n] * 5000 + 100 for n in top_5_nodes if n in subG], 
                           node_color='red', edgecolors='black', linewidths=2)
    
    labels = {node: subG.nodes[node].get('name', '').split('_')[0] for node in subG.nodes()}
    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=8, font_family='sans-serif')
    
    plt.title("Delhivery Key Network Hubs & Bottlenecks\nNode Size: Betweenness Centrality | Color intensity: Total Delay Minutes", fontsize=14, fontweight='bold')
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Oranges, norm=plt.Normalize(vmin=min(delay_colors), vmax=max(delay_colors)))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Total Delay Minutes (Mins)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'network_bottlenecks.png'), dpi=300)
    plt.close()
    print("Saved network bottlenecks visualization.")

if __name__ == '__main__':
    graph_path = "5. Serialized Models & Graphs/logistics_graph.pkl"
    od_data_path = "3. Precomputed Artifacts & Visuals/od_data_processed.csv"
    output_dir = "3. Precomputed Artifacts & Visuals"
    run_audit(graph_path, od_data_path, output_dir)
