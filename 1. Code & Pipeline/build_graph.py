import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os

def parse_and_clean_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    df['source_name'] = df['source_name'].fillna("Unknown Source")
    df['destination_name'] = df['destination_name'].fillna("Unknown Destination")
    
    print("Parsing timestamps...")
    df['cutoff_dt'] = pd.to_datetime(df['cutoff_timestamp'], errors='coerce')
    df['cutoff_dt'] = df.groupby('trip_uuid')['cutoff_dt'].ffill().bfill()
    
    df['hour'] = df['cutoff_dt'].dt.hour.fillna(12).astype(int)
    df['time_of_day'] = pd.cut(df['hour'], bins=[-1, 5, 11, 17, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    df = df.sort_values(by=['trip_uuid', 'source_center', 'destination_center', 'actual_distance_to_destination'])
    
    print("Aggregating to OD-pair level...")
    od_groups = df.groupby(['trip_uuid', 'source_center', 'destination_center'])
    od_df = od_groups.last().reset_index()
    
    od_df = od_df.rename(columns={
        'actual_time': 'od_actual_time',
        'osrm_time': 'od_osrm_time',
        'osrm_distance': 'od_osrm_distance',
        'actual_distance_to_destination': 'od_actual_distance'
    })
    
    od_df['delay_ratio'] = od_df['od_actual_time'] / (od_df['od_osrm_time'] + 1)
    od_df['delay_minutes'] = od_df['od_actual_time'] - od_df['od_osrm_time']
    
    print(f"Aggregated {len(df)} rows into {len(od_df)} OD-pair segments.")
    return od_df

def build_logistics_graph(od_df):
    print("Building logistics graph from training data...")
    train_od = od_df[od_df['data'] == 'training']
    
    G = nx.DiGraph()
    corridors = train_od.groupby(['source_center', 'destination_center'])
    
    source_names = train_od.set_index('source_center')['source_name'].to_dict()
    dest_names = train_od.set_index('destination_center')['destination_name'].to_dict()
    node_names = {**source_names, **dest_names}
    
    for node_id, name in node_names.items():
        G.add_node(node_id, name=name)
        
    print(f"Added {G.number_of_nodes()} nodes.")
    
    for (src, dest), group in corridors:
        median_delay = group['delay_ratio'].median()
        total_delay = group['delay_minutes'].sum()
        trip_count = len(group)
        
        group_ftl = group[group['route_type'] == 'FTL']
        group_cart = group[group['route_type'] == 'Carting']
        
        delay_ftl = group_ftl['delay_ratio'].median() if len(group_ftl) > 0 else median_delay
        delay_cart = group_cart['delay_ratio'].median() if len(group_cart) > 0 else median_delay
        
        delay_tod = {}
        for tod in ['Night', 'Morning', 'Afternoon', 'Evening']:
            g_tod = group[group['time_of_day'] == tod]
            delay_tod[tod] = g_tod['delay_ratio'].median() if len(g_tod) > 0 else median_delay
            
        G.add_edge(
            src, dest,
            weight=median_delay,
            total_delay=total_delay,
            trip_count=trip_count,
            weight_ftl=delay_ftl,
            weight_carting=delay_cart,
            weight_night=delay_tod['Night'],
            weight_morning=delay_tod['Morning'],
            weight_afternoon=delay_tod['Afternoon'],
            weight_evening=delay_tod['Evening']
        )
        
    print(f"Added {G.number_of_edges()} edges.")
    return G

if __name__ == '__main__':
    data_path = "delivery_data.csv"
    od_df = parse_and_clean_data(data_path)
    
    # Save OD dataframe for downstream steps
    od_df.to_csv("3. Precomputed Artifacts & Visuals/od_data_processed.csv", index=False)
    print("Saved aggregated OD data.")
    
    G = build_logistics_graph(od_df)
    
    with open("5. Serialized Models & Graphs/logistics_graph.pkl", 'wb') as f:
        pickle.dump(G, f)
    print("Saved network graph pickle.")
