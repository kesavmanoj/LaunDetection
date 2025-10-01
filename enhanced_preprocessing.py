#!/usr/bin/env python3
"""
Enhanced Preprocessing for IBM AML Multi-GNN Model
==================================================

This script implements comprehensive preprocessing for the IBM AML Synthetic Dataset
based on analysis of HI-Small_report.json and sample_subgraph.gpickle.

Key Features:
- Enhanced node and edge feature engineering
- Advanced class imbalance handling
- Temporal feature encoding
- Network topology features
- Memory-efficient processing
- Graph sampling and augmentation
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx, from_networkx
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import json
import os
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedAMLPreprocessor:
    """
    Enhanced preprocessing for IBM AML Multi-GNN model
    """
    
    def __init__(self, data_path, output_path, chunk_size=10000):
        self.data_path = data_path
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        
    def load_data(self):
        """Load and validate data"""
        print("Loading IBM AML dataset...")
        
        # Load transactions
        trans_file = os.path.join(self.data_path, 'HI-Small_Trans.csv')
        if os.path.exists(trans_file):
            self.transactions = pd.read_csv(trans_file)
            print(f"✓ Loaded {len(self.transactions)} transactions")
        else:
            raise FileNotFoundError(f"Transaction file not found: {trans_file}")
        
        # Load accounts
        accounts_file = os.path.join(self.data_path, 'HI-Small_accounts.csv')
        if os.path.exists(accounts_file):
            self.accounts = pd.read_csv(accounts_file)
            print(f"✓ Loaded {len(self.accounts)} accounts")
        else:
            # Create accounts from transactions
            print("Creating accounts from transaction data...")
            self.accounts = self._create_accounts_from_transactions()
        
        # Validate data
        self._validate_data()
        
    def _create_accounts_from_transactions(self):
        """Create accounts from transaction data"""
        all_accounts = set(self.transactions['Account'].tolist() + 
                          self.transactions['Account.1'].tolist())
        
        accounts_data = {
            'Account Number': list(all_accounts),
            'Bank Name': [f"Bank_{i}" for i in range(len(all_accounts))],
            'Bank ID': [f"B{i}" for i in range(len(all_accounts))],
            'Entity ID': [f"E{i}" for i in range(len(all_accounts))],
            'Entity Name': [f"Entity_{i}" for i in range(len(all_accounts))]
        }
        
        return pd.DataFrame(accounts_data)
    
    def _validate_data(self):
        """Validate data quality"""
        print("Validating data quality...")
        
        # Check for missing values
        missing_trans = self.transactions.isnull().sum().sum()
        missing_accounts = self.accounts.isnull().sum().sum()
        
        print(f"Missing values - Transactions: {missing_trans}, Accounts: {missing_accounts}")
        
        # Check class distribution
        if 'Is Laundering' in self.transactions.columns:
            class_dist = self.transactions['Is Laundering'].value_counts()
            print(f"Class distribution: {class_dist}")
            print(f"SAR rate: {class_dist[1] / len(self.transactions):.4f}")
        
        # Check data types
        print(f"Transaction columns: {list(self.transactions.columns)}")
        print(f"Account columns: {list(self.accounts.columns)}")
        
    def create_enhanced_node_features(self):
        """Create comprehensive node features"""
        print("Creating enhanced node features...")
        
        node_features = {}
        
        for _, account in self.accounts.iterrows():
            account_id = account['Account Number']
            
            # Get account transactions
            account_trans = self.transactions[
                (self.transactions['Account'] == account_id) | 
                (self.transactions['Account.1'] == account_id)
            ]
            
            if len(account_trans) == 0:
                # Default features for accounts with no transactions
                node_features[account_id] = self._get_default_node_features()
                continue
            
            # Basic transaction features
            sent_trans = account_trans[account_trans['Account'] == account_id]
            received_trans = account_trans[account_trans['Account.1'] == account_id]
            
            # Amount features
            total_sent = sent_trans['Amount Paid'].sum() if len(sent_trans) > 0 else 0
            total_received = received_trans['Amount Received'].sum() if len(received_trans) > 0 else 0
            avg_amount = account_trans['Amount Paid'].mean()
            max_amount = account_trans['Amount Paid'].max()
            min_amount = account_trans['Amount Paid'].min()
            
            # Temporal features
            timestamps = pd.to_datetime(account_trans['Timestamp'])
            temporal_span = (timestamps.max() - timestamps.min()).days
            transaction_frequency = len(account_trans) / max(1, temporal_span)
            
            # Currency and bank diversity
            currency_diversity = account_trans['Payment Currency'].nunique()
            bank_diversity = account_trans['To Bank'].nunique()
            
            # Time-based features
            night_transactions = timestamps.dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5, 6]).sum()
            weekend_transactions = timestamps.dt.weekday.isin([5, 6]).sum()
            night_ratio = night_transactions / len(account_trans)
            weekend_ratio = weekend_transactions / len(account_trans)
            
            # Risk indicators
            is_crypto_bank = 'Crytpo' in str(account_id)
            is_international = currency_diversity > 1
            is_high_frequency = transaction_frequency > 1.0
            
            # Network features (will be updated later)
            in_degree = 0
            out_degree = 0
            betweenness_centrality = 0
            pagerank = 0
            
            node_features[account_id] = {
                'transaction_count': len(account_trans),
                'total_sent': total_sent,
                'total_received': total_received,
                'avg_amount': avg_amount,
                'max_amount': max_amount,
                'min_amount': min_amount,
                'temporal_span': temporal_span,
                'transaction_frequency': transaction_frequency,
                'currency_diversity': currency_diversity,
                'bank_diversity': bank_diversity,
                'night_ratio': night_ratio,
                'weekend_ratio': weekend_ratio,
                'is_crypto_bank': int(is_crypto_bank),
                'is_international': int(is_international),
                'is_high_frequency': int(is_high_frequency),
                'in_degree': in_degree,
                'out_degree': out_degree,
                'betweenness_centrality': betweenness_centrality,
                'pagerank': pagerank
            }
        
        return node_features
    
    def _get_default_node_features(self):
        """Get default features for accounts with no transactions"""
        return {
            'transaction_count': 0,
            'total_sent': 0,
            'total_received': 0,
            'avg_amount': 0,
            'max_amount': 0,
            'min_amount': 0,
            'temporal_span': 0,
            'transaction_frequency': 0,
            'currency_diversity': 0,
            'bank_diversity': 0,
            'night_ratio': 0,
            'weekend_ratio': 0,
            'is_crypto_bank': 0,
            'is_international': 0,
            'is_high_frequency': 0,
            'in_degree': 0,
            'out_degree': 0,
            'betweenness_centrality': 0,
            'pagerank': 0
        }
    
    def create_enhanced_edge_features(self):
        """Create comprehensive edge features"""
        print("Creating enhanced edge features...")
        
        edge_features = []
        edge_labels = []
        
        # Prepare encoders
        self._prepare_encoders()
        
        for _, transaction in self.transactions.iterrows():
            # Temporal features
            timestamp = pd.to_datetime(transaction['Timestamp'])
            temporal_features = self._create_temporal_features(timestamp)
            
            # Amount features
            amount_features = self._create_amount_features(transaction)
            
            # Categorical features
            categorical_features = self._create_categorical_features(transaction)
            
            # Combine all features
            edge_feature = temporal_features + amount_features + categorical_features
            edge_features.append(edge_feature)
            
            # Label
            edge_labels.append(transaction['Is Laundering'])
        
        return np.array(edge_features), np.array(edge_labels)
    
    def _prepare_encoders(self):
        """Prepare encoders for categorical features"""
        # Currency encoder
        currencies = self.transactions['Payment Currency'].unique()
        self.encoders['currency'] = LabelEncoder()
        self.encoders['currency'].fit(currencies)
        
        # Format encoder
        formats = self.transactions['Payment Format'].unique()
        self.encoders['format'] = LabelEncoder()
        self.encoders['format'].fit(formats)
        
        # Bank encoder
        banks = self.transactions['From Bank'].unique()
        self.encoders['bank'] = LabelEncoder()
        self.encoders['bank'].fit(banks)
    
    def _create_temporal_features(self, timestamp):
        """Create cyclic temporal features"""
        # Hour features
        hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
        hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
        
        # Day of week features
        day_sin = np.sin(2 * np.pi * timestamp.dayofweek / 7)
        day_cos = np.cos(2 * np.pi * timestamp.dayofweek / 7)
        
        # Day of month features
        day_month_sin = np.sin(2 * np.pi * timestamp.day / 31)
        day_month_cos = np.cos(2 * np.pi * timestamp.day / 31)
        
        # Month features
        month_sin = np.sin(2 * np.pi * timestamp.month / 12)
        month_cos = np.cos(2 * np.pi * timestamp.month / 12)
        
        # Year features (normalized)
        year_normalized = (timestamp.year - 2020) / 5
        
        return [hour_sin, hour_cos, day_sin, day_cos, 
                day_month_sin, day_month_cos, month_sin, month_cos, year_normalized]
    
    def _create_amount_features(self, transaction):
        """Create amount-based features"""
        amount_paid = transaction['Amount Paid']
        amount_received = transaction['Amount Received']
        
        # Log transformation
        amount_paid_log = np.log1p(amount_paid)
        amount_received_log = np.log1p(amount_received)
        
        # Normalized amounts (will be updated with scaler)
        amount_paid_norm = amount_paid  # Will be normalized later
        amount_received_norm = amount_received  # Will be normalized later
        
        # Amount ratios
        amount_ratio = amount_paid / max(amount_received, 1)
        
        return [amount_paid_log, amount_received_log, 
                amount_paid_norm, amount_received_norm, amount_ratio]
    
    def _create_categorical_features(self, transaction):
        """Create categorical features"""
        # Currency encoding
        currency_encoded = self.encoders['currency'].transform([transaction['Payment Currency']])[0]
        
        # Format encoding
        format_encoded = self.encoders['format'].transform([transaction['Payment Format']])[0]
        
        # Bank encoding
        bank_encoded = self.encoders['bank'].transform([transaction['From Bank']])[0]
        
        return [currency_encoded, format_encoded, bank_encoded]
    
    def create_graph_structure(self, node_features, edge_features, edge_labels):
        """Create graph structure with network features"""
        print("Creating graph structure...")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with features
        for account_id, features in node_features.items():
            G.add_node(account_id, **features)
        
        # Add edges with features
        for i, (_, transaction) in enumerate(self.transactions.iterrows()):
            from_account = transaction['Account']
            to_account = transaction['Account.1']
            
            if from_account in G.nodes() and to_account in G.nodes():
                G.add_edge(from_account, to_account, 
                          features=edge_features[i], 
                          label=edge_labels[i])
        
        # Add network topology features
        self._add_network_features(G)
        
        # Convert to PyTorch Geometric format
        pyg_data = self._convert_to_pyg(G, node_features, edge_features, edge_labels)
        
        return pyg_data, G
    
    def _add_network_features(self, G):
        """Add network topology features"""
        print("Adding network topology features...")
        
        # Centrality measures
        in_degree_centrality = nx.in_degree_centrality(G)
        out_degree_centrality = nx.out_degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # PageRank
        pagerank = nx.pagerank(G)
        
        # Clustering coefficient
        clustering = nx.clustering(G.to_undirected())
        
        # Update node features
        for node in G.nodes():
            G.nodes[node]['in_degree_centrality'] = in_degree_centrality.get(node, 0)
            G.nodes[node]['out_degree_centrality'] = out_degree_centrality.get(node, 0)
            G.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
            G.nodes[node]['closeness_centrality'] = closeness_centrality.get(node, 0)
            G.nodes[node]['pagerank'] = pagerank.get(node, 0)
            G.nodes[node]['clustering_coefficient'] = clustering.get(node, 0)
            G.nodes[node]['in_degree'] = G.in_degree(node)
            G.nodes[node]['out_degree'] = G.out_degree(node)
            G.nodes[node]['total_degree'] = G.degree(node)
    
    def _convert_to_pyg(self, G, node_features, edge_features, edge_labels):
        """Convert NetworkX graph to PyTorch Geometric format"""
        print("Converting to PyTorch Geometric format...")
        
        # Node features
        node_list = list(G.nodes())
        node_feature_matrix = []
        node_labels = []
        
        for node in node_list:
            features = G.nodes[node]
            feature_vector = [features.get(key, 0) for key in [
                'transaction_count', 'total_sent', 'total_received', 'avg_amount',
                'max_amount', 'min_amount', 'temporal_span', 'transaction_frequency',
                'currency_diversity', 'bank_diversity', 'night_ratio', 'weekend_ratio',
                'is_crypto_bank', 'is_international', 'is_high_frequency',
                'in_degree_centrality', 'out_degree_centrality', 'betweenness_centrality',
                'closeness_centrality', 'pagerank', 'clustering_coefficient',
                'in_degree', 'out_degree', 'total_degree'
            ]]
            node_feature_matrix.append(feature_vector)
            
            # Node label (majority vote of incident edges)
            incident_edges = [edge_labels[i] for i, (_, transaction) in enumerate(self.transactions.iterrows())
                            if transaction['Account'] == node or transaction['Account.1'] == node]
            node_label = 1 if sum(incident_edges) > len(incident_edges) / 2 else 0
            node_labels.append(node_label)
        
        # Edge features
        edge_list = list(G.edges())
        edge_feature_matrix = []
        edge_label_list = []
        
        for edge in edge_list:
            edge_data = G.edges[edge]
            edge_feature_matrix.append(edge_data['features'])
            edge_label_list.append(edge_data['label'])
        
        # Create PyTorch Geometric Data object
        node_features_tensor = torch.tensor(node_feature_matrix, dtype=torch.float32)
        edge_features_tensor = torch.tensor(edge_feature_matrix, dtype=torch.float32)
        node_labels_tensor = torch.tensor(node_labels, dtype=torch.long)
        edge_labels_tensor = torch.tensor(edge_label_list, dtype=torch.long)
        
        # Edge index
        edge_index = torch.tensor([[node_list.index(edge[0]), node_list.index(edge[1])] 
                                 for edge in edge_list], dtype=torch.long).t().contiguous()
        
        return Data(x=node_features_tensor, 
                   edge_index=edge_index,
                   edge_attr=edge_features_tensor,
                   y=node_labels_tensor,
                   edge_y=edge_labels_tensor)
    
    def handle_class_imbalance(self, X, y, strategy='smote'):
        """Handle class imbalance using multiple strategies"""
        print(f"Handling class imbalance using {strategy}...")
        
        if strategy == 'smote':
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
        elif strategy == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
            
        elif strategy == 'combined':
            # First oversample minority class
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_oversampled, y_oversampled = smote.fit_resample(X, y)
            
            # Then undersample majority class
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X_oversampled, y_oversampled)
        
        print(f"Original distribution: {np.bincount(y)}")
        print(f"Resampled distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def create_cost_sensitive_weights(self, y):
        """Create cost-sensitive class weights"""
        print("Creating cost-sensitive class weights...")
        
        # Compute balanced class weights
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        
        # Additional cost for false negatives (missed illicit transactions)
        cost_multiplier = 10.0  # Emphasize false negatives
        adjusted_weights = class_weights * cost_multiplier
        
        weight_dict = dict(zip(classes, adjusted_weights))
        print(f"Class weights: {weight_dict}")
        
        return weight_dict
    
    def normalize_features(self, features, feature_type='node'):
        """Normalize features using robust scaling"""
        print(f"Normalizing {feature_type} features...")
        
        scaler = RobustScaler()
        normalized_features = scaler.fit_transform(features)
        
        # Store scaler for later use
        self.scalers[feature_type] = scaler
        
        return normalized_features
    
    def create_graph_samples(self, G, sample_size=1000, n_samples=100):
        """Create multiple graph samples for training"""
        print(f"Creating {n_samples} graph samples of size {sample_size}...")
        
        samples = []
        
        for i in range(n_samples):
            # Random node sampling
            nodes = list(G.nodes())
            sampled_nodes = np.random.choice(nodes, 
                                           size=min(sample_size, len(nodes)), 
                                           replace=False)
            
            # Create subgraph
            subgraph = G.subgraph(sampled_nodes)
            
            # Ensure connectivity
            if nx.is_weakly_connected(subgraph):
                samples.append(subgraph)
            else:
                # Try to find connected components
                components = list(nx.weakly_connected_components(subgraph))
                if components:
                    largest_component = max(components, key=len)
                    if len(largest_component) > 100:  # Minimum size threshold
                        samples.append(subgraph.subgraph(largest_component))
        
        print(f"Created {len(samples)} valid graph samples")
        return samples
    
    def save_processed_data(self, pyg_data, G, node_features, edge_features, edge_labels):
        """Save processed data"""
        print("Saving processed data...")
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Save PyTorch Geometric data
        torch.save(pyg_data, os.path.join(self.output_path, 'enhanced_graph.pt'))
        
        # Save NetworkX graph
        nx.write_gpickle(G, os.path.join(self.output_path, 'enhanced_graph.gpickle'))
        
        # Save feature matrices
        np.save(os.path.join(self.output_path, 'node_features.npy'), node_features)
        np.save(os.path.join(self.output_path, 'edge_features.npy'), edge_features)
        np.save(os.path.join(self.output_path, 'edge_labels.npy'), edge_labels)
        
        # Save scalers and encoders
        import pickle
        with open(os.path.join(self.output_path, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)
        
        with open(os.path.join(self.output_path, 'encoders.pkl'), 'wb') as f:
            pickle.dump(self.encoders, f)
        
        # Save metadata
        metadata = {
            'num_nodes': pyg_data.num_nodes,
            'num_edges': pyg_data.num_edges,
            'node_feature_dim': pyg_data.x.shape[1],
            'edge_feature_dim': pyg_data.edge_attr.shape[1],
            'class_distribution': np.bincount(pyg_data.y.numpy()).tolist(),
            'edge_class_distribution': np.bincount(pyg_data.edge_y.numpy()).tolist()
        }
        
        with open(os.path.join(self.output_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved processed data to {self.output_path}")
        print(f"  - Nodes: {metadata['num_nodes']}")
        print(f"  - Edges: {metadata['num_edges']}")
        print(f"  - Node features: {metadata['node_feature_dim']}")
        print(f"  - Edge features: {metadata['edge_feature_dim']}")
        print(f"  - Class distribution: {metadata['class_distribution']}")
    
    def run_full_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("=" * 60)
        print("Enhanced AML Preprocessing Pipeline")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Create enhanced features
        node_features = self.create_enhanced_node_features()
        edge_features, edge_labels = self.create_enhanced_edge_features()
        
        # Normalize features
        node_feature_matrix = np.array([list(features.values()) for features in node_features.values()])
        node_features_normalized = self.normalize_features(node_feature_matrix, 'node')
        edge_features_normalized = self.normalize_features(edge_features, 'edge')
        
        # Create graph structure
        pyg_data, G = self.create_graph_structure(node_features, edge_features_normalized, edge_labels)
        
        # Handle class imbalance
        if hasattr(pyg_data, 'y'):
            X = pyg_data.x.numpy()
            y = pyg_data.y.numpy()
            X_resampled, y_resampled = self.handle_class_imbalance(X, y, strategy='smote')
            
            # Update PyTorch Geometric data
            pyg_data.x = torch.tensor(X_resampled, dtype=torch.float32)
            pyg_data.y = torch.tensor(y_resampled, dtype=torch.long)
        
        # Create cost-sensitive weights
        if hasattr(pyg_data, 'y'):
            class_weights = self.create_cost_sensitive_weights(pyg_data.y.numpy())
        
        # Create graph samples
        graph_samples = self.create_graph_samples(G, sample_size=1000, n_samples=50)
        
        # Save processed data
        self.save_processed_data(pyg_data, G, node_features_normalized, 
                                edge_features_normalized, edge_labels)
        
        print("✓ Enhanced preprocessing completed successfully!")
        return pyg_data, G, class_weights, graph_samples

def main():
    """Main execution function"""
    # Configuration
    data_path = "/content/drive/MyDrive/LaunDetection/data/raw"
    output_path = "/content/drive/MyDrive/LaunDetection/data/processed/enhanced"
    
    # Create preprocessor
    preprocessor = EnhancedAMLPreprocessor(data_path, output_path)
    
    # Run preprocessing
    pyg_data, G, class_weights, graph_samples = preprocessor.run_full_preprocessing()
    
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"✓ Enhanced node features: {pyg_data.x.shape[1]} dimensions")
    print(f"✓ Enhanced edge features: {pyg_data.edge_attr.shape[1]} dimensions")
    print(f"✓ Graph samples created: {len(graph_samples)}")
    print(f"✓ Class weights: {class_weights}")
    print(f"✓ Data saved to: {output_path}")

if __name__ == "__main__":
    main()
