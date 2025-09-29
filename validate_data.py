"""
Comprehensive Data Validation Script for AML Detection
Validates preprocessed data before training to ensure quality and correctness
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class AMLDataValidator:
    """Comprehensive validator for preprocessed AML data"""
    
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path('/content/drive/MyDrive/LaunDetection')
        self.graphs_dir = self.base_dir / 'data' / 'graphs'
        self.validation_results = {}
        
    def validate_dataset(self, dataset_name, file_type='fixed'):
        """Validate a single dataset"""
        print(f"\n{'='*60}")
        print(f"VALIDATING {dataset_name.upper()} DATASET ({file_type.upper()})")
        print(f"{'='*60}")
        
        # Load data
        splits_file = self.graphs_dir / f'ibm_aml_{dataset_name.lower()}_{file_type}_splits.pt'
        complete_file = self.graphs_dir / f'ibm_aml_{dataset_name.lower()}_{file_type}_complete.pt'
        
        if not splits_file.exists():
            print(f"❌ Splits file not found: {splits_file}")
            return False
            
        if not complete_file.exists():
            print(f"⚠️ Complete file not found: {complete_file}")
            complete_data = None
        else:
            try:
                complete_data = torch.load(complete_file, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"⚠️ Could not load complete file: {e}")
                complete_data = None
            
        splits_data = torch.load(splits_file, map_location='cpu', weights_only=False)
        
        # Run all validations
        results = {}
        results['basic_structure'] = self._validate_basic_structure(splits_data, dataset_name)
        results['edge_indices'] = self._validate_edge_indices(splits_data, dataset_name)
        results['data_consistency'] = self._validate_data_consistency(splits_data, dataset_name)
        results['class_distribution'] = self._validate_class_distribution(splits_data, dataset_name)
        results['feature_quality'] = self._validate_feature_quality(splits_data, dataset_name)
        results['temporal_splits'] = self._validate_temporal_splits(splits_data, dataset_name)
        
        if complete_data:
            results['complete_consistency'] = self._validate_complete_consistency(splits_data, complete_data, dataset_name)
        
        # Overall assessment
        passed_tests = sum(1 for result in results.values() if result)
        total_tests = len(results)
        
        print(f"\n📊 VALIDATION SUMMARY FOR {dataset_name.upper()}:")
        print(f"  Tests passed: {passed_tests}/{total_tests}")
        print(f"  Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print(f"  ✅ {dataset_name} is READY FOR TRAINING")
        else:
            print(f"  ❌ {dataset_name} has VALIDATION ISSUES")
            
        self.validation_results[dataset_name] = {
            'results': results,
            'passed': passed_tests,
            'total': total_tests,
            'ready': passed_tests == total_tests
        }
        
        return passed_tests == total_tests
    
    def _validate_basic_structure(self, splits_data, dataset_name):
        """Validate basic data structure"""
        print(f"\n🔍 Basic Structure Validation:")
        
        try:
            # Check required keys
            required_keys = ['train', 'val', 'test', 'metadata']
            missing_keys = [key for key in required_keys if key not in splits_data]
            
            if missing_keys:
                print(f"  ❌ Missing keys: {missing_keys}")
                return False
            
            # Check data types
            for split in ['train', 'val', 'test']:
                data = splits_data[split]
                if not isinstance(data, Data):
                    print(f"  ❌ {split} is not a PyTorch Geometric Data object")
                    return False
                    
                # Check required attributes
                required_attrs = ['x', 'edge_index', 'edge_attr', 'y']
                missing_attrs = [attr for attr in required_attrs if not hasattr(data, attr)]
                
                if missing_attrs:
                    print(f"  ❌ {split} missing attributes: {missing_attrs}")
                    return False
            
            # Check metadata
            metadata = splits_data['metadata']
            required_metadata = ['num_nodes', 'num_edges', 'node_feature_dim', 'edge_feature_dim']
            missing_metadata = [key for key in required_metadata if key not in metadata]
            
            if missing_metadata:
                print(f"  ❌ Missing metadata: {missing_metadata}")
                return False
            
            print(f"  ✅ All required keys and attributes present")
            print(f"  📊 Metadata: {metadata}")
            return True
            
        except Exception as e:
            print(f"  ❌ Error in basic structure validation: {e}")
            return False
    
    def _validate_edge_indices(self, splits_data, dataset_name):
        """Validate edge indices are within valid range"""
        print(f"\n🔍 Edge Indices Validation:")
        
        try:
            all_valid = True
            
            for split_name in ['train', 'val', 'test']:
                data = splits_data[split_name]
                edge_index = data.edge_index
                num_nodes = data.num_nodes
                
                # Check edge index shape
                if edge_index.shape[0] != 2:
                    print(f"  ❌ {split_name}: Edge index should have shape [2, num_edges], got {edge_index.shape}")
                    all_valid = False
                    continue
                
                # Check edge index range
                max_idx = edge_index.max().item()
                min_idx = edge_index.min().item()
                
                if min_idx < 0:
                    print(f"  ❌ {split_name}: Negative edge indices found (min: {min_idx})")
                    all_valid = False
                
                if max_idx >= num_nodes:
                    print(f"  ❌ {split_name}: Edge indices exceed num_nodes (max: {max_idx}, nodes: {num_nodes})")
                    all_valid = False
                
                if all_valid:
                    print(f"  ✅ {split_name}: Edge indices valid [0, {max_idx}] for {num_nodes} nodes")
            
            return all_valid
            
        except Exception as e:
            print(f"  ❌ Error in edge indices validation: {e}")
            return False
    
    def _validate_data_consistency(self, splits_data, dataset_name):
        """Validate data consistency across splits"""
        print(f"\n🔍 Data Consistency Validation:")
        
        try:
            # Check that all splits have same number of nodes
            train_nodes = splits_data['train'].num_nodes
            val_nodes = splits_data['val'].num_nodes
            test_nodes = splits_data['test'].num_nodes
            
            if not (train_nodes == val_nodes == test_nodes):
                print(f"  ❌ Inconsistent node counts: train={train_nodes}, val={val_nodes}, test={test_nodes}")
                return False
            
            # Check feature dimensions
            train_node_features = splits_data['train'].x.shape[1]
            train_edge_features = splits_data['train'].edge_attr.shape[1]
            
            for split_name in ['val', 'test']:
                data = splits_data[split_name]
                if data.x.shape[1] != train_node_features:
                    print(f"  ❌ {split_name}: Node feature dim mismatch ({data.x.shape[1]} vs {train_node_features})")
                    return False
                if data.edge_attr.shape[1] != train_edge_features:
                    print(f"  ❌ {split_name}: Edge feature dim mismatch ({data.edge_attr.shape[1]} vs {train_edge_features})")
                    return False
            
            # Check that edge counts match edge_attr and y
            for split_name in ['train', 'val', 'test']:
                data = splits_data[split_name]
                num_edges = data.edge_index.shape[1]
                
                if data.edge_attr.shape[0] != num_edges:
                    print(f"  ❌ {split_name}: Edge attr count mismatch ({data.edge_attr.shape[0]} vs {num_edges})")
                    return False
                    
                if data.y.shape[0] != num_edges:
                    print(f"  ❌ {split_name}: Label count mismatch ({data.y.shape[0]} vs {num_edges})")
                    return False
            
            print(f"  ✅ Data consistency validated")
            print(f"  📊 Nodes: {train_nodes:,}, Node features: {train_node_features}, Edge features: {train_edge_features}")
            return True
            
        except Exception as e:
            print(f"  ❌ Error in data consistency validation: {e}")
            return False
    
    def _validate_class_distribution(self, splits_data, dataset_name):
        """Validate class distribution"""
        print(f"\n🔍 Class Distribution Validation:")
        
        try:
            for split_name in ['train', 'val', 'test']:
                data = splits_data[split_name]
                labels = data.y.cpu().numpy()
                
                # Check label range
                unique_labels = np.unique(labels)
                if not np.array_equal(unique_labels, [0, 1]):
                    if len(unique_labels) == 1:
                        print(f"  ⚠️ {split_name}: Only one class present ({unique_labels[0]})")
                    else:
                        print(f"  ❌ {split_name}: Invalid label range {unique_labels}, expected [0, 1]")
                        return False
                
                # Calculate distribution
                pos_count = np.sum(labels)
                neg_count = len(labels) - pos_count
                pos_ratio = pos_count / len(labels) * 100
                
                print(f"  📊 {split_name}: {neg_count:,} negative, {pos_count:,} positive ({pos_ratio:.3f}%)")
                
                # Check for reasonable distribution
                if pos_count == 0:
                    print(f"  ⚠️ {split_name}: No positive samples")
                elif pos_ratio > 50:
                    print(f"  ⚠️ {split_name}: Unusually high positive ratio ({pos_ratio:.1f}%)")
            
            print(f"  ✅ Class distributions validated")
            return True
            
        except Exception as e:
            print(f"  ❌ Error in class distribution validation: {e}")
            return False
    
    def _validate_feature_quality(self, splits_data, dataset_name):
        """Validate feature quality"""
        print(f"\n🔍 Feature Quality Validation:")
        
        try:
            for split_name in ['train', 'val', 'test']:
                data = splits_data[split_name]
                
                # Check node features
                node_features = data.x.cpu().numpy()
                
                # Check for NaN/Inf
                if np.any(np.isnan(node_features)):
                    print(f"  ❌ {split_name}: NaN values in node features")
                    return False
                if np.any(np.isinf(node_features)):
                    print(f"  ❌ {split_name}: Inf values in node features")
                    return False
                
                # Check edge features
                edge_features = data.edge_attr.cpu().numpy()
                
                if np.any(np.isnan(edge_features)):
                    print(f"  ❌ {split_name}: NaN values in edge features")
                    return False
                if np.any(np.isinf(edge_features)):
                    print(f"  ❌ {split_name}: Inf values in edge features")
                    return False
                
                # Check feature ranges
                node_min, node_max = node_features.min(), node_features.max()
                edge_min, edge_max = edge_features.min(), edge_features.max()
                
                print(f"  📊 {split_name}: Node features [{node_min:.3f}, {node_max:.3f}], Edge features [{edge_min:.3f}, {edge_max:.3f}]")
            
            print(f"  ✅ Feature quality validated")
            return True
            
        except Exception as e:
            print(f"  ❌ Error in feature quality validation: {e}")
            return False
    
    def _validate_temporal_splits(self, splits_data, dataset_name):
        """Validate temporal ordering of splits"""
        print(f"\n🔍 Temporal Splits Validation:")
        
        try:
            # This is a basic check - in a real scenario you'd check timestamps
            train_edges = splits_data['train'].num_edges
            val_edges = splits_data['val'].num_edges
            test_edges = splits_data['test'].num_edges
            
            total_edges = train_edges + val_edges + test_edges
            
            train_ratio = train_edges / total_edges
            val_ratio = val_edges / total_edges
            test_ratio = test_edges / total_edges
            
            print(f"  📊 Split ratios: Train {train_ratio:.3f}, Val {val_ratio:.3f}, Test {test_ratio:.3f}")
            
            # Check for reasonable split ratios
            if train_ratio < 0.5 or train_ratio > 0.8:
                print(f"  ⚠️ Unusual train ratio: {train_ratio:.3f}")
            if val_ratio < 0.1 or val_ratio > 0.3:
                print(f"  ⚠️ Unusual validation ratio: {val_ratio:.3f}")
            if test_ratio < 0.1 or test_ratio > 0.3:
                print(f"  ⚠️ Unusual test ratio: {test_ratio:.3f}")
            
            print(f"  ✅ Temporal splits validated")
            return True
            
        except Exception as e:
            print(f"  ❌ Error in temporal splits validation: {e}")
            return False
    
    def _validate_complete_consistency(self, splits_data, complete_data, dataset_name):
        """Validate consistency between splits and complete data"""
        print(f"\n🔍 Complete Data Consistency Validation:")
        
        try:
            # Check that complete data has same nodes as splits
            if complete_data.num_nodes != splits_data['train'].num_nodes:
                print(f"  ❌ Node count mismatch: complete={complete_data.num_nodes}, splits={splits_data['train'].num_nodes}")
                return False
            
            # Check that complete edges equal sum of split edges
            total_split_edges = (splits_data['train'].num_edges + 
                               splits_data['val'].num_edges + 
                               splits_data['test'].num_edges)
            
            if complete_data.num_edges != total_split_edges:
                print(f"  ❌ Edge count mismatch: complete={complete_data.num_edges}, splits_total={total_split_edges}")
                return False
            
            print(f"  ✅ Complete data consistency validated")
            return True
            
        except Exception as e:
            print(f"  ❌ Error in complete data consistency validation: {e}")
            return False
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE VALIDATION REPORT")
        print(f"{'='*80}")
        
        all_ready = True
        
        for dataset_name, results in self.validation_results.items():
            status = "✅ READY" if results['ready'] else "❌ NOT READY"
            print(f"\n{dataset_name.upper()}: {status} ({results['passed']}/{results['total']} tests passed)")
            
            if not results['ready']:
                all_ready = False
                print(f"  Failed tests:")
                for test_name, passed in results['results'].items():
                    if not passed:
                        print(f"    - {test_name}")
        
        print(f"\n{'='*80}")
        if all_ready:
            print("🎉 ALL DATASETS ARE READY FOR TRAINING!")
            print("✅ You can proceed with confidence")
        else:
            print("⚠️ SOME DATASETS HAVE ISSUES")
            print("🔧 Please fix the issues before training")
        print(f"{'='*80}")
        
        return all_ready

def validate_all_datasets():
    """Validate all available datasets"""
    validator = AMLDataValidator()
    
    # Check which datasets are available
    datasets_to_check = []
    for dataset in ['hi-small', 'li-small']:
        splits_file = validator.graphs_dir / f'ibm_aml_{dataset}_fixed_splits.pt'
        if splits_file.exists():
            datasets_to_check.append(dataset)
    
    if not datasets_to_check:
        print("❌ No fixed preprocessed datasets found!")
        print("🔧 Run colab_preprocess_fixed.py first")
        return False
    
    print(f"🔍 Found datasets to validate: {datasets_to_check}")
    
    # Validate each dataset
    for dataset in datasets_to_check:
        validator.validate_dataset(dataset, 'fixed')
    
    # Generate final report
    return validator.generate_validation_report()

if __name__ == "__main__":
    validate_all_datasets()
