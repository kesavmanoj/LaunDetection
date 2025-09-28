"""
Configuration file for IBM AML Dataset preprocessing and training.
"""
import os
from pathlib import Path

class Config:
    """Configuration class for AML dataset processing"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    
    # Raw data directories (adjust paths based on your setup)
    RAW_DATA_DIR = DATA_DIR / "raw"
    
    # Processed data directories
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    GRAPHS_DIR = DATA_DIR / "graphs"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Dataset configurations
    DATASET_CONFIGS = {
        'HI-Small': {
            'accounts_file': 'HI-Small_accounts.csv',
            'transactions_file': 'HI-Small_Trans.csv',
            'patterns_file': 'HI-Small_Patterns.txt'
        },
        'HI-Medium': {
            'accounts_file': 'HI-Medium_accounts.csv',
            'transactions_file': 'HI-Medium_Trans.csv',
            'patterns_file': 'HI-Medium_Patterns.txt'
        },
        'HI-Large': {
            'accounts_file': 'HI-Large_accounts.csv',
            'transactions_file': 'HI-Large_Trans.csv',
            'patterns_file': 'HI-Large_Patterns.txt'
        },
        'LI-Small': {
            'accounts_file': 'LI-Small_accounts.csv',
            'transactions_file': 'LI-Small_Trans.csv',
            'patterns_file': 'LI-Small_Patterns.txt'
        },
        'LI-Medium': {
            'accounts_file': 'LI-Medium_accounts.csv',
            'transactions_file': 'LI-Medium_Trans.csv',
            'patterns_file': 'LI-Medium_Patterns.txt'
        },
        'LI-Large': {
            'accounts_file': 'LI-Large_accounts.csv',
            'transactions_file': 'LI-Large_Trans.csv',
            'patterns_file': 'LI-Large_Patterns.txt'
        }
    }
    
    # Processing parameters
    DEFAULT_CHUNK_SIZE = 50000
    SMALL_CHUNK_SIZE = 25000
    LARGE_CHUNK_SIZE = 100000
    
    # Data splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Feature parameters
    NODE_FEATURE_DIM = 25  # Increased from 20 for more comprehensive features
    EDGE_FEATURE_DIM = 16  # Increased from 14 for additional features
    
    # Memory management
    MAX_MEMORY_USAGE_GB = 8  # Maximum memory usage in GB
    GC_FREQUENCY = 10  # Garbage collection frequency (every N chunks)
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.GRAPHS_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def get_dataset_paths(cls, dataset_name):
        """Get file paths for a specific dataset"""
        if dataset_name not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        config = cls.DATASET_CONFIGS[dataset_name]
        return {
            'accounts': cls.RAW_DATA_DIR / config['accounts_file'],
            'transactions': cls.RAW_DATA_DIR / config['transactions_file'],
            'patterns': cls.RAW_DATA_DIR / config['patterns_file']
        }
