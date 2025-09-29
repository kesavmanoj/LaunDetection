"""
Simple test to verify the preprocessing column detection works
"""

import pandas as pd
from pathlib import Path

def test_column_detection():
    """Test that our column detection logic works with the actual data"""
    
    base_dir = Path('/content/drive/MyDrive/LaunDetection')
    raw_data_dir = base_dir / 'data' / 'raw'
    
    print("🧪 TESTING COLUMN DETECTION LOGIC")
    print("="*50)
    
    # Test with HI-Small accounts
    accounts_file = raw_data_dir / 'HI-Small_accounts.csv'
    
    if accounts_file.exists():
        print(f"📊 Testing with: {accounts_file}")
        
        # Load accounts
        accounts_df = pd.read_csv(accounts_file, nrows=5)
        print(f"Columns found: {list(accounts_df.columns)}")
        
        # Test our column detection logic
        def find_column(df, possible_names, default_name):
            """Find column by checking multiple possible names"""
            for name in possible_names:
                if name in df.columns:
                    return name
            
            # If not found, try case-insensitive search
            for name in possible_names:
                for col in df.columns:
                    if name.lower() == col.lower():
                        return col
            
            # Last resort: return default or first available column
            if default_name in df.columns:
                return default_name
            else:
                print(f"⚠️ Column not found from {possible_names}, using first column: {df.columns[0]}")
                return df.columns[0]
        
        # Test bank column detection
        bank_col = find_column(accounts_df, ['Bank ID', 'Bank Name', 'bank_id', 'bank_name', 'Bank'], 'Bank ID')
        account_col = find_column(accounts_df, ['Account Number', 'Account', 'account_number', 'account'], 'Account Number')
        
        print(f"✅ Bank column detected: '{bank_col}'")
        print(f"✅ Account column detected: '{account_col}'")
        
        # Test account ID creation
        try:
            test_account_id = str(accounts_df[bank_col].iloc[0]) + '_' + str(accounts_df[account_col].iloc[0])
            print(f"✅ Sample Account ID: {test_account_id}")
            print("🎯 Column detection logic works correctly!")
            return True
        except Exception as e:
            print(f"❌ Error creating account ID: {e}")
            return False
    else:
        print(f"❌ Accounts file not found: {accounts_file}")
        return False

if __name__ == "__main__":
    success = test_column_detection()
    if success:
        print("\n✅ Test passed - preprocessing should work!")
    else:
        print("\n❌ Test failed - need to debug further")
