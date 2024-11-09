import pandas as pd
import numpy as np
from cryptography.fernet import Fernet

def anonymize_data(data):
    """
    Anonymizes sensitive data by removing or obfuscating personally identifiable information (PII).
    """
    anonymized_data = data.copy()
    if 'name' in anonymized_data.columns:
        anonymized_data['name'] = anonymized_data['name'].apply(lambda x: 'ANONYMIZED')
    if 'email' in anonymized_data.columns:
        anonymized_data['email'] = anonymized_data['email'].apply(lambda x: 'ANONYMIZED')
    return anonymized_data

def implement_access_control(data, authorized_users):
    """
    Implements access control measures to ensure that only authorized personnel have access to sensitive data.
    """
    def check_access(user):
        if user in authorized_users:
            return True
        else:
            return False
    return check_access

def encrypt_data(data, key):
    """
    Encrypts sensitive data using the provided encryption key.
    """
    fernet = Fernet(key)
    encrypted_data = data.applymap(lambda x: fernet.encrypt(x.encode()).decode() if isinstance(x, str) else x)
    return encrypted_data

def secure_data_storage(data, storage_path):
    """
    Stores sensitive data in a secure location.
    """
    data.to_csv(storage_path, index=False)
    print(f"Data securely stored at {storage_path}")

# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
        'age': [25, 30, 35]
    })

    # Anonymize data
    anonymized_data = anonymize_data(data)
    print("Anonymized Data:")
    print(anonymized_data)

    # Implement access control
    authorized_users = ['admin', 'data_scientist']
    check_access = implement_access_control(data, authorized_users)
    print("Access Control Check for 'admin':", check_access('admin'))
    print("Access Control Check for 'guest':", check_access('guest'))

    # Encrypt data
    key = Fernet.generate_key()
    encrypted_data = encrypt_data(data, key)
    print("Encrypted Data:")
    print(encrypted_data)

    # Secure data storage
    storage_path = 'secure_data.csv'
    secure_data_storage(encrypted_data, storage_path)
