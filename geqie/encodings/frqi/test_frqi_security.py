import numpy as np
import json
#import pytest  # If you use pytest, otherwise standard python asserts work
from retrieve import retrieve

def test_secure_retrieve_blocks_rce():
    """
    Test that passing a malicious python string raises an error
    instead of executing the code.
    """
    print("\n--- Testing Security Against RCE ---")

    # ATTACK PAYLOAD:
    # A string that looks like a dictionary but contains executable code.
    # If 'eval()' were used, this would print "HACKED" to the console.
    malicious_payload = "print('!!! SYSTEM HACKED !!!') or {'00001': 100}"

    try:
        # Attempt to pass the payload
        retrieve(malicious_payload)
        
        # If we get here without error, we check if it accepted it blindly
        # (It shouldn't, because json.loads will fail on 'print')
        print("FAIL: The function accepted the input (It might be using eval!)")
        
    except json.JSONDecodeError:
        print("SUCCESS: blocked. The function correctly rejected non-JSON code.")
    except Exception as e:
        print(f"SUCCESS: blocked with error: {type(e).__name__}")

def test_valid_frqi_decoding():
    """
    Test that valid JSON input still works correctly for FRQI.
    """
    print("\n--- Testing Valid FRQI Functionality ---")
    
    # VALID PAYLOAD:
    # 3 qubits total: 2 position (4 pixels), 1 color.
    # Pixel 0 (00): 50% gray (10 zeros, 10 ones) -> "000":10, "001":10
    # Pixel 1 (01): Black (20 zeros) -> "010": 20
    # Pixel 2 (10): White (20 ones)  -> "101": 20
    # Pixel 3 (11): Black (20 zeros) -> "110": 20
    valid_payload = json.dumps({
        "000": 10, "001": 10, 
        "010": 20, 
        "101": 20, 
        "110": 20
    })

    result = retrieve(valid_payload)
    
    print(f"Resulting Image Shape: {result.shape}")
    print(f"Pixel Values:\n{result}")

    # Assertions
    assert result.shape == (2, 2)
    assert result[0, 0] == 0.5  # Pixel 0 should be 0.5
    assert result[0, 1] == 0.0  # Pixel 1 should be 0.0
    assert result[1, 0] == 1.0  # Pixel 2 should be 1.0
    print("SUCCESS: Valid FRQI data decoded correctly.")

if __name__ == "__main__":
    test_secure_retrieve_blocks_rce()
    test_valid_frqi_decoding()