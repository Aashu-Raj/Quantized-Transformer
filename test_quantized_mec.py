#!/usr/bin/env python3
"""
Quick test script for the quantized MEC transformer
This script performs a small-scale test to verify everything works correctly
"""

import torch
import numpy as np
import sys
import os

# Add path to access models
sys.path.append('../models')

from QuantizedMECTransformer import QuantizedMemoryDNN
from config_mec import MECTransformerConfig
from ResourceAllocationtf import Algo1_NUM_VariableEnergy

def test_quantized_transformer():
    """Test the quantized transformer with a small example"""
    print("Testing Quantized MEC Transformer")
    print("=" * 50)
    
    # Test parameters
    N = 10  # Number of users
    net = [N * 3, N]  # Network architecture
    
    # Create quantization config
    config = MECTransformerConfig(
        ptf=True,  # Enable Power-of-Two Factor
        lis=True,  # Enable Log-Int-Softmax
        quant_method='minmax',
        bits_w=8,
        bits_a=8,
        bits_s=4
    )
    
    print(f"Configuration:")
    print(f"  - PTF enabled: {config.INT_NORM}")
    print(f"  - LIS enabled: {config.INT_SOFTMAX}")
    print(f"  - Weight bits: {config.BIT_TYPE_W.bits}")
    print(f"  - Activation bits: {config.BIT_TYPE_A.bits}")
    print(f"  - Attention bits: {config.BIT_TYPE_S.bits}")
    
    # Initialize quantized model
    print(f"\nInitializing quantized model...")
    mem = QuantizedMemoryDNN(
        net=net,
        max_users=N,
        learning_rate=0.01,
        training_interval=5,
        batch_size=32,
        memory_size=100,
        quant=True,
        calibrate=False,
        cfg=config
    )
    
    # Generate test calibration data
    print(f"Generating calibration data...")
    calibration_data = []
    for i in range(20):
        # Random channel gains, queue states, and energy states
        h_test = np.random.exponential(1e10, N)
        Q_test = np.random.exponential(1, N)
        Y_test = np.random.exponential(0.1, N) * 1000
        
        # Create input vector
        input_vec = np.hstack([h_test, Q_test, Y_test])
        calibration_data.append(input_vec)
    
    # Perform calibration
    print(f"Performing quantization calibration...")
    mem.start_calibration(calibration_data)
    
    # Test forward pass
    print(f"Testing forward pass...")
    test_input = calibration_data[0]
    
    # Test decode function
    m_pred, m_list = mem.decode(test_input, K=5)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {m_pred.shape}")
    print(f"Number of modes generated: {len(m_list)}")
    print(f"Mode prediction: {m_pred[:5]}")  # Show first 5 users
    
    # Test with resource allocation
    print(f"\nTesting with resource allocation algorithm...")
    
    # Generate test scenario
    h_curr = np.random.exponential(1e10, N)
    Q_curr = np.random.exponential(1, N)
    Y_curr = np.random.exponential(0.1, N) * 1000
    w = np.ones(N)
    V = 20
    
    # Test the mode with resource allocation
    mode = m_list[0][:N]  # Get first N elements for current users
    
    try:
        result = Algo1_NUM_VariableEnergy(mode, h_curr, w, Q_curr, Y_curr, N, V)
        print(f"Resource allocation result:")
        print(f"  - Objective value: {result[0]:.4f}")
        print(f"  - Average rate: {np.mean(result[1]):.4f}")
        print(f"  - Average energy: {np.mean(result[2]):.4f}")
    except Exception as e:
        print(f"Resource allocation test failed: {e}")
    
    # Test training
    print(f"\nTesting training capability...")
    for i in range(10):
        # Random input and target
        x = np.random.randn(net[0])
        y = np.random.randint(0, 2, (N, 2)).astype(float)  # Binary decisions
        
        mem.encode(x, y)
    
    print(f"Training completed. Cost history length: {len(mem.cost_his)}")
    
    # Test model save/load
    print(f"\nTesting model save/load...")
    temp_path = "temp_quantized_model.pth"
    
    try:
        mem.save_model(temp_path)
        print(f"Model saved successfully")
        
        # Create new instance and load
        mem2 = QuantizedMemoryDNN(
            net=net, max_users=N, quant=True, cfg=config
        )
        mem2.load_model(temp_path)
        print(f"Model loaded successfully")
        
        # Test loaded model
        m_pred2, m_list2 = mem2.decode(test_input, K=5)
        print(f"Loaded model output matches: {np.allclose(m_pred, m_pred2, atol=1e-3)}")
        
        # Clean up
        os.remove(temp_path)
        
    except Exception as e:
        print(f"Save/load test failed: {e}")
    
    print(f"\n" + "=" * 50)
    print("Quantized MEC Transformer test completed successfully!")
    print("=" * 50)
    
    return True

def test_different_configurations():
    """Test different quantization configurations"""
    print("\nTesting Different Quantization Configurations")
    print("=" * 60)
    
    configs = [
        ("Full Precision (No Quantization)", None),
        ("8-bit with PTF+LIS", MECTransformerConfig(ptf=True, lis=True, bits_w=8, bits_a=8, bits_s=4)),
        ("4-bit Aggressive", MECTransformerConfig(ptf=True, lis=True, bits_w=4, bits_a=4, bits_s=4)),
        ("8-bit No PTF/LIS", MECTransformerConfig(ptf=False, lis=False, bits_w=8, bits_a=8, bits_s=8)),
    ]
    
    N = 5  # Smaller for quick test
    test_input = np.random.randn(N * 3)
    
    results = {}
    
    for name, config in configs:
        print(f"\nTesting: {name}")
        
        try:
            if config is None:
                # Test without quantization
                mem = QuantizedMemoryDNN(
                    net=[N * 3, N], max_users=N, quant=False, cfg=MECTransformerConfig()
                )
            else:
                mem = QuantizedMemoryDNN(
                    net=[N * 3, N], max_users=N, quant=True, cfg=config
                )
                
                # Quick calibration
                calib_data = [np.random.randn(N * 3) for _ in range(10)]
                mem.start_calibration(calib_data)
            
            # Test inference
            start_time = time.time()
            m_pred, m_list = mem.decode(test_input, K=3)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            results[name] = {
                'output': m_pred,
                'inference_time': inference_time,
                'success': True
            }
            
            print(f"  ✓ Success - Inference time: {inference_time:.2f}ms")
            
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
            print(f"  ✗ Failed: {e}")
    
    # Compare results
    print(f"\n" + "=" * 60)
    print("CONFIGURATION COMPARISON")
    print("=" * 60)
    
    for name, result in results.items():
        if result['success']:
            print(f"{name:30s}: {result['inference_time']:6.2f}ms")
        else:
            print(f"{name:30s}: FAILED")
    
    print("=" * 60)

if __name__ == "__main__":
    import time
    
    print("Quantized MEC Transformer Test Suite")
    print("=" * 80)
    
    # Set device info
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Running on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    try:
        # Run basic test
        test_quantized_transformer()
        
        # Run configuration comparison
        test_different_configurations()
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nTest suite completed!") 