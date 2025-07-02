import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import math
import pandas as pd

# Set random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

from QuantizedMECTransformer import QuantizedMemoryDNN
from config_mec import MECTransformerConfig
from ResourceAllocationtf import Algo1_NUM_VariableEnergy

# === Create directory for CSV and plots ===
csv_dir = "frame_logs_quantized_transformer"
os.makedirs(csv_dir, exist_ok=True)

# === User Count Generation ===
def generate_user_count_sequence(n, min_users, max_users, max_changes):
    change_points = sorted(random.sample(range(1, n), min(max_changes, n - 1)))
    change_points = [0] + change_points + [n]
    user_counts = []
    for i in range(len(change_points) - 1):
        user_value = random.randint(min_users, max_users)
        user_counts.extend([user_value] * (change_points[i + 1] - change_points[i]))
    return np.array(user_counts)

def load_or_create_user_counts(n, min_users, max_users, max_changes, filename="user_count_list_quantized.npy"):
    if os.path.exists(filename):
        user_counts = np.load(filename)
        print(f"Loaded existing user count sequence from {filename}")
    else:
        user_counts = generate_user_count_sequence(n, min_users, max_users, max_changes)
        np.save(filename, user_counts)
        print(f"Generated new user count sequence and saved to {filename}")
    return user_counts

# === Rician Fading Channel Model ===
def racian_mec(h, factor):
    n = len(h)
    beta = np.sqrt(h * factor)
    sigma = np.sqrt(h * (1 - factor) / 2)
    x = sigma * np.random.randn(n) + beta
    y = sigma * np.random.randn(n)
    return np.power(x, 2) + np.power(y, 2)

# === Save array as CSV with optional header ===
def save_array_as_csv(arr, filename, header=None):
    path = os.path.join(csv_dir, filename)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    np.savetxt(path, arr, delimiter=",", header=",".join(header) if header else "", comments='')

# === Plot with rolling mean and save ===
def plot_array(data_array, rolling_intv=50, ylabel='Value', save_path=None):
    df = pd.DataFrame(data_array)
    plt.style.use('default')  # Use default style for compatibility
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.plot(
        np.arange(len(data_array)) + 1,
        np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values),
        linewidth=2,
        label='Rolling Mean'
    )
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close()

def run_quantized_simulation(quantization_config):
    """Run MEC simulation with quantized transformer"""
    
    print("="*60)
    print("QUANTIZED MEC TRANSFORMER SIMULATION")
    print("="*60)
    print(f"Quantization Config:")
    print(f"  - PTF (Power-of-Two Factor): {quantization_config.INT_NORM}")
    print(f"  - LIS (Log-Int-Softmax): {quantization_config.INT_SOFTMAX}")
    print(f"  - Weight bits: {quantization_config.BIT_TYPE_W.bits}")
    print(f"  - Activation bits: {quantization_config.BIT_TYPE_A.bits}")
    print(f"  - Attention bits: {quantization_config.BIT_TYPE_S.bits}")
    print("="*60)
    
    # === Simulation Parameters ===
    n = 10000  # Number of time frames
    min_users = 5
    N = 30  # Maximum users
    max_changes = 20
    lambda_param = 3
    nu = 1000
    Delta = 32
    V = 20
    Memory_capacity = 1024
    decoder_mode = 'OP'
    CHFACT = 10**10
    
    # Load or generate user count sequence
    user_count_list = load_or_create_user_counts(n, min_users, N, max_changes)
    
    # Initialize state arrays
    channel = np.zeros((n, N))
    dataA = np.zeros((n, N))
    Q = np.zeros((n, N))
    Y = np.zeros((n, N))
    Obj = np.zeros(n)
    energy_arr = np.zeros((n, N))
    rate_arr = np.zeros((n, N))
    
    # Thresholds and weights
    energy_thresh = np.ones(N) * 0.08
    w = np.array([1.5 if i % 2 == 0 else 1 for i in range(N)])
    arrival_lambda = lambda_param * np.ones(N)
    
    # Channel pathloss calculation
    dist_v = np.linspace(start=120, stop=255, num=N)
    Ad, fc = 3, 915e6
    loss_exponent = 3
    light = 3e8
    h0 = np.array([Ad * (light / (4 * math.pi * fc * dist_v[j])) ** loss_exponent for j in range(N)])
    
    # Initialize Quantized MemoryDNN
    print("\nInitializing Quantized Transformer MemoryDNN...")
    mem = QuantizedMemoryDNN(
        net=[N * 3, N], 
        max_users=N, 
        learning_rate=0.001,
        training_interval=20, 
        batch_size=128, 
        memory_size=Memory_capacity,
        quant=True,  # Enable quantization
        calibrate=False,  # Start without calibration
        cfg=quantization_config
    )
    
    # Collect calibration data
    print("Collecting calibration data...")
    calibration_data = []
    calibration_frames = min(100, n // 10)  # Use 10% of frames for calibration
    
    for i in range(calibration_frames):
        current_N = user_count_list[i]
        h_tmp = racian_mec(h0[:current_N], 0.3)
        h_curr = h_tmp * CHFACT
        
        # Simple initialization for calibration
        Q_cal = np.random.exponential(lambda_param, current_N)
        Y_cal = np.random.exponential(0.1, current_N) * nu
        
        nn_input_raw = np.hstack([
            h_curr,
            Q_cal / 10000,
            Y_cal / 10000
        ])
        
        # Pad input vector to fixed size
        if len(nn_input_raw) < mem.net[0]:
            nn_input_raw = np.hstack([nn_input_raw, np.zeros(mem.net[0] - len(nn_input_raw))])
        elif len(nn_input_raw) > mem.net[0]:
            nn_input_raw = nn_input_raw[:mem.net[0]]
        
        calibration_data.append(nn_input_raw)
    
    # Perform quantization calibration
    mem.start_calibration(calibration_data)
    
    # Main simulation loop
    mode_his, k_idx_his = [], []
    start_time = time.time()
    
    print(f"\nStarting main simulation with {n} time frames...")
    print("Progress: ", end="")
    
    for i in range(n):
        current_N = user_count_list[i]
        
        if i % (n // 10) == 0:
            progress = 100 * i / n
            print(f"{progress:.0f}%...", end="", flush=True)
        
        # Compute K (number of users to select) safely
        if i > 0 and i % Delta == 0:
            if Delta > 1 and len(k_idx_his) >= Delta:
                recent_k = np.array(k_idx_his[-Delta:]) % current_N
                max_k = recent_k.max() + 1
            else:
                max_k = 1
        else:
            max_k = k_idx_his[-1] + 1 if k_idx_his else 1
        
        K = min(max_k, current_N)
        
        # Generate current channel gains and data arrivals
        h_tmp = racian_mec(h0[:current_N], 0.3)
        h_curr = h_tmp * CHFACT
        channel[i, :current_N] = h_curr
        dataA[i, :current_N] = np.random.exponential(arrival_lambda[:current_N])
        
        # Update queues and energy state
        if i > 0:
            Q[i, :current_N] = Q[i - 1, :current_N] + dataA[i - 1, :current_N] - rate_arr[i - 1, :current_N]
            Q[i, :current_N] = np.maximum(Q[i, :current_N], 0)
            Y[i, :current_N] = np.maximum(
                Y[i - 1, :current_N] + (energy_arr[i - 1, :current_N] - energy_thresh[:current_N]) * nu, 0)
        
        # Prepare input vector for quantized NN
        nn_input_raw = np.hstack([
            h_curr,
            Q[i, :current_N] / 10000,
            Y[i, :current_N] / 10000
        ])
        
        # Pad input vector to fixed size
        if len(nn_input_raw) < mem.net[0]:
            nn_input_raw = np.hstack([nn_input_raw, np.zeros(mem.net[0] - len(nn_input_raw))])
        elif len(nn_input_raw) > mem.net[0]:
            nn_input_raw = nn_input_raw[:mem.net[0]]
        
        # Decode modes with Quantized MemoryDNN Transformer
        m_pred, m_list = mem.decode(nn_input_raw, K, decoder_mode)
        
        # Evaluate different modes
        r_list, v_list = [], []
        for m in m_list:
            m_slice = np.array(m[:current_N])
            w_slice = w[:current_N]
            Q_slice = Q[i, :current_N]
            Y_slice = Y[i, :current_N]
            
            res = Algo1_NUM_VariableEnergy(m_slice, h_curr, w_slice, Q_slice, Y_slice, current_N, V)
            r_list.append(res)
            v_list.append(res[0])
        
        # Select best mode and store experience
        best_idx = np.argmax(v_list)
        k_idx_his.append(best_idx)
        mem.encode(nn_input_raw, m_list[best_idx])
        mode_his.append(m_list[best_idx])
        
        # Record results
        Obj[i] = r_list[best_idx][0]
        rate_arr[i, :current_N] = r_list[best_idx][1]
        energy_arr[i, :current_N] = r_list[best_idx][2]
    
    print("100%")
    total_time = time.time() - start_time
    print(f"\nSimulation completed in {total_time:.2f} seconds")
    
    # Plot training cost
    mem.plot_cost()
    
    # Save results
    print("\nSaving simulation results...")
    user_headers = [f'user_{i+1}' for i in range(N)]
    
    save_array_as_csv(channel / CHFACT, "channel_quantized.csv", header=user_headers)
    save_array_as_csv(dataA, "data_arrival_quantized.csv", header=user_headers)
    save_array_as_csv(Q, "data_queue_quantized.csv", header=user_headers)
    save_array_as_csv(Y, "energy_queue_quantized.csv", header=user_headers)
    save_array_as_csv(rate_arr, "rate_quantized.csv", header=user_headers)
    save_array_as_csv(energy_arr, "energy_consumption_quantized.csv", header=user_headers)
    save_array_as_csv(Obj, "objective_quantized.csv", header=["objective"])
    save_array_as_csv(user_count_list, "user_count_list_quantized.csv", header=["user_count"])
    
    # Computation rate per frame
    comp_rate = np.sum(rate_arr, axis=1)
    save_array_as_csv(comp_rate, "computational_rate_quantized.csv", header=["comp_rate"])
    
    # Save plots
    print("Generating and saving plots...")
    plot_array(np.sum(Q, axis=1) / user_count_list, rolling_intv=50,
               ylabel='Average Data Queue (Quantized)',
               save_path=os.path.join(csv_dir, "avg_queue_plot_quantized.png"))
    
    plot_array(np.sum(energy_arr, axis=1) / user_count_list, rolling_intv=50,
               ylabel='Average Energy Consumption (Quantized)',
               save_path=os.path.join(csv_dir, "avg_energy_plot_quantized.png"))
    
    plot_array(comp_rate, rolling_intv=50,
               ylabel='Computational Rate (Quantized)',
               save_path=os.path.join(csv_dir, "computational_rate_plot_quantized.png"))
    
    # Save model checkpoint
    model_path = os.path.join(csv_dir, "quantized_mec_transformer.pth")
    mem.save_model(model_path)
    print(f"Quantized model saved to {model_path}")
    
    # Save as .mat file
    mat_filename = "result_quantized_transformer_dynamic_users.mat"
    sio.savemat(mat_filename, {
        'input_h': channel / CHFACT,
        'data_arrival': dataA,
        'data_queue': Q,
        'energy_queue': Y,
        'off_mode': mode_his,
        'rate': rate_arr,
        'energy_consumption': energy_arr,
        'data_rate': rate_arr,
        'objective': Obj,
        'user_count_list': user_count_list,
        'quantization_config': {
            'ptf_enabled': quantization_config.INT_NORM,
            'lis_enabled': quantization_config.INT_SOFTMAX,
            'weight_bits': quantization_config.BIT_TYPE_W.bits,
            'activation_bits': quantization_config.BIT_TYPE_A.bits,
            'attention_bits': quantization_config.BIT_TYPE_S.bits
        }
    })
    print(f"Results saved to {mat_filename}")
    
    # Performance summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Total time frames: {n}")
    print(f"Average objective value: {np.mean(Obj):.4f}")
    print(f"Average computational rate: {np.mean(comp_rate):.4f}")
    print(f"Average energy consumption: {np.mean(np.sum(energy_arr, axis=1)):.4f}")
    print(f"Average queue length: {np.mean(np.sum(Q, axis=1)):.4f}")
    print(f"Simulation time: {total_time:.2f} seconds")
    print("="*60)
    
    return {
        'objective': Obj,
        'computational_rate': comp_rate,
        'energy_consumption': energy_arr,
        'queue_length': Q,
        'simulation_time': total_time
    }

if __name__ == "__main__":
    print("Quantized MEC Transformer Simulation")
    print("====================================")
    
    # Test different quantization configurations
    configs_to_test = [
        {
            'name': '8-bit with PTF+LIS',
            'config': MECTransformerConfig(ptf=True, lis=True, quant_method='minmax', 
                                         bits_w=8, bits_a=8, bits_s=4)
        },
        {
            'name': '4-bit aggressive quantization',
            'config': MECTransformerConfig(ptf=True, lis=True, quant_method='omse', 
                                         bits_w=4, bits_a=4, bits_s=4)
        }
    ]
    
    results = {}
    
    for config_info in configs_to_test:
        print(f"\n{'='*80}")
        print(f"RUNNING CONFIGURATION: {config_info['name']}")
        print(f"{'='*80}")
        
        result = run_quantized_simulation(config_info['config'])
        results[config_info['name']] = result
        
        print(f"\nCompleted configuration: {config_info['name']}")
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON OF QUANTIZATION CONFIGURATIONS")
    print(f"{'='*80}")
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Average Objective: {np.mean(result['objective']):.4f}")
        print(f"  Average Comp Rate: {np.mean(result['computational_rate']):.4f}")
        print(f"  Simulation Time: {result['simulation_time']:.2f}s")
    
    print(f"\n{'='*80}")
    print("Quantized MEC simulation completed successfully!")
    print("Check the 'frame_logs_quantized_transformer' directory for detailed results.")
    print(f"{'='*80}") 