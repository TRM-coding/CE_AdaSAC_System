#!/usr/bin/env python3
"""
ASTO算法日志记录演示脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from astogptJ import asto, analyze_asto_logs, save_detailed_population_history

def demo_asto_with_logging():
    """
    演示ASTO算法的完整日志记录功能
    """
    print("ASTO Algorithm Logging Demo")
    print("="*60)
    
    # 运行ASTO算法（使用较小的参数进行演示）
    print("Step 1: Running ASTO algorithm with logging...")
    complete_log = asto(warm_epoch=3, generate_epoch=2)
    
    print("\nStep 2: Analyzing the logs...")
    analyze_asto_logs()
    
    print("\nStep 3: Saving detailed population history...")
    save_detailed_population_history()
    
    print("\nStep 4: Log files created:")
    log_files = [
        './warm_asto_log.pkl',
        './asto_complete_log.pkl', 
        './warm_phase_detailed.csv'
    ]
    
    # 检查生成阶段的日志文件
    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for alpha in alpha_values:
        pkl_file = f'./asto_v2_alpha_{alpha:.1f}_log.pkl'
        csv_file = f'./generate_alpha_{alpha:.1f}_detailed.csv'
        if os.path.exists(pkl_file):
            log_files.append(pkl_file)
        if os.path.exists(csv_file):
            log_files.append(csv_file)
    
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            print(f"  ✓ {log_file} ({size} bytes)")
        else:
            print(f"  ✗ {log_file} (not found)")
    
    print("\nDemo completed!")
    return complete_log

def read_log_example():
    """
    演示如何读取和使用日志数据
    """
    import pickle
    
    print("\nReading Log Example:")
    print("-" * 30)
    
    try:
        # 读取完整日志
        with open('./asto_complete_log.pkl', 'rb') as f:
            complete_log = pickle.load(f)
        
        print("Complete Log Structure:")
        print(f"  - Algorithm: {complete_log.get('algorithm', 'N/A')}")
        print(f"  - Total time: {complete_log.get('total_time', 0):.3f}s")
        print(f"  - Parameters: {complete_log.get('parameters', {})}")
        print(f"  - Timestamp: {complete_log.get('timestamp', 'N/A')}")
        
        # 读取热身阶段详细日志
        with open('./warm_asto_log.pkl', 'rb') as f:
            warm_log = pickle.load(f)
        
        print(f"\nWarm Log Structure:")
        print(f"  - Total iterations: {len(warm_log.get('iterations', []))}")
        print(f"  - Total time: {warm_log.get('total_time', 0):.3f}s")
        
        if warm_log.get('iterations'):
            first_iter = warm_log['iterations'][0]
            print(f"  - First iteration example:")
            print(f"    * Epoch: {first_iter.get('epoch')}")
            print(f"    * Alpha: {first_iter.get('alpha')}")
            print(f"    * Time: {first_iter.get('time', 0):.3f}s")
            print(f"    * Population size: {first_iter.get('population_size')}")
            print(f"    * Best fitness: {first_iter.get('best_fitness', 0):.4f}")
        
        # 尝试读取一个生成阶段的日志
        alpha_test = 0.5
        try:
            with open(f'./asto_v2_alpha_{alpha_test:.1f}_log.pkl', 'rb') as f:
                v2_log = pickle.load(f)
            
            print(f"\nGenerate Phase Log (alpha={alpha_test}):")
            print(f"  - Alpha: {v2_log.get('alpha')}")
            print(f"  - Total iterations: {len(v2_log.get('iterations', []))}")
            print(f"  - Total time: {v2_log.get('total_time', 0):.3f}s")
            
        except FileNotFoundError:
            print(f"\nGenerate phase log for alpha={alpha_test} not found")
            
    except FileNotFoundError:
        print("Log files not found. Please run the ASTO algorithm first.")
    except Exception as e:
        print(f"Error reading logs: {e}")

if __name__ == "__main__":
    # 运行演示
    complete_log = demo_asto_with_logging()
    
    # 演示日志读取
    read_log_example()
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("You can now analyze the generated log files to understand")
    print("the complete evolution process of the ASTO algorithm.")
