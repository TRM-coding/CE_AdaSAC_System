#!/usr/bin/env python3
"""
ASTO算法日志数据分析和可视化脚本
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Any

def plot_fitness_evolution():
    """
    绘制适应度随迭代次数的变化图
    """
    try:
        # 读取热身阶段数据
        with open('./warm_asto_log.pkl', 'rb') as f:
            warm_log = pickle.load(f)
        
        # 提取热身阶段的适应度数据
        warm_epochs = []
        warm_best_fitness = []
        warm_avg_fitness = []
        warm_times = []
        
        for iter_data in warm_log['iterations']:
            warm_epochs.append(iter_data['epoch'])
            warm_best_fitness.append(iter_data['best_fitness'])
            warm_avg_fitness.append(iter_data['avg_fitness'])
            warm_times.append(iter_data['time'])
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ASTO Algorithm Performance Analysis', fontsize=16)
        
        # 1. 热身阶段适应度变化
        axes[0, 0].plot(warm_epochs, warm_best_fitness, 'b-', label='Best Fitness', marker='o')
        axes[0, 0].plot(warm_epochs, warm_avg_fitness, 'r--', label='Average Fitness', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].set_title('Warm Phase: Fitness Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 热身阶段时间分析
        axes[0, 1].bar(warm_epochs, warm_times, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_title('Warm Phase: Time per Iteration')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 生成阶段不同alpha值的比较
        alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        alpha_final_fitness = []
        alpha_total_times = []
        
        for alpha in alpha_values:
            try:
                with open(f'./asto_v2_alpha_{alpha:.1f}_log.pkl', 'rb') as f:
                    v2_log = pickle.load(f)
                
                if v2_log['iterations']:
                    final_fitness = v2_log['iterations'][-1]['best_fitness']
                    total_time = v2_log['total_time']
                    alpha_final_fitness.append(final_fitness)
                    alpha_total_times.append(total_time)
                else:
                    alpha_final_fitness.append(0)
                    alpha_total_times.append(0)
                    
            except FileNotFoundError:
                alpha_final_fitness.append(0)
                alpha_total_times.append(0)
        
        # 过滤掉没有数据的alpha值
        valid_indices = [i for i, fitness in enumerate(alpha_final_fitness) if fitness > 0]
        valid_alphas = [alpha_values[i] for i in valid_indices]
        valid_fitness = [alpha_final_fitness[i] for i in valid_indices]
        valid_times = [alpha_total_times[i] for i in valid_indices]
        
        if valid_alphas:
            axes[1, 0].bar(valid_alphas, valid_fitness, alpha=0.7, color='purple')
            axes[1, 0].set_xlabel('Alpha Value')
            axes[1, 0].set_ylabel('Final Best Fitness')
            axes[1, 0].set_title('Generate Phase: Final Fitness by Alpha')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 生成阶段时间比较
            axes[1, 1].bar(valid_alphas, valid_times, alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('Alpha Value')
            axes[1, 1].set_ylabel('Total Time (seconds)')
            axes[1, 1].set_title('Generate Phase: Total Time by Alpha')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No generate phase data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, 'No generate phase data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig('./asto_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Analysis plot saved as: ./asto_analysis.png")
        
    except FileNotFoundError:
        print("Log files not found. Please run the ASTO algorithm first.")
    except Exception as e:
        print(f"Error creating plots: {e}")

def generate_summary_report():
    """
    生成ASTO算法运行的汇总报告
    """
    try:
        # 读取完整日志
        with open('./asto_complete_log.pkl', 'rb') as f:
            complete_log = pickle.load(f)
        
        # 读取热身阶段详细数据
        with open('./warm_asto_log.pkl', 'rb') as f:
            warm_log = pickle.load(f)
        
        # 生成报告
        report = []
        report.append("ASTO Algorithm Execution Report")
        report.append("=" * 50)
        report.append(f"Execution Time: {complete_log['timestamp']}")
        report.append(f"Total Runtime: {complete_log['total_time']:.3f} seconds")
        report.append("")
        
        # 参数信息
        params = complete_log['parameters']
        report.append("Algorithm Parameters:")
        report.append(f"  - Warm epochs: {params['warm_epochs']}")
        report.append(f"  - Generate epochs: {params['generate_epochs']}")
        report.append("")
        
        # 热身阶段分析
        report.append("Warm Phase Analysis:")
        report.append(f"  - Total time: {warm_log['total_time']:.3f}s")
        report.append(f"  - Number of iterations: {len(warm_log['iterations'])}")
        report.append(f"  - Average time per iteration: {warm_log['total_time']/len(warm_log['iterations']):.3f}s")
        
        # 适应度进展
        best_fitness_values = [iter_data['best_fitness'] for iter_data in warm_log['iterations']]
        avg_fitness_values = [iter_data['avg_fitness'] for iter_data in warm_log['iterations']]
        
        report.append(f"  - Initial best fitness: {best_fitness_values[0]:.4f}")
        report.append(f"  - Final best fitness: {best_fitness_values[-1]:.4f}")
        report.append(f"  - Fitness improvement: {best_fitness_values[-1] - best_fitness_values[0]:.4f}")
        report.append(f"  - Average final fitness: {avg_fitness_values[-1]:.4f}")
        report.append("")
        
        # 生成阶段分析
        report.append("Generate Phase Analysis:")
        alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        generate_results = complete_log.get('generate_results', {})
        total_generate_time = 0
        successful_alphas = 0
        
        for alpha in alpha_values:
            try:
                with open(f'./asto_v2_alpha_{alpha:.1f}_log.pkl', 'rb') as f:
                    v2_log = pickle.load(f)
                
                report.append(f"  Alpha {alpha:.1f}:")
                report.append(f"    - Total time: {v2_log['total_time']:.3f}s")
                report.append(f"    - Iterations: {len(v2_log['iterations'])}")
                
                if v2_log['iterations']:
                    final_best = v2_log['iterations'][-1]['best_fitness']
                    initial_best = v2_log['iterations'][0]['best_fitness']
                    improvement = final_best - initial_best
                    
                    report.append(f"    - Final best fitness: {final_best:.4f}")
                    report.append(f"    - Fitness improvement: {improvement:.4f}")
                    
                    # 最佳解决方案
                    if alpha in generate_results:
                        best_solution = generate_results[alpha].get('best_solution', 'N/A')
                        report.append(f"    - Best solution: {best_solution}")
                
                total_generate_time += v2_log['total_time']
                successful_alphas += 1
                
            except FileNotFoundError:
                report.append(f"  Alpha {alpha:.1f}: No data available")
        
        report.append("")
        report.append(f"Generate Phase Summary:")
        report.append(f"  - Total generate time: {total_generate_time:.3f}s")
        report.append(f"  - Successful alpha values: {successful_alphas}/{len(alpha_values)}")
        report.append(f"  - Average time per alpha: {total_generate_time/max(successful_alphas, 1):.3f}s")
        
        # 整体性能
        report.append("")
        report.append("Overall Performance:")
        warm_time = warm_log['total_time']
        total_time = complete_log['total_time']
        report.append(f"  - Warm phase time: {warm_time:.3f}s ({warm_time/total_time*100:.1f}%)")
        report.append(f"  - Generate phase time: {total_generate_time:.3f}s ({total_generate_time/total_time*100:.1f}%)")
        report.append(f"  - Other time: {total_time - warm_time - total_generate_time:.3f}s")
        
        # 保存报告
        with open('./asto_summary_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # 打印报告
        print('\n'.join(report))
        
        print(f"\nSummary report saved to: ./asto_summary_report.txt")
        
    except FileNotFoundError:
        print("Log files not found. Please run the ASTO algorithm first.")
    except Exception as e:
        print(f"Error generating report: {e}")

def main():
    """
    主函数：执行所有分析任务
    """
    print("ASTO Algorithm Log Analysis")
    print("=" * 50)
    
    # 检查日志文件是否存在
    required_files = ['./asto_complete_log.pkl', './warm_asto_log.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing log files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run the ASTO algorithm first to generate log files.")
        return
    
    print("Found log files. Starting analysis...")
    
    # 生成汇总报告
    print("\n1. Generating summary report...")
    generate_summary_report()
    
    # 生成可视化图表
    print("\n2. Creating visualization plots...")
    try:
        plot_fitness_evolution()
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        print("Skipping visualization plots.")
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()
