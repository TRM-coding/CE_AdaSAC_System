#!/usr/bin/env python3
"""
测试性能监控功能的简单脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gptJtestsvd import PerformanceMonitor
import time
import torch

def test_performance_monitor():
    """测试性能监控器的基本功能"""
    print("🧪 测试性能监控器...")
    
    monitor = PerformanceMonitor()
    
    # 开始内存跟踪
    monitor.start_memory_tracking()
    monitor.record_memory_snapshot("测试开始")
    
    # 模拟一些操作
    for i in range(5):
        # 模拟GPU操作
        start_time = time.time()
        time.sleep(0.01)  # 模拟10ms的GPU操作
        gpu_time = time.time() - start_time
        monitor.record_cloud_time(gpu_time)
        
        # 模拟CPU操作
        start_time = time.time()
        time.sleep(0.02)  # 模拟20ms的CPU操作
        cpu_time = time.time() - start_time
        monitor.record_edge_time(cpu_time)
        
        # 模拟网络传输
        monitor.record_network_time(0.005)  # 5ms网络延迟
        
        # 增加计数器
        monitor.increment_token_count()
        monitor.increment_counters()
        
        monitor.record_memory_snapshot(f"迭代 {i+1}")
    
    # 停止内存跟踪
    monitor.stop_memory_tracking()
    
    # 打印报告
    monitor.print_detailed_report()
    monitor.print_memory_timeline()
    
    # 获取摘要统计
    stats = monitor.get_summary_stats()
    print(f"\n📊 测试摘要:")
    print(f"   Token数: {stats['token_count']}")
    print(f"   GPU总时间: {stats['cloud_total_time']:.4f}s")
    print(f"   CPU总时间: {stats['edge_total_time']:.4f}s")
    print(f"   网络总时间: {stats['network_total_time']:.4f}s")
    
    print("✅ 性能监控器测试完成！")

if __name__ == "__main__":
    test_performance_monitor()
