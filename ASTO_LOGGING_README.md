# ASTO算法日志记录系统

本项目为ASTO（自适应搜索优化）算法添加了完整的日志记录和分析系统，可以详细追踪算法运行过程中的每一轮迭代数据。

## 新增功能

### 1. 完整的日志记录
- **每轮迭代时间记录**：精确记录每次迭代的执行时间
- **种群适应度记录**：保存每个个体的适应度值和基因型
- **算法参数记录**：记录所有关键参数（alpha值、迭代次数等）
- **阶段性统计**：记录最佳适应度、平均适应度等统计信息

### 2. 多层次日志文件
- `warm_asto_log.pkl`：热身阶段的详细日志
- `asto_v2_alpha_X.X_log.pkl`：每个alpha值对应的生成阶段日志  
- `asto_complete_log.pkl`：整个算法的完整汇总日志
- `warm_phase_detailed.csv`：热身阶段的CSV格式详细数据
- `generate_alpha_X.X_detailed.csv`：生成阶段的CSV格式数据

## 使用方法

### 基本使用
```python
from astogptJ import asto

# 运行ASTO算法并自动记录日志
complete_log = asto(warm_epoch=20, generate_epoch=10)
```

### 分析日志数据
```python
from astogptJ import analyze_asto_logs, save_detailed_population_history

# 分析生成的日志
analyze_asto_logs()

# 保存详细的种群历史数据为CSV格式
save_detailed_population_history()
```

### 完整演示
```bash
# 运行完整的演示脚本
python demo_asto_logging.py

# 运行日志分析脚本
python analyze_asto_logs.py
```

## 日志数据结构

### 完整日志结构 (`asto_complete_log.pkl`)
```python
{
    'algorithm': 'ASTO',
    'parameters': {
        'warm_epochs': int,
        'generate_epochs': int
    },
    'total_time': float,           # 总执行时间（秒）
    'warm_results': dict,          # 热身阶段结果
    'generate_results': dict,      # 生成阶段结果
    'timestamp': str               # 执行时间戳
}
```

### 迭代日志结构
```python
{
    'epoch': int,                  # 迭代轮次
    'alpha': float,                # 当前alpha值
    'time': float,                 # 本轮迭代时间
    'population_size': int,        # 种群大小
    'population': [                # 种群详细信息
        {
            'species': list,       # 个体基因型
            'fitness': float       # 适应度值
        }
    ],
    'best_fitness': float,         # 最佳适应度
    'avg_fitness': float          # 平均适应度
}
```

## 分析工具

### 1. 实时监控
算法运行过程中会实时输出：
- 每轮迭代的执行时间
- 种群大小和最佳适应度
- 各阶段的进度信息

### 2. 可视化分析
`analyze_asto_logs.py`脚本提供：
- 适应度随迭代次数的变化曲线
- 每轮迭代时间的柱状图
- 不同alpha值的性能对比
- 算法各阶段的时间分布

### 3. 汇总报告
自动生成包含以下内容的文本报告：
- 算法执行参数和总时间
- 热身阶段详细分析
- 各alpha值的生成阶段性能
- 整体性能统计

## 文件说明

### 核心文件
- `astogptJ.py`：主算法文件，包含增强的日志记录功能
- `demo_asto_logging.py`：演示脚本，展示如何使用日志功能
- `analyze_asto_logs.py`：日志分析和可视化脚本

### 输出文件
- `*.pkl`：Python pickle格式的二进制日志文件
- `*.csv`：CSV格式的详细数据文件，便于Excel等工具分析
- `asto_analysis.png`：算法性能可视化图表
- `asto_summary_report.txt`：文本格式的汇总报告

## 依赖要求

### 基本依赖
- `pickle`：日志数据序列化
- `time`：时间记录
- `os`：文件操作

### 可选依赖（用于高级分析）
```bash
pip install pandas matplotlib
```

## 示例输出

### 运行时输出
```
Warm epoch 1/20: alpha=0.0, time=0.123s, pop_size=100, best_fitness=0.7234
Warm epoch 2/20: alpha=0.1, time=0.118s, pop_size=100, best_fitness=0.7456
...
Generate epoch 1/10 (alpha=0.5): time=0.089s, pop_size=30, best_fitness=0.8123
```

### 汇总报告示例
```
ASTO Algorithm Execution Report
==================================================
Execution Time: 2025-07-19 10:30:45
Total Runtime: 45.678 seconds

Algorithm Parameters:
  - Warm epochs: 20
  - Generate epochs: 10

Warm Phase Analysis:
  - Total time: 12.345s
  - Number of iterations: 20
  - Average time per iteration: 0.617s
  - Initial best fitness: 0.3456
  - Final best fitness: 0.7890
  - Fitness improvement: 0.4434
```

## 注意事项

1. **存储空间**：日志文件可能较大，特别是包含完整种群历史的文件
2. **性能影响**：日志记录会略微增加算法运行时间（通常<5%）
3. **文件清理**：建议定期清理旧的日志文件以节省空间

## 扩展说明

该日志系统设计为模块化和可扩展的：
- 可以轻松添加新的统计指标
- 支持自定义输出格式
- 可以集成到现有的实验管理系统中

通过这个完整的日志系统，您可以：
1. 深入分析ASTO算法的收敛行为
2. 比较不同参数设置的效果
3. 优化算法性能
4. 生成学术论文所需的详细实验数据
