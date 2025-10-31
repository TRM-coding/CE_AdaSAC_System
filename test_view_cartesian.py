import sys
sys.path.insert(0, './src/llama.cpp/build/ops')
import opslib as ops
import csv
import os
import itertools

def test_view_1d():
    """测试 VIEW_1D 的笛卡尔积"""
    results = []
    times = 50
    type_f32 = ops.ggml_type.GGML_TYPE_F32
    
    # VIEW_1D 参数范围
    D_values = [8, 64, 256, 1024]
    T_values = [16, 256, 1024] 
    group_values = [1, 2, 4, 8]
    
    print("Testing VIEW_1D with Cartesian product...")
    
    for D, T, group in itertools.product(D_values, T_values, group_values):
        try:
            # VIEW_1D: 将 (D, T, group, 1) 视为 (D*T*group, 1, 1, 1)
            input_ne = [D, T, group, 1]
            view_ne = [D * T * group]
            
            info = ops.run_view_1d(times, type_f32, input_ne, view_ne)
            results.append({
                "op": "VIEW_1D",
                "in_shapes": f"{{{','.join(map(str, input_ne))}}}",
                "out_shape": f"{{{view_ne[0]}}}",
                "D": D,
                "T": T,
                "group": group,
                "backend": "cpu-ggml",
                "device": "host",
                "avg_ms": f"{info.time_per_op_ms:.6f}"
            })
            print(f"VIEW_1D D={D}, T={T}, group={group}: {info.time_per_op_ms:.6f} ms")
        except Exception as e:
            print(f"VIEW_1D D={D}, T={T}, group={group} failed: {e}")
    
    return results

def test_view_2d():
    """测试 VIEW_2D 的笛卡尔积"""
    results = []
    times = 50
    type_f32 = ops.ggml_type.GGML_TYPE_F32
    
    # VIEW_2D 参数范围
    D_values = [8, 64, 256, 1024]
    T_values = [4, 64, 256, 1024]
    group_values = [1, 2, 4, 8]
    
    print("Testing VIEW_2D with Cartesian product...")
    
    for D, T, group in itertools.product(D_values, T_values, group_values):
        try:
            # VIEW_2D: 将 (D, T, group, 1) 视为 (D*group, T, 1, 1)
            input_ne = [D, T, group, 1]
            view_ne = [D * group, T]
            
            info = ops.run_view_2d(times, type_f32, input_ne, view_ne)
            results.append({
                "op": "VIEW_2D",
                "in_shapes": f"{{{','.join(map(str, input_ne))}}}",
                "out_shape": f"{{{view_ne[0]},{view_ne[1]}}}",
                "D": D,
                "T": T,
                "group": group,
                "backend": "cpu-ggml",
                "device": "host",
                "avg_ms": f"{info.time_per_op_ms:.6f}"
            })
            print(f"VIEW_2D D={D}, T={T}, group={group}: {info.time_per_op_ms:.6f} ms")
        except Exception as e:
            print(f"VIEW_2D D={D}, T={T}, group={group} failed: {e}")
    
    return results

def test_view_3d():
    """测试 VIEW_3D 的笛卡尔积"""
    results = []
    times = 50
    type_f32 = ops.ggml_type.GGML_TYPE_F32
    
    # VIEW_3D 参数范围
    D_values = [8, 64, 256, 1024]
    T_values = [4, 64, 256, 1024]
    H_values = [2, 4, 8, 16]
    group_values = [1, 2, 4, 8]
    
    print("Testing VIEW_3D with Cartesian product...")
    
    for D, T, H, group in itertools.product(D_values, T_values, H_values, group_values):
        try:
            # VIEW_3D: 将 (D, T, H, group) 视为 (D*group, T, H, 1)
            input_ne = [D, T, H, group]
            view_ne = [D * group, T, H]
            
            info = ops.run_view_3d(times, type_f32, input_ne, view_ne)
            results.append({
                "op": "VIEW_3D",
                "in_shapes": f"{{{','.join(map(str, input_ne))}}}",
                "out_shape": f"{{{view_ne[0]},{view_ne[1]},{view_ne[2]}}}",
                "D": D,
                "T": T,
                "H": H,
                "group": group,
                "backend": "cpu-ggml",
                "device": "host",
                "avg_ms": f"{info.time_per_op_ms:.6f}"
            })
            print(f"VIEW_3D D={D}, T={T}, H={H}, group={group}: {info.time_per_op_ms:.6f} ms")
        except Exception as e:
            print(f"VIEW_3D D={D}, T={T}, H={H}, group={group} failed: {e}")
    
    return results

def test_view_4d():
    """测试 VIEW_4D 的笛卡尔积"""
    results = []
    times = 50
    type_f32 = ops.ggml_type.GGML_TYPE_F32
    
    # VIEW_4D 参数范围
    D_values = [8, 64, 256, 1024]
    T_values = [4, 64, 256, 1024]
    H_values = [2, 4, 8, 16]
    G_values = [1, 2, 4, 8]  # 注意这里参数名是 G
    
    print("Testing VIEW_4D with Cartesian product...")
    
    for D, T, H, G in itertools.product(D_values, T_values, H_values, G_values):
        try:
            # VIEW_4D: 将 (D, T, H, G) 视为 (D*H*G, T, 1, 1) 或其他4D转换
            input_ne = [D, T, H, G]
            view_ne = [D, T, H, G]  # 4D到4D的视图转换
            
            info = ops.run_view_4d(times, type_f32, input_ne, view_ne)
            results.append({
                "op": "VIEW_4D",
                "in_shapes": f"{{{','.join(map(str, input_ne))}}}",
                "out_shape": f"{{{view_ne[0]},{view_ne[1]},{view_ne[2]},{view_ne[3]}}}",
                "D": D,
                "T": T,
                "H": H,
                "G": G,
                "backend": "cpu-ggml",
                "device": "host",
                "avg_ms": f"{info.time_per_op_ms:.6f}"
            })
            print(f"VIEW_4D D={D}, T={T}, H={H}, G={G}: {info.time_per_op_ms:.6f} ms")
        except Exception as e:
            print(f"VIEW_4D D={D}, T={T}, H={H}, G={G} failed: {e}")
    
    return results

def main():
    # 创建VIEW目录
    os.makedirs('VIEW', exist_ok=True)
    
    # 分别测试4种VIEW操作
    print("Starting Cartesian product tests for 4 VIEW operations...")
    
    # VIEW_1D
    results_1d = test_view_1d()
    if results_1d:
        with open('VIEW/view_1d_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_1d[0].keys())
            writer.writeheader()
            writer.writerows(results_1d)
        print(f"VIEW_1D: {len(results_1d)} results saved to VIEW/view_1d_results.csv")
    
    # VIEW_2D
    results_2d = test_view_2d()
    if results_2d:
        with open('VIEW/view_2d_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_2d[0].keys())
            writer.writeheader()
            writer.writerows(results_2d)
        print(f"VIEW_2D: {len(results_2d)} results saved to VIEW/view_2d_results.csv")
    
    # VIEW_3D
    results_3d = test_view_3d()
    if results_3d:
        with open('VIEW/view_3d_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_3d[0].keys())
            writer.writeheader()
            writer.writerows(results_3d)
        print(f"VIEW_3D: {len(results_3d)} results saved to VIEW/view_3d_results.csv")
    
    # VIEW_4D
    results_4d = test_view_4d()
    if results_4d:
        with open('VIEW/view_4d_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_4d[0].keys())
            writer.writeheader()
            writer.writerows(results_4d)
        print(f"VIEW_4D: {len(results_4d)} results saved to VIEW/view_4d_results.csv")
    
    # 汇总统计
    print(f"\nAll tests completed!")
    print(f"Total tests: VIEW_1D({len(results_1d)}), VIEW_2D({len(results_2d)}), VIEW_3D({len(results_3d)}), VIEW_4D({len(results_4d)})")

if __name__ == "__main__":
    main()
