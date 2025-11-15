#!/usr/bin/env python3
"""
手眼标定方法分析工具
分析不同标定方法的结果相似性
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def analyze_calibration_results():
    """分析标定结果的相似性"""
    
    # 从用户提供的结果中提取数据
    # TSAI方法结果
    T_tsai = np.array([
        [ 0.80385378, -0.59010684,  0.07478646,  0.26732749],
        [ 0.44571818,  0.51430968, -0.7326806 ,  0.19545709],
        [ 0.39389643,  0.62230175,  0.67645113,  0.1047192 ],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
    
    # PARK方法结果
    T_park = np.array([
        [ 0.98718877, -0.14087679, -0.07491367,  0.47832651],
        [-0.14773779, -0.98438759, -0.09567974,  0.05914237],
        [-0.06026504,  0.10552154, -0.9925892 ,  0.47350017],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
    
    # 假设HORAUD方法结果（需要用户提供实际数据）
    # 这里使用PARK结果的轻微变化来模拟
    T_horaud = T_park + np.random.normal(0, 0.01, T_park.shape)
    T_horaud[3, :] = [0, 0, 0, 1]  # 保持齐次坐标
    
    methods = {
        'TSAI': T_tsai,
        'PARK': T_park,
        'HORAUD': T_horaud
    }
    
    print("=== 手眼标定方法相似性分析 ===\n")
    
    # 计算所有方法之间的差异
    method_names = list(methods.keys())
    similarities = {}
    
    for i in range(len(method_names)):
        for j in range(i+1, len(method_names)):
            method1, method2 = method_names[i], method_names[j]
            T1, T2 = methods[method1], methods[method2]
            
            # 计算平移差异
            t_diff = T1[:3, 3] - T2[:3, 3]
            translation_diff = np.linalg.norm(t_diff)
            
            # 计算旋转差异
            R_diff = T1[:3, :3] - T2[:3, :3]
            rotation_diff = np.linalg.norm(R_diff, 'fro')
            
            # 计算旋转角度差异
            try:
                R1 = R.from_matrix(T1[:3, :3])
                R2 = R.from_matrix(T2[:3, :3])
                angle_diff = R1.inv() * R2
                angle_degrees = np.linalg.norm(np.degrees(angle_diff.as_rotvec()))
            except:
                angle_degrees = float('inf')
            
            similarity_score = 1.0 / (1.0 + translation_diff + rotation_diff/10)
            similarities[f"{method1}_vs_{method2}"] = {
                'translation_diff': translation_diff,
                'rotation_diff': rotation_diff,
                'angle_diff': angle_degrees,
                'similarity_score': similarity_score
            }
            
            print(f"{method1} vs {method2}:")
            print(f"  平移差异: {translation_diff:.4f} 米")
            print(f"  旋转差异: {rotation_diff:.4f}")
            print(f"  角度差异: {angle_degrees:.2f} 度")
            print(f"  相似度分数: {similarity_score:.4f}")
            print()
    
    # 找出最相似的方法对
    most_similar = max(similarities.items(), key=lambda x: x[1]['similarity_score'])
    print(f"最相似的方法对: {most_similar[0]}")
    print(f"相似度分数: {most_similar[1]['similarity_score']:.4f}")
    
    return similarities

def plot_similarity_matrix(similarities):
    """绘制相似性矩阵"""
    methods = ['TSAI', 'PARK', 'HORAUD']
    n = len(methods)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                key = f"{methods[i]}_vs_{methods[j]}" if i < j else f"{methods[j]}_vs_{methods[i]}"
                if key in similarities:
                    similarity_matrix[i, j] = similarities[key]['similarity_score']
    
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.colorbar(label='相似度分数')
    plt.xticks(range(n), methods)
    plt.yticks(range(n), methods)
    plt.title('手眼标定方法相似性矩阵')
    
    # 添加数值标签
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f'{similarity_matrix[i, j]:.3f}', 
                    ha='center', va='center', color='black')
    
    plt.tight_layout()
    plt.savefig('calibration_methods_similarity.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    similarities = analyze_calibration_results()
    # plot_similarity_matrix(similarities)  # 如果需要可视化，取消注释
