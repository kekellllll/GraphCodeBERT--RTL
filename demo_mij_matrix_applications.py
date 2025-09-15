#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mij矩阵应用演示脚本
Demonstration Script for Mij Matrix Applications

这个脚本展示了Mij矩阵在GraphCodeBERT-RTL中的具体应用场景和效果
This script demonstrates the specific applications and effects of Mij matrix in GraphCodeBERT-RTL

运行方式 (How to run):
python demo_mij_matrix_applications.py
"""

import json
from typing import Dict, List, Tuple

class MijMatrixDemo:
    """Mij矩阵应用演示类 (Mij Matrix Application Demo Class)"""
    
    def __init__(self):
        self.setup_demo_data()
    
    def setup_demo_data(self):
        """设置演示数据 (Setup demo data)"""
        # 示例1：简单的RTL代码
        self.example_1 = {
            'buggy_code': 'module test(input a, output b); assign b = a + 1; endmodule',
            'correct_code': 'module test(input a, output b); assign b = a; endmodule',
            'comments': '简单的线连接模块 (Simple wire connection module)',
            'dfg_nodes': ['a', 'b', 'assign'],
            'dfg_edges': [('b', 'computedFrom', ['a'])]
        }
        
        # 示例2：复杂的时序逻辑
        self.example_2 = {
            'buggy_code': 'always @(posedge clk) begin q = d; end',  # 应该用<=
            'correct_code': 'always @(posedge clk) begin q <= d; end',
            'comments': '时钟边沿触发的寄存器 (Clock edge triggered register)',
            'dfg_nodes': ['clk', 'q', 'd', 'always'],
            'dfg_edges': [('q', 'computedFrom', ['d']), ('always', 'triggeredBy', ['clk'])]
        }
        
        # 示例3：逻辑表达式错误
        self.example_3 = {
            'buggy_code': 'assign out = in1 & in2 | in3;',  # 缺少括号
            'correct_code': 'assign out = (in1 & in2) | in3;',
            'comments': '逻辑门组合电路 (Logic gate combination circuit)',
            'dfg_nodes': ['out', 'in1', 'in2', 'in3', 'assign'],
            'dfg_edges': [('out', 'computedFrom', ['in1', 'in2', 'in3'])]
        }
    
    def demonstrate_mij_fusion(self, example: Dict) -> Dict:
        """
        演示Mij矩阵融合过程
        Demonstrate Mij matrix fusion process
        """
        print(f"\n{'='*60}")
        print(f"Mij矩阵融合演示 (Mij Matrix Fusion Demonstration)")
        print(f"{'='*60}")
        
        # 1. 代码标记化
        code_tokens = self.tokenize_verilog(example['buggy_code'])
        comment_tokens = self.tokenize_comments(example['comments'])
        dfg_nodes = example['dfg_nodes']
        
        print(f"代码标记 (Code tokens): {code_tokens}")
        print(f"注释标记 (Comment tokens): {comment_tokens}")
        print(f"DFG节点 (DFG nodes): {dfg_nodes}")
        
        # 2. 构建输入序列
        source_tokens = ['<cls>'] + comment_tokens + ['<sep>'] + code_tokens + dfg_nodes
        print(f"\n输入序列 (Input sequence): {source_tokens}")
        
        # 3. 创建位置编码
        position_idx = self.create_position_encoding(source_tokens, comment_tokens, code_tokens, dfg_nodes)
        print(f"位置编码 (Position encoding): {position_idx}")
        
        # 4. 创建Mij矩阵
        mij_matrix = self.create_mij_matrix(source_tokens, dfg_nodes, example['dfg_edges'])
        
        # 5. 显示融合效果
        fusion_result = self.simulate_fusion_effect(mij_matrix, source_tokens)
        
        return {
            'source_tokens': source_tokens,
            'position_idx': position_idx,
            'mij_matrix': mij_matrix,
            'fusion_result': fusion_result
        }
    
    def tokenize_verilog(self, code: str) -> List[str]:
        """简单的Verilog代码标记化"""
        # 移除标点符号并分词
        tokens = []
        for word in code.replace(';', ' ; ').replace('(', ' ( ').replace(')', ' ) ').split():
            if word.strip():
                tokens.append(word.strip())
        return tokens
    
    def tokenize_comments(self, comments: str) -> List[str]:
        """注释标记化"""
        # 提取中文和英文部分
        return comments.split()[:3]  # 简化处理
    
    def create_position_encoding(self, source_tokens: List[str], comment_tokens: List[str], 
                               code_tokens: List[str], dfg_nodes: List[str]) -> List[int]:
        """
        创建位置编码
        0: DFG节点 (DFG nodes)
        1: 注释 (Comments)  
        2+: 代码标记 (Code tokens)
        """
        position_idx = []
        comment_start = 1  # 跳过<cls>
        code_start = comment_start + len(comment_tokens) + 1  # +1 for <sep>
        dfg_start = code_start + len(code_tokens)
        
        for i, token in enumerate(source_tokens):
            if i == 0:  # <cls>
                position_idx.append(2)
            elif comment_start <= i < code_start - 1:  # 注释部分
                position_idx.append(1)
            elif i == code_start - 1:  # <sep>
                position_idx.append(1)
            elif code_start <= i < dfg_start:  # 代码部分
                position_idx.append(i - code_start + 2)
            else:  # DFG节点部分
                position_idx.append(0)
        
        return position_idx
    
    def create_mij_matrix(self, source_tokens: List[str], dfg_nodes: List[str], 
                         dfg_edges: List[Tuple]) -> List[List[float]]:
        """创建Mij矩阵"""
        n = len(source_tokens)
        # 创建基础单位矩阵
        mij_matrix = [[0.5 if i == j else 0.0 for j in range(n)] for i in range(n)]
        
        # 添加DFG边的权重
        for edge in dfg_edges:
            target_node = edge[0]
            source_nodes = edge[2]
            
            # 找到节点在token序列中的位置
            if target_node in source_tokens and isinstance(source_nodes, list):
                target_idx = source_tokens.index(target_node)
                for source_node in source_nodes:
                    if source_node in source_tokens:
                        source_idx = source_tokens.index(source_node)
                        mij_matrix[target_idx][source_idx] = 0.9
                        mij_matrix[source_idx][target_idx] = 0.9
        
        # 添加相邻token的弱连接
        for i in range(n-1):
            mij_matrix[i][i+1] = 0.3
            mij_matrix[i+1][i] = 0.3
        
        return mij_matrix
    
    def simulate_fusion_effect(self, mij_matrix: List[List[float]], source_tokens: List[str]) -> Dict:
        """模拟融合效果"""
        # 创建模拟的嵌入向量
        embedding_dim = 8  # 简化的嵌入维度
        import random
        random.seed(42)  # 确保结果可重现
        
        embeddings = [[random.gauss(0, 1) for _ in range(embedding_dim)] for _ in range(len(source_tokens))]
        
        # 应用Mij矩阵融合 (简化的矩阵乘法)
        fused_embeddings = []
        for i in range(len(source_tokens)):
            fused_row = [0.0] * embedding_dim
            for j in range(len(source_tokens)):
                for k in range(embedding_dim):
                    fused_row[k] += mij_matrix[i][j] * embeddings[j][k]
            fused_embeddings.append(fused_row)
        
        # 计算融合强度
        fusion_strength = 0.0
        count = 0
        for i in range(len(embeddings)):
            for j in range(len(embeddings[i])):
                fusion_strength += abs(fused_embeddings[i][j] - embeddings[i][j])
                count += 1
        fusion_strength = fusion_strength / count if count > 0 else 0.0
        
        return {
            'original_embeddings': embeddings,
            'fused_embeddings': fused_embeddings,
            'fusion_strength': fusion_strength
        }
    
    def demonstrate_error_detection(self, example: Dict) -> Dict:
        """演示错误检测过程"""
        print(f"\n{'='*60}")
        print(f"错误检测演示 (Error Detection Demonstration)")
        print(f"{'='*60}")
        
        buggy_code = example['buggy_code']
        correct_code = example['correct_code']
        
        print(f"缺陷代码 (Buggy code): {buggy_code}")
        print(f"正确代码 (Correct code): {correct_code}")
        
        # 模拟错误检测过程
        errors = self.detect_errors(buggy_code, example)
        
        for error in errors:
            print(f"\n检测到错误 (Error detected):")
            print(f"  位置 (Position): 第{error['line']}行, 第{error['column_start']}-{error['column_end']}列")
            print(f"  类型 (Type): {error['type']}")
            print(f"  置信度 (Confidence): {error['confidence']:.2f}")
            print(f"  描述 (Description): {error['description']}")
        
        return {'detected_errors': errors}
    
    def detect_errors(self, code: str, example: Dict) -> List[Dict]:
        """模拟错误检测算法"""
        errors = []
        
        # 检测不必要的算术运算
        if '+ 1' in code and 'assign' in code:
            pos = code.find('+ 1')
            errors.append({
                'line': 1,
                'column_start': pos,
                'column_end': pos + 3,
                'type': 'unnecessary_arithmetic',
                'confidence': 0.95,
                'description': '不必要的算术运算，应该是直接连接'
            })
        
        # 检测阻塞赋值错误
        if 'always' in code and ' = ' in code and '<=' not in code:
            pos = code.find(' = ')
            errors.append({
                'line': 1,
                'column_start': pos,
                'column_end': pos + 3,
                'type': 'blocking_assignment',
                'confidence': 0.75,
                'description': '在时序逻辑中应使用非阻塞赋值(<= )'
            })
        
        # 检测缺少括号
        if '&' in code and '|' in code and '(' not in code:
            pos = code.find('&')
            errors.append({
                'line': 1,
                'column_start': pos,
                'column_end': len(code),
                'type': 'missing_parentheses',
                'confidence': 0.85,
                'description': '逻辑表达式中缺少括号，可能导致优先级错误'
            })
        
        return errors
    
    def visualize_mij_matrix(self, demo_result: Dict):
        """可视化Mij矩阵"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            mij_matrix = demo_result['mij_matrix']
            source_tokens = demo_result['source_tokens']
            
            plt.figure(figsize=(12, 10))
            
            # 创建热力图
            sns.heatmap(mij_matrix, 
                       xticklabels=source_tokens, 
                       yticklabels=source_tokens,
                       annot=True, 
                       fmt='.2f', 
                       cmap='Blues',
                       cbar_kws={'label': 'Attention Weight'})
            
            plt.title('Mij矩阵可视化 (Mij Matrix Visualization)', fontsize=16, pad=20)
            plt.xlabel('Source Tokens', fontsize=12)
            plt.ylabel('Target Tokens', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # 保存图片
            plt.savefig('mij_matrix_visualization.png', dpi=300, bbox_inches='tight')
            print(f"\nMij矩阵可视化已保存为: mij_matrix_visualization.png")
            
            # 显示图片（如果在支持的环境中）
            plt.show()
            
        except ImportError:
            print("\n注意: matplotlib/seaborn未安装，使用文本方式显示Mij矩阵")
            print("Note: matplotlib/seaborn not installed, showing Mij matrix in text format")
            self.print_mij_matrix_text(demo_result)
    
    def print_mij_matrix_text(self, demo_result: Dict):
        """以文本形式显示Mij矩阵"""
        mij_matrix = demo_result['mij_matrix']
        source_tokens = demo_result['source_tokens']
        
        print(f"\nMij矩阵 (文本格式):")
        print("=" * 80)
        
        # 打印列标题
        print(f"{'Token':<12}", end="")
        for token in source_tokens[:10]:  # 只显示前10个token
            print(f"{token:<8}", end="")
        print()
        
        # 打印矩阵行
        for i, token in enumerate(source_tokens[:10]):
            print(f"{token:<12}", end="")
            for j in range(min(10, len(source_tokens))):
                print(f"{mij_matrix[i][j]:<8.2f}", end="")
            print()
        
        print("=" * 80)
    
    def generate_summary_report(self, all_results: List[Dict]):
        """生成总结报告"""
        print(f"\n{'='*60}")
        print(f"Mij矩阵应用总结报告 (Mij Matrix Application Summary Report)")
        print(f"{'='*60}")
        
        total_errors = sum(len(result.get('detected_errors', [])) for result in all_results)
        
        print(f"演示样例数量 (Demo examples): {len(all_results)}")
        print(f"检测到的错误总数 (Total errors detected): {total_errors}")
        
        # 错误类型统计
        error_types = {}
        for result in all_results:
            for error in result.get('detected_errors', []):
                error_type = error['type']
                if error_type not in error_types:
                    error_types[error_type] = {'count': 0, 'avg_confidence': 0}
                error_types[error_type]['count'] += 1
                error_types[error_type]['avg_confidence'] += error['confidence']
        
        print(f"\n错误类型分析 (Error type analysis):")
        for error_type, stats in error_types.items():
            avg_conf = stats['avg_confidence'] / stats['count']
            print(f"  {error_type}: {stats['count']}个错误, 平均置信度 {avg_conf:.2f}")
        
        # 应用场景总结
        print(f"\nMij矩阵主要应用场景 (Main application scenarios):")
        print(f"1. DFG与代码融合 - 增强语义理解")
        print(f"2. 多模态位置编码 - 区分不同信息类型")
        print(f"3. 错误检测定位 - 精确定位缺陷位置")
        print(f"4. 代码修正生成 - 基于结构化理解")
        
        # 技术优势
        print(f"\n技术优势 (Technical advantages):")
        print(f"✓ 保持GraphCodeBERT原有架构")
        print(f"✓ 增强多模态信息融合能力")
        print(f"✓ 提供精确的错误定位")
        print(f"✓ 支持自动代码修正")
    
    def run_full_demonstration(self):
        """运行完整演示"""
        print("Mij矩阵应用演示开始 (Mij Matrix Application Demonstration Started)")
        print("=" * 80)
        
        examples = [self.example_1, self.example_2, self.example_3]
        all_results = []
        
        for i, example in enumerate(examples, 1):
            print(f"\n【示例 {i} / Example {i}】")
            
            # 演示Mij矩阵融合
            fusion_result = self.demonstrate_mij_fusion(example)
            
            # 演示错误检测
            detection_result = self.demonstrate_error_detection(example)
            
            # 合并结果
            result = {**fusion_result, **detection_result}
            all_results.append(result)
            
            # 可视化第一个例子的Mij矩阵
            if i == 1:
                self.visualize_mij_matrix(fusion_result)
        
        # 生成总结报告
        self.generate_summary_report(all_results)
        
        print(f"\n{'='*80}")
        print("演示完成！(Demonstration completed!)")
        print("详细文档请参考: MIJ_MATRIX_APPLICATIONS.md")
        print("For detailed documentation, please refer to: MIJ_MATRIX_APPLICATIONS.md")

def main():
    """主函数"""
    demo = MijMatrixDemo()
    demo.run_full_demonstration()

if __name__ == "__main__":
    main()