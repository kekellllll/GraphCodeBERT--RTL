#!/usr/bin/env python3
"""
RTL数据集生成工具 / RTL Dataset Generation Tool

该工具用于生成RTL错误修正训练数据集，包括：
This tool generates RTL error correction training datasets, including:
- 错误的Verilog代码 / Buggy Verilog code
- 对应的正确代码 / Corresponding correct code  
- 代码注释 / Code comments
- 错误类型标注 / Error type annotations

使用方法 / Usage:
python generate_rtl_dataset.py --output datasets/rtl_training --size 1000
"""

import os
import json
import random
import argparse
from typing import List, Dict, Tuple
from datetime import datetime

class RTLDatasetGenerator:
    """RTL错误修正数据集生成器"""
    
    def __init__(self):
        self.error_types = [
            "unnecessary_arithmetic",
            "missing_parentheses", 
            "blocking_assignment",
            "clock_sensitivity",
            "wire_reg_mismatch",
            "port_connection",
            "syntax_error",
            "logic_error"
        ]
        
        # 基础模块模板
        self.module_templates = [
            {
                "name": "simple_assign",
                "correct": "module {name}(input {in_sig}, output {out_sig}); assign {out_sig} = {in_sig}; endmodule",
                "buggy": "module {name}(input {in_sig}, output {out_sig}); assign {out_sig} = {in_sig} + 1; endmodule", 
                "error_type": "unnecessary_arithmetic",
                "comment": "Simple wire connection module"
            },
            {
                "name": "logic_gate", 
                "correct": "module {name}(input {in1}, {in2}, output {out}); assign {out} = ({in1} & {in2}) | {in3}; endmodule",
                "buggy": "module {name}(input {in1}, {in2}, output {out}); assign {out} = {in1} & {in2} | {in3}; endmodule",
                "error_type": "missing_parentheses", 
                "comment": "Logic gate with proper operator precedence"
            },
            {
                "name": "dff_register",
                "correct": "always @(posedge clk) begin q <= d; end",
                "buggy": "always @(posedge clk) begin q = d; end", 
                "error_type": "blocking_assignment",
                "comment": "D flip-flop register with non-blocking assignment"
            },
            {
                "name": "counter",
                "correct": "always @(posedge clk, negedge rst_n) begin if (!rst_n) count <= 0; else count <= count + 1; end",
                "buggy": "always @(posedge clk) begin if (!rst_n) count <= 0; else count <= count + 1; end",
                "error_type": "clock_sensitivity", 
                "comment": "Counter with proper reset sensitivity"
            },
            {
                "name": "mux_2to1",
                "correct": "assign out = sel ? in1 : in0;",
                "buggy": "assign out = sel ? in1; in0;",
                "error_type": "syntax_error",
                "comment": "2-to-1 multiplexer with conditional assignment" 
            }
        ]
        
        # 信号名称选项
        self.signal_names = {
            'input': ['a', 'b', 'c', 'd', 'in', 'data', 'x', 'y', 'clk', 'rst', 'en'],
            'output': ['out', 'result', 'q', 'y', 'sum', 'prod', 'valid'], 
            'module': ['test', 'example', 'demo', 'simple', 'basic', 'logic', 'arith']
        }

    def generate_signals(self) -> Dict[str, str]:
        """生成随机信号名称"""
        return {
            'name': random.choice(self.signal_names['module']) + f"_{random.randint(1, 999)}",
            'in_sig': random.choice(self.signal_names['input']),
            'out_sig': random.choice(self.signal_names['output']),
            'in1': random.choice(self.signal_names['input']),
            'in2': random.choice(self.signal_names['input']), 
            'in3': random.choice(self.signal_names['input']),
            'out': random.choice(self.signal_names['output'])
        }

    def generate_single_example(self) -> Dict[str, str]:
        """生成单个训练样本"""
        template = random.choice(self.module_templates)
        signals = self.generate_signals()
        
        try:
            correct_code = template['correct'].format(**signals)
            buggy_code = template['buggy'].format(**signals) 
        except KeyError:
            # 如果格式化失败，使用默认信号名
            default_signals = {
                'name': 'test_module',
                'in_sig': 'a', 'out_sig': 'b',
                'in1': 'in1', 'in2': 'in2', 'in3': 'in3', 'out': 'out'
            }
            correct_code = template['correct'].format(**default_signals)
            buggy_code = template['buggy'].format(**default_signals)
        
        return {
            'buggy_code': buggy_code,
            'correct_code': correct_code,
            'comments': template['comment'],
            'error_type': template['error_type'],
            'template_name': template['name'],
            'generated_at': datetime.now().isoformat()
        }

    def generate_dataset(self, size: int) -> List[Dict[str, str]]:
        """生成指定大小的数据集"""
        dataset = []
        for i in range(size):
            example = self.generate_single_example()
            example['id'] = i
            dataset.append(example)
        return dataset

    def save_dataset(self, dataset: List[Dict[str, str]], output_dir: str, split_name: str):
        """保存数据集到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSONL格式
        jsonl_file = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # 保存为JSON格式 
        json_file = os.path.join(output_dir, f"{split_name}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已保存 {len(dataset)} 个样本到 {split_name}.jsonl 和 {split_name}.json")

def main():
    parser = argparse.ArgumentParser(description='生成RTL错误修正数据集')
    parser.add_argument('--output', type=str, default='datasets/rtl_error_correction',
                      help='输出目录路径') 
    parser.add_argument('--size', type=int, default=1000,
                      help='生成的总样本数')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                      help='训练集比例')
    parser.add_argument('--valid_ratio', type=float, default=0.15, 
                      help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                      help='测试集比例')
    
    args = parser.parse_args()
    
    # 验证比例
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"❌ 错误：数据集分割比例总和应为1.0，当前为{total_ratio}")
        return
    
    generator = RTLDatasetGenerator()
    
    print(f"🚀 开始生成RTL错误修正数据集...")
    print(f"📊 总样本数: {args.size}")
    print(f"📁 输出目录: {args.output}")
    print(f"📈 数据分割: 训练{args.train_ratio:.0%}, 验证{args.valid_ratio:.0%}, 测试{args.test_ratio:.0%}")
    
    # 生成完整数据集
    full_dataset = generator.generate_dataset(args.size)
    
    # 计算分割大小
    train_size = int(args.size * args.train_ratio)
    valid_size = int(args.size * args.valid_ratio) 
    test_size = args.size - train_size - valid_size
    
    # 随机打乱并分割数据
    random.shuffle(full_dataset)
    train_set = full_dataset[:train_size]
    valid_set = full_dataset[train_size:train_size + valid_size]
    test_set = full_dataset[train_size + valid_size:]
    
    # 保存数据集
    generator.save_dataset(train_set, args.output, 'train')
    generator.save_dataset(valid_set, args.output, 'valid') 
    generator.save_dataset(test_set, args.output, 'test')
    
    # 生成数据集统计信息
    stats = {
        'total_samples': args.size,
        'train_samples': len(train_set),
        'valid_samples': len(valid_set), 
        'test_samples': len(test_set),
        'error_types': generator.error_types,
        'generation_time': datetime.now().isoformat(),
        'templates_used': len(generator.module_templates)
    }
    
    stats_file = os.path.join(args.output, 'dataset_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 数据集生成完成！")
    print(f"📋 统计信息已保存到: {stats_file}")
    print(f"🔍 错误类型覆盖: {len(generator.error_types)} 种")
    
    # 显示样本示例
    print(f"\n📝 样本示例:")
    example = train_set[0]
    print(f"错误代码: {example['buggy_code']}")
    print(f"正确代码: {example['correct_code']}")
    print(f"错误类型: {example['error_type']}")
    print(f"注释: {example['comments']}")

if __name__ == "__main__":
    main()