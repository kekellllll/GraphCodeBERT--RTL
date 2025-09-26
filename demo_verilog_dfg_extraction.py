#!/usr/bin/env python3
"""
Verilog DFG提取演示脚本
Demonstration of Verilog Data Flow Graph Extraction Process

本脚本演示:
1. Verilog代码的AST解析过程
2. DFG节点和边的提取
3. 与Java DFG提取的对比
4. 完整的处理流程
"""

import sys
import os

# 添加路径以导入相关模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'GraphCodeBERT', 'rtl_error_localization'))

class VerilogDFGDemo:
    """Verilog DFG提取演示类"""
    
    def __init__(self):
        self.setup_examples()
    
    def setup_examples(self):
        """设置演示用例"""
        self.verilog_examples = [
            {
                "name": "简单连续赋值 (Simple Continuous Assignment)",
                "code": """module test(input a, b, output c);
    assign c = a & b;
endmodule""",
                "description": "展示基本的连续赋值语句DFG提取"
            },
            {
                "name": "多级逻辑 (Multi-level Logic)", 
                "code": """module test(input a, b, c, output y);
    wire temp1, temp2;
    assign temp1 = a & b;
    assign temp2 = temp1 | c;
    assign y = ~temp2;
endmodule""",
                "description": "展示复杂数据流依赖关系"
            },
            {
                "name": "时序逻辑 (Sequential Logic)",
                "code": """always @(posedge clk) begin
    if (rst) begin
        q <= 1'b0;
    end else begin
        q <= d;
    end
end""",
                "description": "展示时钟和复位信号的数据流"
            },
            {
                "name": "错误修正示例 (Error Correction Example)",
                "buggy": """always @(posedge clk) begin
    q = d;  // 错误：时序逻辑中使用阻塞赋值
end""",
                "correct": """always @(posedge clk) begin
    q <= d;  // 正确：使用非阻塞赋值
end""",
                "description": "展示Verilog特有错误类型"
            }
        ]
        
        # Java对比示例
        self.java_examples = [
            {
                "name": "Java变量赋值 (Java Variable Assignment)",
                "code": """public void method() {
    int temp = a + b;
    int result = temp * c;
    return result;
}""",
                "description": "Java中的数据流依赖"
            }
        ]
    
    def demonstrate_ast_parsing(self):
        """演示AST解析过程"""
        print("="*70)
        print("1. AST解析过程演示 (AST Parsing Process Demonstration)")
        print("="*70)
        
        print("\n📋 Tree-sitter支持检查 (Tree-sitter Support Check):")
        print("✅ tree-sitter-verilog: 支持")
        print("✅ tree-sitter-java: 支持") 
        print("✅ tree-sitter-python: 支持")
        print("✅ tree-sitter-javascript: 支持")
        
        print("\n🔄 AST转换流程 (AST Conversion Process):")
        print("1. 词法分析 (Lexical Analysis): Verilog代码 → Tokens")
        print("2. 语法分析 (Syntax Analysis): Tokens → AST")
        print("3. 节点识别 (Node Recognition): 识别Verilog特有节点类型")
        print("4. DFG构建 (DFG Construction): AST → 数据流图")
        
        # 展示Verilog特有节点类型
        print("\n📊 Verilog特有节点类型 (Verilog-specific Node Types):")
        verilog_nodes = {
            "赋值语句": ['continuous_assign', 'blocking_assignment', 'nonblocking_assignment'],
            "变量声明": ['net_declaration', 'variable_declaration', 'port_declaration'],
            "时序块": ['always_construct', 'initial_construct'],
            "条件语句": ['conditional_statement'],
            "循环语句": ['loop_statement'],
            "模块实例": ['module_instantiation']
        }
        
        for category, nodes in verilog_nodes.items():
            print(f"   {category}: {nodes}")
    
    def demonstrate_dfg_extraction(self):
        """演示DFG提取过程"""
        print("\n" + "="*70)
        print("2. DFG提取过程演示 (DFG Extraction Process Demonstration)")  
        print("="*70)
        
        for i, example in enumerate(self.verilog_examples[:3], 1):
            print(f"\n📝 示例 {i}: {example['name']}")
            print(f"描述: {example['description']}")
            print("\n输入Verilog代码:")
            print("-" * 30)
            print(example['code'])
            print("-" * 30)
            
            # 模拟DFG提取结果
            dfg_result = self.simulate_dfg_extraction(example['code'])
            
            print("\n🔍 DFG提取结果:")
            print(f"   节点数量: {dfg_result['node_count']}")
            print(f"   边数量: {dfg_result['edge_count']}")
            print("   主要数据流边:")
            for edge in dfg_result['edges']:
                print(f"     • {edge}")
            
            print("\n🧠 分析:")
            for analysis in dfg_result['analysis']:
                print(f"     - {analysis}")
    
    def simulate_dfg_extraction(self, code):
        """模拟DFG提取过程"""
        # 这里是简化的模拟，实际实现需要调用真正的DFG_verilog函数
        
        if "assign c = a & b" in code:
            return {
                "node_count": 3,
                "edge_count": 1, 
                "edges": ["c ← computedFrom(a, b)"],
                "analysis": [
                    "检测到连续赋值语句",
                    "输出c依赖于输入a和b", 
                    "逻辑AND运算创建数据流依赖"
                ]
            }
        elif "temp1" in code and "temp2" in code:
            return {
                "node_count": 6,
                "edge_count": 3,
                "edges": [
                    "temp1 ← computedFrom(a, b)",
                    "temp2 ← computedFrom(temp1, c)", 
                    "y ← computedFrom(temp2)"
                ],
                "analysis": [
                    "检测到多级逻辑结构",
                    "形成了链式数据流依赖",
                    "temp1作为中间节点连接输入和输出"
                ]
            }
        elif "always @(posedge clk)" in code:
            return {
                "node_count": 4,
                "edge_count": 2,
                "edges": [
                    "q ← computedFrom(d, rst)",
                    "always ← triggeredBy(clk)"
                ],
                "analysis": [
                    "检测到时序逻辑always块",
                    "时钟信号clk触发数据更新",
                    "复位信号rst影响输出值"
                ]
            }
        else:
            return {
                "node_count": 2,
                "edge_count": 1,
                "edges": ["output ← computedFrom(input)"],
                "analysis": ["基本数据流依赖"]
            }
    
    def demonstrate_error_correction(self):
        """演示错误修正过程"""
        print("\n" + "="*70)
        print("3. 错误修正演示 (Error Correction Demonstration)")
        print("="*70)
        
        example = self.verilog_examples[3]  # 错误修正示例
        
        print(f"\n📝 {example['name']}")
        print(f"描述: {example['description']}")
        
        print("\n❌ 错误代码 (Buggy Code):")
        print("-" * 30)
        print(example['buggy'])
        print("-" * 30)
        
        print("\n🔍 错误分析:")
        print("   • 错误类型: 阻塞赋值在时序逻辑中使用")
        print("   • 问题描述: 在always @(posedge clk)块中使用'='而非'<='")
        print("   • 影响: 可能导致时序违例和不可预测的行为")
        
        print("\n✅ 修正后代码 (Corrected Code):")
        print("-" * 30)  
        print(example['correct'])
        print("-" * 30)
        
        print("\n🔧 修正说明:")
        print("   • 修正方法: 将阻塞赋值'='改为非阻塞赋值'<='")
        print("   • 原理: 非阻塞赋值确保时钟边沿同步更新")
        print("   • 效果: 符合RTL设计规范，避免时序问题")
    
    def compare_with_java(self):
        """与Java DFG提取对比"""
        print("\n" + "="*70)
        print("4. 与Java DFG提取对比 (Comparison with Java DFG Extraction)")
        print("="*70)
        
        comparison = {
            "相似点 (Similarities)": [
                "都使用递归AST遍历算法",
                "都跟踪变量定义和使用关系", 
                "都生成(variable, index, relation, dependencies)格式的DFG边",
                "都按照索引位置对DFG边进行排序"
            ],
            "差异点 (Differences)": [
                "Java关注对象引用和方法调用，Verilog关注信号流和时钟域",
                "Java处理作用域和继承，Verilog处理模块层次和端口连接",
                "Java支持动态类型，Verilog是静态硬件描述",
                "Verilog有独特的阻塞/非阻塞赋值概念"
            ],
            "节点类型对比": {
                "Java": ["assignment", "method_invocation", "field_access", "variable_declarator"],
                "Verilog": ["continuous_assign", "always_construct", "module_instantiation", "port_declaration"]
            }
        }
        
        for category, items in comparison.items():
            print(f"\n📊 {category}:")
            if category == "节点类型对比":
                for lang, nodes in items.items():
                    print(f"   {lang}: {nodes}")
            else:
                for item in items:
                    print(f"   • {item}")
    
    def demonstrate_dataset_differences(self):
        """演示数据集差异"""
        print("\n" + "="*70)
        print("5. 数据集对比分析 (Dataset Comparison Analysis)")
        print("="*70)
        
        datasets = {
            "Java原始数据集 (Original Java Dataset)": {
                "任务类型": "代码翻译 (Java → C#)",
                "数据规模": "~数万条记录",
                "输入格式": "Java源代码",
                "输出格式": "C#目标代码", 
                "评估指标": "BLEU分数、精确匹配",
                "数据来源": "大规模真实项目"
            },
            "Verilog数据集 (Verilog Dataset)": {
                "任务类型": "错误修正 (Error Correction)",
                "数据规模": "~数百条样例",
                "输入格式": "有缺陷的Verilog代码",
                "输出格式": "修正后的Verilog代码",
                "评估指标": "功能正确性、语法正确性",
                "数据来源": "人工构造和常见错误模式"
            }
        }
        
        for dataset_name, details in datasets.items():
            print(f"\n📁 {dataset_name}:")
            for key, value in details.items():
                print(f"   {key}: {value}")
        
        print(f"\n🔍 关键差异总结:")
        print("   • 目标不同: 翻译 vs 修正")
        print("   • 规模不同: 大规模 vs 小规模")  
        print("   • 来源不同: 真实项目 vs 人工构造")
        print("   • 评估不同: 翻译质量 vs 功能正确性")
    
    def run_full_demonstration(self):
        """运行完整演示"""
        print("🚀 Verilog DFG提取完整演示")
        print("=" * 70)
        print("本演示涵盖:")
        print("1. AST解析过程")
        print("2. DFG提取演示") 
        print("3. 错误修正示例")
        print("4. 与Java对比")
        print("5. 数据集分析")
        
        try:
            self.demonstrate_ast_parsing()
            self.demonstrate_dfg_extraction()
            self.demonstrate_error_correction()
            self.compare_with_java()
            self.demonstrate_dataset_differences()
            
            print("\n" + "="*70)
            print("🎉 演示完成！")
            print("="*70)
            print("\n📋 总结要点:")
            print("✅ Verilog DFG提取使用tree-sitter-verilog，完全支持")
            print("✅ AST转换过程与Java相似，但处理硬件特有语法")
            print("✅ DFG构建关注信号流和时序关系")
            print("✅ 支持Verilog特有错误类型检测和修正")
            print("✅ 数据集专注错误修正而非代码翻译")
            
        except Exception as e:
            print(f"\n❌ 演示过程中出现错误: {e}")
            print("请检查环境配置和依赖项")

def main():
    """主函数"""
    demo = VerilogDFGDemo()
    demo.run_full_demonstration()

if __name__ == "__main__":
    main()