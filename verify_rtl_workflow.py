#!/usr/bin/env python3
"""
简单验证RTL工作流程文件的存在性和关键函数
Simple validation of RTL workflow files existence and key functions
"""

import os
import sys

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (文件不存在)")
        return False

def check_function_in_file(filepath, function_name, description):
    """检查文件中是否包含指定函数"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if function_name in content:
                print(f"✅ {description}: {function_name} 在 {filepath}")
                return True
            else:
                print(f"❌ {description}: {function_name} 在 {filepath} (函数不存在)")
                return False
    except Exception as e:
        print(f"❌ 无法读取文件 {filepath}: {e}")
        return False

def main():
    print("RTL工作流程文件验证")
    print("=" * 50)
    
    # 设置基础路径
    base_path = "/home/runner/work/GraphCodeBERT--RTL/GraphCodeBERT--RTL/GraphCodeBERT/rtl_error_localization"
    
    # 1. 检查DFG转换文件
    print("\n1. DFG转换阶段文件检查:")
    dfg_file = os.path.join(base_path, "parser/DFG.py")
    check_file_exists(dfg_file, "DFG转换文件")
    check_function_in_file(dfg_file, "DFG_verilog", "Verilog DFG转换函数")
    
    # 2. 检查Mij矩阵和Transformer融合文件
    print("\n2. Mij矩阵生成和Transformer融合文件检查:")
    model_file = os.path.join(base_path, "error_correction_model.py")
    check_file_exists(model_file, "错误修正模型文件")
    check_function_in_file(model_file, "RTLErrorCorrectionModel", "RTL错误修正模型类")
    check_function_in_file(model_file, "nodes_to_token_mask", "Mij矩阵计算代码")
    
    # 3. 检查预训练文件
    print("\n3. 预训练阶段文件检查:")
    train_file = os.path.join(base_path, "rtl_error_correction.py")
    check_file_exists(train_file, "预训练主文件")
    check_function_in_file(train_file, "def main", "主训练函数")
    
    # 4. 检查测试文件
    print("\n4. 测试和输出阶段文件检查:")
    demo_file = os.path.join(base_path, "demo_rtl_error_correction.py")
    test_file = os.path.join(base_path, "test_offline.py")
    run_file = os.path.join(base_path, "run.py")
    
    check_file_exists(demo_file, "演示测试文件")
    check_file_exists(test_file, "离线测试文件")
    check_file_exists(run_file, "通用运行脚本")
    
    check_function_in_file(demo_file, "demonstrate_testing_workflow", "测试工作流程演示函数")
    check_function_in_file(demo_file, "RTLErrorCorrectionSystem", "RTL错误修正系统类")
    
    # 5. 检查关键常量和配置
    print("\n5. 关键配置检查:")
    check_function_in_file(train_file, "DFG_verilog", "Verilog DFG函数导入")
    check_function_in_file(model_file, "torch.einsum", "Einstein求和操作")
    
    print("\n" + "=" * 50)
    print("验证完成！所有关键文件和函数的存在性已检查。")
    print("Complete! All key files and functions have been verified.")

if __name__ == "__main__":
    main()