#!/usr/bin/env python3
"""
RTL模型数据状况检查工具 / RTL Model Data Status Checker

该脚本检查并显示当前RTL模型的数据状况，包括：
This script checks and displays the current data status of RTL model, including:
- 现有样本数据 / Existing sample data
- 数据集位置 / Dataset locations  
- 训练数据需求 / Training data requirements
- 生成工具状态 / Generation tool status

使用方法 / Usage:
python check_data_status.py
"""

import os
import sys
import json
import glob
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_sample_data():
    """检查内置样本数据"""
    print("🔍 检查内置样本数据 / Checking Built-in Sample Data")
    print("=" * 60)
    
    try:
        # Import the sample data creation function
        rtl_path = os.path.join(os.path.dirname(__file__), 'GraphCodeBERT', 'rtl_error_localization')
        sys.path.insert(0, rtl_path)
        from rtl_error_correction import create_sample_data
        
        samples = create_sample_data()
        print(f"📊 内置样本数量 / Built-in Samples: {len(samples)}")
        print(f"📍 代码位置 / Code Location: GraphCodeBERT/rtl_error_localization/rtl_error_correction.py")
        print(f"🔧 函数名称 / Function Name: create_sample_data()")
        
        print(f"\n📝 样本详情 / Sample Details:")
        for i, sample in enumerate(samples, 1):
            print(f"\n  样本 {i} / Sample {i}:")
            print(f"    错误代码 / Buggy:   {sample['buggy_code']}")
            print(f"    正确代码 / Correct: {sample['correct_code']}")
            print(f"    注释 / Comments: {sample['comments']}")
        
        return len(samples)
    except Exception as e:
        print(f"❌ 无法加载样本数据 / Cannot load sample data: {e}")
        return 0

def check_generated_datasets():
    """检查生成的数据集"""
    print("\n🗂️  检查生成的数据集 / Checking Generated Datasets")
    print("=" * 60)
    
    # Check common dataset locations
    dataset_locations = [
        'datasets/rtl_error_correction',
        'datasets/rtl_training', 
        'datasets/sample_rtl_data',
        '../datasets/rtl_error_correction',
        '../datasets/rtl_training',
        '../datasets/sample_rtl_data'
    ]
    
    found_datasets = []
    for location in dataset_locations:
        if os.path.exists(location):
            files = glob.glob(os.path.join(location, "*.json*"))
            if files:
                found_datasets.append((location, files))
    
    if found_datasets:
        print("✅ 发现生成的数据集 / Found Generated Datasets:")
        for location, files in found_datasets:
            print(f"\n📁 位置 / Location: {location}")
            for file in files:
                try:
                    if file.endswith('.json'):
                        with open(file, 'r') as f:
                            data = json.load(f)
                            count = len(data) if isinstance(data, list) else 1
                    elif file.endswith('.jsonl'):
                        with open(file, 'r') as f:
                            count = sum(1 for _ in f)
                    else:
                        count = "Unknown"
                    
                    print(f"  📄 {os.path.basename(file)}: {count} 样本")
                except Exception as e:
                    print(f"  📄 {os.path.basename(file)}: 无法读取 / Cannot read")
        
        return len(found_datasets)
    else:
        print("⚠️  未找到生成的数据集 / No Generated Datasets Found")
        print("💡 提示 / Tip: 使用以下命令生成数据集 / Use following command to generate:")
        print("   python tools/generate_rtl_dataset.py --output datasets/rtl_training --size 1000")
        return 0

def check_tools():
    """检查可用工具"""
    print("\n🛠️  检查可用工具 / Checking Available Tools") 
    print("=" * 60)
    
    tools_dir = "tools"
    if not os.path.exists(tools_dir):
        tools_dir = "../tools" 
    
    if os.path.exists(tools_dir):
        dataset_generator = os.path.join(tools_dir, "generate_rtl_dataset.py")
        if os.path.exists(dataset_generator):
            print(f"✅ 数据集生成工具 / Dataset Generator: {dataset_generator}")
            print(f"🚀 使用方法 / Usage:")
            print(f"   python {dataset_generator} --help")
            print(f"   python {dataset_generator} --output datasets/rtl_training --size 1000")
            return True
        else:
            print(f"❌ 数据集生成工具未找到 / Dataset Generator Not Found")
            return False
    else:
        print(f"❌ 工具目录未找到 / Tools Directory Not Found: {tools_dir}")
        return False

def show_training_requirements():
    """显示训练数据需求"""
    print("\n📋 训练数据需求 / Training Data Requirements")
    print("=" * 60)
    
    requirements = {
        "研究原型 / Research Prototype": "100-1,000 样本",
        "基础模型 / Basic Model": "1,000-5,000 样本", 
        "生产级模型 / Production Model": "10,000+ 样本",
        "高质量模型 / High Quality Model": "50,000+ 样本"
    }
    
    for level, count in requirements.items():
        print(f"  {level}: {count}")
    
    print(f"\n📊 推荐数据格式 / Recommended Data Format:")
    print(f"  JSONL格式，每行包含 / JSONL format, each line contains:")
    print(f"  - buggy_code: 错误的RTL代码")
    print(f"  - correct_code: 正确的RTL代码") 
    print(f"  - comments: 功能说明")
    print(f"  - error_type: 错误类型")

def show_current_status_summary():
    """显示当前状态总结"""
    print("\n📈 当前状态总结 / Current Status Summary")
    print("=" * 60)
    
    print("🔴 当前阶段 / Current Stage: 演示和概念验证 / Demo & Proof of Concept")
    print("📊 可用数据 / Available Data: 3个硬编码样本 / 3 hardcoded samples")
    print("🎯 用途 / Purpose: 功能演示，架构验证 / Feature demo, architecture validation")
    print("⚠️  限制 / Limitations: 不适用于实际训练 / Not suitable for real training")
    
    print(f"\n✅ 下一步行动 / Next Steps:")
    print(f"  1. 使用数据生成工具创建更多样本 / Use generation tool for more samples")
    print(f"  2. 收集真实的RTL错误数据 / Collect real RTL error data")
    print(f"  3. 手动标注错误-修正对 / Manually annotate error-correction pairs")
    print(f"  4. 验证数据质量 / Validate data quality")
    print(f"  5. 开始真实训练 / Start real training")

def main():
    print("🚀 RTL模型数据状况检查报告 / RTL Model Data Status Report")
    print("=" * 80)
    
    # Check built-in samples
    sample_count = check_sample_data()
    
    # Check generated datasets 
    dataset_count = check_generated_datasets()
    
    # Check available tools
    tools_available = check_tools()
    
    # Show training requirements
    show_training_requirements()
    
    # Show current status summary
    show_current_status_summary()
    
    # Final recommendation
    print(f"\n💡 建议 / Recommendations:")
    if sample_count > 0 and tools_available:
        print(f"✅ 基础设施完备，可以开始数据生成和收集")
        print(f"✅ Infrastructure ready, can start data generation and collection")
    else:
        print(f"⚠️  需要修复基础设施问题后再进行数据准备") 
        print(f"⚠️  Need to fix infrastructure issues before data preparation")
    
    print(f"\n📚 详细信息参考 / For detailed information see:")
    print(f"   - RTL_DATA_SOURCES.md (完整数据源说明)")
    print(f"   - tools/generate_rtl_dataset.py (数据生成工具)")
    print(f"   - GraphCodeBERT/rtl_error_localization/README.md (使用指南)")

if __name__ == "__main__":
    main()