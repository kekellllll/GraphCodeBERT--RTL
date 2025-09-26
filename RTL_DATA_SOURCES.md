# RTL模型训练数据源说明 / RTL Model Training Data Sources

## 中文说明

### 🔍 用户问题回答
**问题**: 这个RTL模型训练没有数据么，这些测试的输出是训练的结果还是你自己添加的？如果有数据集请告诉我具体在哪里？

**回答**:

### ✅ 当前数据状况
1. **现有数据类型**: 目前项目中使用的是**演示样本数据**，不是真实的大规模训练数据集
2. **数据位置**: 硬编码在 `GraphCodeBERT/rtl_error_localization/rtl_error_correction.py` 的 `create_sample_data()` 函数中
3. **数据规模**: 仅有3个基础示例，用于功能演示和代码验证
4. **测试输出**: 当前的测试输出是基于这些预定义样本，**不是**真实训练的结果

### 📊 具体数据内容
```python
# 位置: GraphCodeBERT/rtl_error_localization/rtl_error_correction.py:255-274
def create_sample_data():
    examples = [
        {
            'buggy_code': 'module test(input a, output b); assign b = a + 1; endmodule',
            'correct_code': 'module test(input a, output b); assign b = a; endmodule',
            'comments': 'Simple wire connection module'
        },
        # ... 另外2个类似示例
    ]
```

### 🎯 项目现状
- **目的**: 这是一个**概念验证和框架实现**，展示如何将GraphCodeBERT适配到RTL错误修正任务
- **实现状态**: 完整的模型架构和处理流程，但缺少大规模训练数据
- **功能验证**: 所有功能都可以运行，但基于小规模样本数据

### 📁 数据集需求和创建指南

#### 真实训练数据集应包含:
1. **错误RTL代码**: 包含各种语法和逻辑错误的Verilog代码
2. **正确RTL代码**: 对应的修正版本
3. **注释信息**: 代码功能说明
4. **数据流图**: 自动提取的DFG信息
5. **错误位置**: 精确的错误定位信息

#### 推荐的数据集大小:
- **训练集**: 至少10,000个错误-修正代码对
- **验证集**: 2,000个代码对
- **测试集**: 2,000个代码对

---

## English Explanation

### 🔍 User Question Response
**Question**: Does this RTL model training have no data? Are these test outputs the result of training or did you add them yourself? If there are datasets, please tell me specifically where they are?

**Answer**:

### ✅ Current Data Status
1. **Data Type**: The project currently uses **demonstration sample data**, not real large-scale training datasets
2. **Data Location**: Hard-coded in the `create_sample_data()` function in `GraphCodeBERT/rtl_error_localization/rtl_error_correction.py`
3. **Data Scale**: Only 3 basic examples for functionality demonstration and code validation
4. **Test Output**: Current test outputs are based on these predefined samples, **NOT** real training results

### 📊 Specific Data Content
The sample data consists of 3 hardcoded examples:
- Module with unnecessary arithmetic operation
- Always block with incorrect assignment
- Logic expression missing parentheses

### 🎯 Project Status
- **Purpose**: This is a **proof-of-concept and framework implementation** showing how to adapt GraphCodeBERT for RTL error correction
- **Implementation Status**: Complete model architecture and processing pipeline, but lacking large-scale training data
- **Functionality**: All functions work, but based on small sample data

### 📁 Dataset Requirements and Creation Guide

#### Real Training Dataset Should Include:
1. **Buggy RTL Code**: Verilog code with various syntax and logic errors
2. **Correct RTL Code**: Corresponding corrected versions
3. **Comments**: Code functionality descriptions
4. **Data Flow Graphs**: Automatically extracted DFG information
5. **Error Locations**: Precise error localization information

#### Recommended Dataset Size:
- **Training Set**: At least 10,000 error-correction code pairs
- **Validation Set**: 2,000 code pairs
- **Test Set**: 2,000 code pairs

---

## 🛠️ 如何创建真实数据集 / How to Create Real Datasets

### 方法一: 手动标注 / Method 1: Manual Annotation
```bash
# 创建数据集目录
mkdir -p datasets/rtl_error_correction/{train,valid,test}

# 数据格式示例 (datasets/rtl_error_correction/train/sample.jsonl)
{"buggy_code": "module ...", "correct_code": "module ...", "comments": "...", "error_type": "syntax"}
```

### 方法二: 自动生成 / Method 2: Automatic Generation
```bash
# 运行数据生成工具
python tools/generate_rtl_dataset.py --output datasets/rtl_error_correction --size 10000
```

### 方法三: 现有数据集适配 / Method 3: Existing Dataset Adaptation
- 寻找现有的Verilog代码错误数据集
- 适配到项目要求的格式
- 添加DFG提取和错误位置标注

---

## 📝 使用现有样本数据进行测试 / Testing with Current Sample Data

```bash
# 运行演示 (基于样本数据)
cd GraphCodeBERT/rtl_error_localization
python demo_rtl_error_correction.py

# 运行离线测试
python test_offline.py

# 查看样本数据
python -c "from rtl_error_correction import create_sample_data; print(create_sample_data())"
```

---

## ⚠️ 重要说明 / Important Notes

### 中文
- **当前版本**: 这是一个研究原型，展示技术可行性
- **数据状况**: 需要用户根据具体需求创建真实的训练数据集
- **训练建议**: 建议收集至少10,000个RTL错误-修正对进行真实训练
- **功能完整性**: 所有必要的代码和工具都已提供，只需要添加真实数据

### English
- **Current Version**: This is a research prototype demonstrating technical feasibility
- **Data Status**: Users need to create real training datasets based on specific requirements
- **Training Recommendation**: Recommend collecting at least 10,000 RTL error-correction pairs for real training
- **Functionality**: All necessary code and tools are provided, only need to add real data

---

## 📞 获取帮助 / Get Help

如果需要协助创建数据集或有其他问题，请：
If you need help creating datasets or have other questions, please:

1. 查看项目文档 / Check project documentation
2. 运行现有示例 / Run existing examples
3. 参考其他类似项目的数据集格式 / Refer to similar projects' dataset formats