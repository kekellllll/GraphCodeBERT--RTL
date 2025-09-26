# RTL模块完整流程文件指南
# Complete File Guide for RTL Module Workflow

## 问题解答
针对RTL模块将代码运行完整的流程，以下是各个阶段对应的具体文件：

## 1. 代码转换成DFG的文件
**文件位置**: `GraphCodeBERT/rtl_error_localization/parser/DFG.py`
- **关键函数**: `DFG_verilog(root_node, index_to_code, states)` (第1184-1347行)
- **功能**: 将Verilog RTL代码解析为数据流图(DFG)
- **输入**: Verilog代码的语法树节点
- **输出**: DFG元组列表，格式为 `(变量名, 索引, 关系类型, 依赖变量, 依赖索引)`

## 2. 生成Mij矩阵的文件  
**文件位置**: `GraphCodeBERT/rtl_error_localization/error_correction_model.py`
- **关键代码**: 第78-82行的 `forward()` 方法
- **功能**: 计算Mij矩阵，实现DFG节点与代码标记的融合
- **核心实现**:
```python
nodes_to_token_mask = nodes_mask[:,:,None] & token_mask[:,None,:] & attn_mask.bool()
nodes_to_token_mask = nodes_to_token_mask.float() / (nodes_to_token_mask.sum(-1, keepdim=True).float() + 1e-10)
avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
```

## 3. 融合Transformer的文件
**文件位置**: `GraphCodeBERT/rtl_error_localization/error_correction_model.py`
- **关键代码**: 第84-89行
- **功能**: 将融合了DFG信息的嵌入输入到Transformer编码器
- **架构**: 使用RoBERTa编码器 + Transformer解码器的seq2seq架构

## 4. 完成预训练的文件
**文件位置**: `GraphCodeBERT/rtl_error_localization/rtl_error_correction.py`
- **关键函数**: `main()` 函数中的训练循环
- **功能**: 使用正确的RTL代码+注释+DFG进行模型预训练
- **训练数据**: 正确的Verilog代码配对 (源码→目标码)
- **损失函数**: 交叉熵损失 + 序列生成损失

## 5. 进行测试和输出的文件

### 5.1 演示测试文件
**文件位置**: `GraphCodeBERT/rtl_error_localization/demo_rtl_error_correction.py`
- **关键函数**: `demonstrate_testing_workflow()`, `main()`
- **功能**: 演示完整的错误检测和修正流程
- **输入**: 有缺陷的RTL代码
- **输出**: 缺陷位置 + 修正后的代码

### 5.2 离线测试文件  
**文件位置**: `GraphCodeBERT/rtl_error_localization/test_offline.py`
- **功能**: 批量测试和性能评估
- **评估指标**: BLEU分数、准确率等

### 5.3 通用运行脚本
**文件位置**: `GraphCodeBERT/rtl_error_localization/run.py`
- **功能**: 可配置的训练和测试入口点

## 完整执行流程图

```
输入RTL代码
    ↓
parser/DFG.py (DFG转换)
    ↓  
error_correction_model.py (Mij矩阵生成 + Transformer融合)
    ↓
rtl_error_correction.py (预训练)
    ↓
demo_rtl_error_correction.py / test_offline.py (测试输出)
```

## 快速运行指南

1. **预训练模型**:
```bash
python rtl_error_correction.py --do_train --model_type roberta
```

2. **运行演示**:
```bash
python demo_rtl_error_correction.py
```

3. **离线测试**:
```bash
python test_offline.py --model_path ./output/pytorch_model.bin
```

## 关键技术点

- **DFG提取**: 使用tree-sitter解析Verilog语法，提取数据依赖关系
- **Mij矩阵**: 实现多模态信息融合的核心组件  
- **位置编码**: DFG节点(0), 注释(1), 代码(≥2)的分层编码策略
- **序列生成**: 基于编码器-解码器架构的错误修正生成

通过这些文件的协同工作，实现了从RTL代码输入到错误检测和修正输出的完整pipeline。