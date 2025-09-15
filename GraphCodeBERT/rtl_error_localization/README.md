# GraphCodeBERT for RTL-Verilog Code Error Localization

This implementation adapts GraphCodeBERT for RTL (Verilog/SystemVerilog) code error localization and correction. The model takes buggy Verilog code along with comments and data flow graph information as multimodal input and outputs corrected code.

## Features

### ✅ Implemented
- **Verilog DFG Parser**: Custom Data Flow Graph extraction for Verilog syntax
- **Error Correction Model**: Sequence-to-sequence model for code correction
- **Multimodal Input Processing**: Handles source code + comments + DFG nodes
- **GraphCodeBERT Architecture**: Maintains Mij matrix fusion for DFG integration
- **Beam Search Generation**: Generates corrected code using beam search

### 🚧 Architecture Details

The implementation includes:

1. **DFG_verilog Function**: Extracts data flow graphs from Verilog code
   - Handles assignments (`assign`, `<=`, `=`)
   - Processes always blocks and control structures
   - Creates edges for variable dependencies

2. **RTLErrorCorrectionModel**: Extends GraphCodeBERT's encoder-decoder architecture
   - Encoder: GraphCodeBERT with DFG fusion
   - Decoder: Transformer decoder for sequence generation
   - Error confidence scoring (for future enhancements)

3. **Training Pipeline**: End-to-end training for error correction
   - Converts examples to features with DFG information
   - Handles position encoding (0 for DFG nodes, 2+ for code tokens)
   - Supports batch training with attention masking

## Usage

### Training
```bash
cd rtl_error_localization
python rtl_error_correction.py \
    --do_train \
    --model_name_or_path microsoft/graphcodebert-base \
    --output_dir ./saved_models \
    --max_source_length 256 \
    --max_target_length 128 \
    --train_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3
```

### Testing Setup
```bash
python test_setup.py
```

### Inference (Future)
```bash
python rtl_error_correction.py \
    --do_test \
    --model_name_or_path ./saved_models \
    --test_filename your_test_data.jsonl
```

## Data Format

Expected input format for training data:
```json
{
    "buggy_code": "module test(input a, output b); assign b = a + 1; endmodule",
    "correct_code": "module test(input a, output b); assign b = a; endmodule", 
    "comments": "Simple wire connection module"
}
```

## Model Architecture

The model maintains GraphCodeBERT's key innovations:

1. **Position Encoding**:
   - Position 0: DFG nodes
   - Position 1: Comments/documentation
   - Position 2+: Code tokens

2. **Attention Mechanism**:
   - DFG nodes can attend to related code tokens
   - Code tokens can attend to relevant DFG nodes
   - Full multimodal attention matrix

3. **DFG-Code Fusion**:
   - Node embeddings averaged with corresponding code tokens
   - Mij matrix for structure-aware attention

## Pretraining Tasks (Future Work)

To fully match the original GraphCodeBERT, implement:

1. **Masked Language Modeling**: Predict masked code tokens
2. **DFG Edge Prediction**: Predict masked node-to-node relationships  
3. **Code-DFG Alignment**: Predict masked node-to-code relationships

## File Structure

```
rtl_error_localization/
├── error_correction_model.py    # RTL error correction model
├── rtl_error_correction.py      # Training/inference pipeline
├── parser/                      # Tree-sitter parsers
│   ├── DFG.py                  # DFG extraction (includes DFG_verilog)
│   ├── __init__.py
│   └── ...
├── test_setup.py               # Verification tests
└── README.md                   # This file
```

## Current Limitations

1. **Tree-sitter Verilog**: Currently uses mock DFG extraction due to tree-sitter version compatibility
2. **Training Data**: Uses synthetic sample data for demonstration
3. **Evaluation Metrics**: Need to implement BLEU/CodeBLEU for Verilog
4. **Pretraining**: Currently only supports fine-tuning, not pretraining from scratch

## Next Steps

1. **Fix Tree-sitter Verilog**: Resolve version compatibility for proper AST parsing
2. **Real Dataset**: Integrate with actual Verilog error correction datasets
3. **Evaluation**: Implement comprehensive evaluation metrics
4. **Pretraining**: Add mask prediction tasks for Verilog
5. **Error Localization**: Add explicit error location prediction

## Dependencies

- torch >= 1.7.0
- transformers >= 4.0.0
- tree_sitter >= 0.20.0
- numpy
- tqdm

## References

- [GraphCodeBERT: Pre-training Code Representations with Data Flow](https://openreview.net/forum?id=jLoC4ez43PZ)
- [Tree-sitter Verilog Grammar](https://github.com/tree-sitter/tree-sitter-verilog)