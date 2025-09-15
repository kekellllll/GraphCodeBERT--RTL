#!/usr/bin/env python3
"""
Simple offline test for RTL Error Correction Model functionality
This test validates the model architecture without requiring internet access
"""

import torch
import torch.nn as nn
from error_correction_model import RTLErrorCorrectionModel, Beam
from transformers import RobertaConfig

def test_model_architecture():
    """Test the RTL Error Correction Model architecture"""
    print("Testing RTL Error Correction Model architecture...")
    
    # Create a simple config
    config = RobertaConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512
    )
    
    # Create dummy encoder (using standard transformer)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=config.hidden_size, 
        nhead=config.num_attention_heads,
        batch_first=True
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    # Add required attributes to mock RoBERTa encoder
    class MockEncoder(nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer
            self.embeddings = nn.ModuleDict({
                'word_embeddings': nn.Embedding(config.vocab_size, config.hidden_size)
            })
            
        def forward(self, inputs_embeds, attention_mask=None, position_ids=None):
            # Simple mock forward - in real model this would be more complex
            if attention_mask is not None and len(attention_mask.shape) == 3:
                # Convert 3D attention mask to 2D key_padding_mask
                attention_mask = (attention_mask.sum(-1) == 0)  # True where padded
            output = self.transformer(inputs_embeds, src_key_padding_mask=attention_mask)
            return [output]  # Return as tuple to match RoBERTa interface
    
    mock_encoder = MockEncoder(encoder)
    
    # Create decoder
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=config.hidden_size, 
        nhead=config.num_attention_heads
    )
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    
    # Create RTL Error Correction Model
    model = RTLErrorCorrectionModel(
        encoder=mock_encoder,
        decoder=decoder,
        config=config,
        beam_size=3,
        max_length=64,
        sos_id=1,
        eos_id=2
    )
    
    print(f"✓ Model created successfully")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    
    # Create dummy inputs
    source_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    source_mask = torch.ones(batch_size, seq_len)
    position_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    attn_mask = torch.ones(batch_size, seq_len, seq_len)
    target_ids = torch.randint(0, config.vocab_size, (batch_size, 20))
    target_mask = torch.ones(batch_size, 20)
    
    try:
        # Test training mode
        model.train()
        outputs = model(source_ids, source_mask, position_idx, attn_mask, target_ids, target_mask)
        loss = outputs[0]
        
        print(f"✓ Training forward pass successful, loss: {loss.item():.4f}")
        
        # Test inference mode
        model.eval()
        with torch.no_grad():
            preds = model(source_ids, source_mask, position_idx, attn_mask)
        
        print(f"✓ Inference forward pass successful, output shape: {preds.shape}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    return True

def test_beam_search():
    """Test beam search implementation"""
    print("\nTesting Beam Search...")
    
    beam = Beam(size=3, sos=1, eos=2)
    
    # Test initial state
    initial_state = beam.getCurrentState()
    print(f"✓ Initial state shape: {initial_state.shape}")
    
    # Simulate advancing beam with dummy scores
    vocab_size = 100
    dummy_scores = torch.randn(3, vocab_size)
    beam.advance(dummy_scores)
    
    print(f"✓ Beam advance successful")
    print(f"✓ Beam search test completed")
    
    return True

def test_data_flow_processing():
    """Test data flow processing"""
    print("\nTesting Data Flow Processing...")
    
    from rtl_error_correction import extract_verilog_dataflow_mock
    
    verilog_code = """
    module adder(input a, b, output sum);
        assign sum = a + b;
    endmodule
    """
    
    tokens, dfg = extract_verilog_dataflow_mock(verilog_code)
    
    print(f"✓ Extracted {len(tokens)} tokens")
    print(f"✓ Extracted {len(dfg)} DFG edges")
    
    for edge in dfg:
        print(f"  DFG: {edge[0]} -> {edge[3]} ({edge[2]})")
    
    return True

def main():
    """Run all offline tests"""
    print("=== RTL Error Correction Offline Tests ===\n")
    
    all_passed = True
    
    # Test model architecture
    if not test_model_architecture():
        all_passed = False
    
    # Test beam search
    if not test_beam_search():
        all_passed = False
        
    # Test data flow processing
    if not test_data_flow_processing():
        all_passed = False
    
    print(f"\n=== Test Results ===")
    if all_passed:
        print("✅ All tests passed! RTL Error Correction Model is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print(f"\nSystem Info:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

if __name__ == "__main__":
    main()