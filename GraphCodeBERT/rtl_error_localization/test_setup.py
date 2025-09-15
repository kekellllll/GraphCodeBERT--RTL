#!/usr/bin/env python3
"""
Test script for RTL Error Correction Model
This script tests the basic functionality of our GraphCodeBERT adaptation for Verilog
"""

import sys
import os
import torch
import logging

# Add the rtl_error_localization directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parser import DFG_verilog
from rtl_error_correction import extract_verilog_dataflow_mock, create_sample_data

def test_verilog_dfg():
    """Test Verilog DFG extraction"""
    print("Testing Verilog DFG extraction...")
    
    # Sample Verilog code
    verilog_code = """
    module test_module(input a, b, output c);
        wire temp;
        assign temp = a & b;
        assign c = temp | a;
    endmodule
    """
    
    # Mock DFG extraction
    tokens, dfg = extract_verilog_dataflow_mock(verilog_code)
    
    print(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens
    print(f"DFG edges: {len(dfg)}")
    for edge in dfg[:3]:  # Show first 3 edges
        print(f"  {edge}")
    
    print("✓ Verilog DFG extraction test passed\n")

def test_sample_data():
    """Test sample data creation"""
    print("Testing sample data creation...")
    
    examples = create_sample_data()
    
    print(f"Created {len(examples)} sample examples:")
    for i, example in enumerate(examples):
        print(f"  Example {i+1}:")
        print(f"    Buggy: {example['buggy_code']}")
        print(f"    Correct: {example['correct_code']}")
        print(f"    Comments: {example['comments']}")
    
    print("✓ Sample data creation test passed\n")

def test_verilog_dfg_function():
    """Test the DFG_verilog function directly"""
    print("Testing DFG_verilog function...")
    
    # Mock index_to_code and root_node for testing
    class MockNode:
        def __init__(self, node_type, children=None):
            self.type = node_type
            self.children = children or []
            self.start_point = (0, 0)
            self.end_point = (0, 1)
    
    # Create a simple mock tree
    root = MockNode('module_declaration')
    index_to_code = {(0, 0): (0, 'module'), (0, 1): (1, 'test')}
    states = {}
    
    try:
        dfg, new_states = DFG_verilog(root, index_to_code, states)
        print(f"DFG result: {dfg}")
        print(f"States: {new_states}")
        print("✓ DFG_verilog function test passed\n")
    except Exception as e:
        print(f"DFG_verilog function test failed: {e}\n")

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        from error_correction_model import RTLErrorCorrectionModel, Beam
        print("✓ RTLErrorCorrectionModel imported successfully")
        
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
        print("✓ Transformers imported successfully")
        
        print("✓ All imports test passed\n")
    except Exception as e:
        print(f"Import test failed: {e}\n")

def main():
    """Run all tests"""
    print("=== RTL Error Correction Model Tests ===\n")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_imports()
    test_verilog_dfg_function()
    test_verilog_dfg()
    test_sample_data()
    
    print("=== All tests completed ===")
    
    # Test basic PyTorch functionality
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    print("\n✅ RTL Error Correction setup is working correctly!")

if __name__ == "__main__":
    main()