import re

def analyze_bytecode(bytecode_str):
    """
    Analyze QuickJS bytecode to extract closure vars, local vars, and stack size metrics.
    
    Args:
        bytecode_str: String containing the bytecode dump
        
    Returns:
        Dictionary with sum and max values for closure_vars, local_vars, and stack_size
    """
    # Initialize counters
    total_closure = 0
    total_local = 0
    total_stack = 0
    max_closure = 0
    max_local = 0
    max_stack = 0
    
    # Split bytecode into function blocks
    # Each function starts with a line like: "/path/to/file:line:col: function: FunctionName"
    function_blocks = re.split(r'/[^\n]+:\d+:\d+: function:', bytecode_str)
    
    for block in function_blocks[1:]:  # Skip first empty split
        # Extract closure vars count (if present)
        # Pattern: "  closures:" followed by variable declarations
        closure_section = re.search(r'closures:\s*\n((?:\s+\d+:.*\n)*)', block)
        if closure_section:
            closure_lines = re.findall(r'^\s+\d+:', closure_section.group(1), re.MULTILINE)
            closure_count = len(closure_lines)
        else:
            closure_count = 0
        
        # Extract local vars count
        # Pattern: "  locals:" followed by variable declarations
        local_section = re.search(r'locals:\s*\n((?:\s+\d+:.*\n)*)', block)
        if local_section:
            local_lines = re.findall(r'^\s+\d+:', local_section.group(1), re.MULTILINE)
            local_count = len(local_lines)
        else:
            local_count = 0
        
        # Extract stack size
        # Pattern: "  stack_size: <number>"
        stack_match = re.search(r'stack_size:\s*(\d+)', block)
        if stack_match:
            stack_size = int(stack_match.group(1))
        else:
            stack_size = 0
        
        # Update totals
        total_closure += closure_count
        total_local += local_count
        total_stack += stack_size
        
        # Update maxes
        max_closure = max(max_closure, closure_count)
        max_local = max(max_local, local_count)
        max_stack = max(max_stack, stack_size)
    
    return {
        "sum_closure_vars": total_closure,
        "sum_local_vars": total_local,
        "sum_stack_size": total_stack,
        "max_closure_vars": max_closure,
        "max_local_vars": max_local,
        "max_stack_size": max_stack,
    }


# Example usage with your dump
if __name__ == "__main__":
    # Test with a sample
    sample_bytecode = """
/home/maverick/js_project/Iridium/tests/benchmarks/sunspider/3d:24:1: function: DrawLine
  mode: sloppy
  args: From To
  locals:
    0: var x1
    1: var x2
    2: var y1
  stack_size: 4
  byte_code_len: 271
"""
    
    result = analyze_bytecode(sample_bytecode)
    print(result)
    
    with open('dump.txt', 'r') as f:
        bytecode_content = f.read()

    result = analyze_bytecode(bytecode_content)
    print(result)