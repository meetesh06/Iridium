#!/bin/bash

# ---
# A script to extract specific opcode counts from a QJSIRIDIUM stats file.
# Usage: ./get_op_counts.sh <path_to_stats_file>
# ---

# Check if a file argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_stats_file>"
    exit 1
fi

STATS_FILE=$1

# Check if the file exists
if [ ! -f "$STATS_FILE" ]; then
    echo "Error: File not found at '$STATS_FILE'"
    exit 1
fi

# Define the full list of opcodes to search for, in the exact order requested.
# The 'OP_' prefix will be stripped before searching.
OPCODES=(
    # Column 1: Call Site Related
    "OP_call"
    "OP_call_method"
    "OP_return"
    "OP_return_undef"
    "OP_call0"
    "OP_call1"
    "OP_call2"
    "OP_call3"
    "OP_tail_call"
    "OP_tail_call_method"
    # Column 2: Control Flow Related
    "OP_if_false"
    "OP_if_true"
    "OP_goto"
    "OP_goto16"
    "OP_if_false8"
    "OP_if_true8"
    "OP_goto8"
    # Column 3: Stack Frame References
    "OP_get_loc_check"
    "OP_put_loc_check"
    "OP_put_loc_check_init"
    "OP_get_var_ref_check"
    "OP_put_var_ref_check"
    "OP_put_var_ref_check_init"
    "OP_get_loc"
    "OP_put_loc"
    "OP_set_loc"
    "OP_get_arg"
    "OP_put_arg"
    "OP_set_arg"
    "OP_get_var_ref"
    "OP_put_var_ref"
    "OP_set_var_ref"
    "OP_get_loc8"
    "OP_put_loc8"
    "OP_set_loc8"
    "OP_get_loc0"
    "OP_get_loc1"
    "OP_get_loc2"
    "OP_get_loc3"
    "OP_put_loc0"
    "OP_put_loc1"
    "OP_put_loc2"
    "OP_put_loc3"
    "OP_get_arg0"
    "OP_get_arg1"
    "OP_get_arg2"
    "OP_get_arg3"
    "OP_put_arg0"
    "OP_put_arg1"
    "OP_put_arg2"
    "OP_put_arg3"
    "OP_get_var_ref0"
    "OP_get_var_ref1"
    "OP_get_var_ref2"
    "OP_get_var_ref3"
    "OP_put_var_ref0"
    "OP_put_var_ref1"
    "OP_put_var_ref2"
    "OP_put_var_ref3"
    "OP_get_loc0_loc1"
    "OP_set_arg0"
    "OP_set_arg1"
    "OP_set_arg2"
    "OP_set_arg3"
    "OP_set_var_ref0"
    "OP_set_var_ref1"
    "OP_set_var_ref2"
    "OP_set_var_ref3"
    "OP_set_loc0"
    "OP_set_loc1"
    "OP_set_loc2"
    "OP_set_loc3"
    # Column 4: Field Operations
    "OP_get_array_el"
    "OP_to_propkey"
    "OP_to_propkey2"
    "OP_put_field"
    "OP_get_field"
    "OP_get_array_el2"
    "OP_get_field2"
    "OP_get_length"
)

# Extract the relevant section from the file to optimize searching
OPCODE_DATA=$(awk '/=== Opcode Frequencies ===/{flag=1;next}/=== Breakdown per function ===/{flag=0}flag' "$STATS_FILE")

# Loop through the defined list of opcodes
for op_with_prefix in "${OPCODES[@]}"; do
    # Remove the "OP_" prefix to match the format in the file
    op_name="${op_with_prefix#OP_}"

    # Search for the exact opcode name at the beginning of a line
    # The output of grep will be like "call_method: 761"
    line=$(echo "$OPCODE_DATA" | grep "^${op_name}:")

    # Extract the number from the line
    if [ -n "$line" ]; then
        count=$(echo "$line" | awk '{print $2}')
    else
        count=0
    fi

    # Print the final count
    echo "$count"
done