import os
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data")
input_path = os.path.join(data_dir, "find_orientation.txt")

try:
    content = open(input_path, 'rb').read().decode('utf-16', 'ignore')
    for kind in ['C2', 'C4', 'PINCH']:
        # Find the block for this kind
        # SUCCESS!
        #   Mapping: ...
        #   x neighbors: ...
        #   x rot: ...
        #   y rot: ...
        block_match = re.search(fr'SUCCESS for {kind}!.*?(Mapping:.*?)(x neighbors:.*?)(x rot:.*?)(y rot:.*?)\n', content, re.DOTALL)
        if block_match:
            print(f"--- {kind} ---")
            print(block_match.group(1).strip())
            print(block_match.group(2).strip())
            print(block_match.group(3).strip())
            print(block_match.group(4).strip())
        else:
            print(f"--- {kind} FAILED ---")
except Exception as e:
    print(f"Error: {e}")
