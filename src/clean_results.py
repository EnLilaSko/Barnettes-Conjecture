import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data")
input_path = os.path.join(data_dir, "find_orientation.txt")
output_path = os.path.join(data_dir, "final_results.txt")

try:
    content = open(input_path, 'rb').read().decode('utf-16', 'ignore')
    with open(output_path, 'w') as f:
        for kind in ['C2', 'C4', 'PINCH']:
            match = re.search(fr'SUCCESS for {kind}!.*?(Mapping:.*?)(x neighbors:.*?)(x rot:.*?)(y rot:.*?)\n', content, re.DOTALL)
            if match:
                f.write(f"--- {kind} ---\n")
                f.write(match.group(1).strip() + "\n")
                f.write(match.group(2).strip() + "\n")
                f.write(match.group(3).strip() + "\n")
                f.write(match.group(4).strip() + "\n\n")
            else:
                f.write(f"--- {kind} FAILED ---\n\n")
except Exception as e:
    with open(output_path, 'w') as f:
        f.write(f"Error: {e}")
