import os
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data")
input_path = os.path.join(data_dir, "find_orientation.txt")

try:
    content = open(input_path, 'rb').read().decode('utf-16', 'ignore')
    for kind in ['C2', 'C4', 'PINCH']:
        m = re.search(fr'SUCCESS for {kind}!.*?x_rot: (.*?)\s*y_rot: (.*?)\s', content, re.DOTALL)
        if m:
            print(f"{kind}: x_rot={m.group(1)}, y_rot={m.group(2)}")
        else:
            print(f"{kind}: FAILED")
except Exception as e:
    print(f"Error: {e}")
