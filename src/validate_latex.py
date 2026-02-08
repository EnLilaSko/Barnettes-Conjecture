import re
import sys
import os

def validate_latex(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find all \label{...}
    label_pattern = re.compile(r'\\label\{([^}]*)\}')
    # Regex to find all \ref{...}, \secref{...}, etc.
    # Note: we specifically look for \ref, \secref, \eqref, \pageref
    ref_pattern = re.compile(r'\\(ref|secref|eqref|pageref)\{([^}]*)\}')

    labels = label_pattern.findall(content)
    refs = ref_pattern.findall(content)

    label_counts = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1

    duplicate_labels = [l for l, count in label_counts.items() if count > 1]
    
    unique_labels = set(labels)
    missing_labels = set()
    for ref_type, ref_name in refs:
        # Ignore macro parameters like #1
        if ref_name not in unique_labels and not re.match(r'#\d+', ref_name):
            missing_labels.add(ref_name)

    print(f"--- LaTeX Consistency Audit: {filepath} ---")
    
    if duplicate_labels:
        print(f"\n[!] ERROR: Found {len(duplicate_labels)} duplicate labels:")
        for l in duplicate_labels:
            print(f"    - {l} (appears {label_counts[l]} times)")
    else:
        print("\n[+] No duplicate labels found.")

    if missing_labels:
        print(f"\n[!] ERROR: Found {len(missing_labels)} undefined references:")
        for l in sorted(missing_labels):
            print(f"    - {l}")
    else:
        print("\n[+] All references resolved to labels.")

    print("\n--- Audit Complete ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_latex.py <file.tex>")
    else:
        validate_latex(sys.argv[1])
