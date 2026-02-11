import hashlib, os
files = ['artifacts/local_types.jsonl', 'artifacts/extendible_witnesses.jsonl', 'artifacts/obstruction_witnesses.jsonl']
for f in files:
    if os.path.exists(f):
        data = open(f, 'rb').read()
        h = hashlib.sha256(data).hexdigest()
        print(f"FILE: {f}")
        print(f"HASH: {h}")
        print(f"SIZE: {len(data)} bytes")
        print("-" * 20)
    else:
        print(f"MISSING: {f}")
