import hashlib, os
files = ['artifacts/local_types.jsonl', 'artifacts/extendible_witnesses.jsonl', 'artifacts/obstruction_witnesses.jsonl']
for f in files:
    if os.path.exists(f):
        data = open(f, 'rb').read()
        print(f"{f}: {hashlib.md5(data).hexdigest()} ({len(data)} bytes)")
    else:
        print(f"{f}: MISSING")
