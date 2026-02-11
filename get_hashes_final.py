import hashlib, os
files = ['artifacts/local_types.jsonl', 'artifacts/extendible_witnesses.jsonl', 'artifacts/obstruction_witnesses.jsonl']
with open('hashes_output.txt', 'w', encoding='utf-8') as f_out:
    for f in files:
        if os.path.exists(f):
            data = open(f, 'rb').read()
            h = hashlib.sha256(data).hexdigest()
            f_out.write(f"{f}:{h}\n")
        else:
            f_out.write(f"{f}:MISSING\n")
