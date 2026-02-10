.PHONY: all verify-local verify-logic verify-base verify-trace check-n48 clean

all: verify-logic verify-base verify-local check-n48

verify-local:
	python src/enumerate_local_types.py --ncap 14 --nmax 14

verify-logic:
	python src/verify_gadgets.py

verify-base:
	python tests/verify_base_cases.py

verify-trace:
	python src/check_trace.py artifacts/traces/trace_n48.jsonl
	python src/check_trace.py artifacts/traces/trace_n128.jsonl

check-n48:
	python src/check_trace.py artifacts/traces/trace_n48.jsonl

clean:
	rm -f artifacts/local_types.jsonl artifacts/extendible_witnesses.jsonl artifacts/obstruction_witnesses.jsonl
