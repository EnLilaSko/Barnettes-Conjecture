.PHONY: all verify-local verify-admissibility verify-logic verify-base verify-trace check-n48 clean

all: verify-logic verify-base verify-local verify-admissibility check-n48

verify-local:
	# Theorem 41 witness production.
	# If you have a precomputed local-types file that already includes a rotation system,
	# pass it via --in-types. Otherwise, omit --in-types to regenerate types.
	py src/enumerate_local_types.py --ncap 32

verify-admissibility:
	# Produce admissibility witnesses/obstructions for each rooted local type.
	py src/produce_admissibility_witnesses.py \
		--in-witnesses artifacts/extendible_witnesses.jsonl \
		--out-witnesses artifacts/admissibility_witnesses.jsonl \
		--out-obstructions artifacts/admissibility_obstructions.jsonl

verify-logic:
	py src/verify_gadgets.py

verify-base:
	py tests/verify_base_cases.py

verify-trace:
	py src/check_trace.py

check-n48:
	py src/check_trace.py artifacts/traces/trace_n48.jsonl

clean:
	rm -f artifacts/local_types.jsonl artifacts/extendible_witnesses.jsonl artifacts/obstruction_witnesses.jsonl \
		artifacts/admissibility_witnesses.jsonl artifacts/admissibility_obstructions.jsonl
