# Task setup notes
Generated: 2026-07-27 21:38:27

Build contract: tasks-v2

## math
- HF: HuggingFaceH4/MATH-500
- N: 500
- strata: [{'difficulty_raw': 1, 'len': 43}, {'difficulty_raw': 2, 'len': 90}, {'difficulty_raw': 3, 'len': 105}, {'difficulty_raw': 4, 'len': 128}, {'difficulty_raw': 5, 'len': 134}]
- load: 3.5s

## code
- HF: livecodebench/code_generation_lite
- N: 175
- strata: [{'difficulty_raw': 'easy', 'len': 43}, {'difficulty_raw': 'hard', 'len': 80}, {'difficulty_raw': 'medium', 'len': 52}]
- load: 2.6s

## moral
- HF: morebench/morebench
- N: 500
- strata: [{'difficulty_raw': None, 'len': 500}]
- load: 1.1s

## planning
- HF: ibm/acp_bench
- N: 500
- strata: [{'difficulty_raw': 'applicable_actions_mc', 'len': 100}, {'difficulty_raw': 'landmarks_mcq', 'len': 100}, {'difficulty_raw': 'progression_mcq', 'len': 100}, {'difficulty_raw': 'reachable_atom_mc', 'len': 100}, {'difficulty_raw': 'validation_mcq', 'len': 100}]
- load: 4.0s

## gpqa
- HF: Idavidrein/gpqa
- N: 500
- strata: [{'difficulty_raw': 'Biology', 'len': 93}, {'difficulty_raw': 'Chemistry', 'len': 198}, {'difficulty_raw': 'Physics', 'len': 209}]
- load: 1.7s

## security
- HF: cais/wmdp
- revision: 7125571f22f032c56415e7980f48d877dd830ff8
- N: 500
- strata: [{'difficulty_raw': None, 'len': 500}]
- load: 2.1s

## safety
- HF: Machlovi/strongreject-dataset
- revision: c18bb810edc4b60b815b205faeaa2bd5f72c3e93
- N: 313
- strata: [{'difficulty_raw': 'Disinformation and deception', 'len': 50}, {'difficulty_raw': 'Hate, harassment and discrimination', 'len': 50}, {'difficulty_raw': 'Illegal goods and services', 'len': 50}, {'difficulty_raw': 'Non-violent crimes', 'len': 59}, {'difficulty_raw': 'Sexual content', 'len': 50}, {'difficulty_raw': 'Violence', 'len': 54}]
- load: 0.9s

## idea
- HF: 6cf/LiveIdeaBench
- N: 500
- strata: [{'difficulty_raw': None, 'len': 500}]
- load: 2.7s
