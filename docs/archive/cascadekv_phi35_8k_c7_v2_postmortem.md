# CascadeKV Phi-3.5 8K C7-v2 stdout-recovered result postmortem

C7's frozen robust-objective schedule passed development at T5 but did not pass the fresh prospective selected-target quality gates.

C7 changed both the development population (from 15 to 18 already-consumed eligible sources) and the optimizer objective (from global mean relative-L2 to the equal-family cellwise max-of-family-means robust objective). Therefore C6 to C7 does not causally isolate the objective change. The C7 prospective frontier also differs from C6.

The recovered evidence demonstrates a development GO at T5, a frozen T5 schedule, no optimizer rerun during prospective, and a prospective no-pass because T5 failed the absolute cosine and relative-L2 gates. It does not establish that the robust objective is proven worse, that the action set mathematically cannot pass, that larger capacity will pass, or that C7 disproves robust optimization generally.

The exact result-member bytes were recovered from retained Kaggle stdout and verified. The original `cascadekv-c7-result.tar.gz` container bytes were lost after the Kaggle session restart; its SHA-256 and byte size are unknown. The recovery ZIP is archival evidence, not the original Kaggle result container.
