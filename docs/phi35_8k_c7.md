# Phi-3.5 8K C7 executor

C7 is a robustness test motivated by the normal C6 result; it does not
predict success.  Its only scientific changes are expanding the consumed
development population from 15 to 18 sources and replacing global cell mean
relative L2 with the maximum of the narrative, report, and QA family means.
Each family supplies six sources, hence 18 observations per family/cell/action
and 8,640 observations per action.

A0--A11, geometry, targets, and integer-microbyte accounting are unchanged.
A10/A11 remain diagnostic at every layer but selectable only at layer 0.
The DP minimizes robust cell loss; global mean cosine, relative L2, and traffic
remain separately recorded and are the only values used by unchanged gates.

The untouched prospective frontier is narrative19, qa20, and report23.  A GO
freezes all schedules and the selected target before any prospective selection;
the prospective phase cannot optimize or reopen development tensors. Because
both population and objective change, C7 cannot attribute any outcome to one
change alone.
