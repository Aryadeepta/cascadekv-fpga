CascadeKV Phi-3.5 8K C7 V2 — recovery evidence

The Kaggle session restarted after the frozen C7-v2 runner had already:
  * completed prospective testing,
  * completed FINAL AUDIT,
  * created the result bundle,
  * emitted the stdout recovery manifest and all artifact frames,
  * completed holdout cleanup, and
  * printed that the frozen runner completed successfully.

The session restarted at the beginning of the notebook's subsequent
"Auditing original result tarball" stage. Therefore:

EXACTLY RECOVERED
-----------------
* The retained notebook-output text included in this archive.
* All 10 result member files under recovered_members/.
* Each result member byte count and SHA-256, verified against the runner's
  printed recovery manifest.
* All nine non-SHA256SUMS member hashes, verified against the recovered
  internal SHA256SUMS member.

NOT RECOVERED
-------------
* The original `cascadekv-c7-result.tar.gz` container bytes.
* The original tar/gzip container SHA-256 and size, because those would have
  been printed only after the point where the session restarted.

Do not label any newly created tar/zip as the original Kaggle result tar.
The member files are exact stdout recoveries; any enclosing container made
afterward is a reconstructed convenience container.
