# Generated manifests

This directory is populated by `tools/prepare_cvact_manifests.py` and
`tools/prepare_cvusa_manifests.py`.

Generated files contain repository-relative dataset paths and are intentionally
ignored. They remain valid when the complete repository is moved, provided the
datasets retain the documented layout under `data/`. Rebuild them from the
official datasets by following the root README. Random evaluation crops are
generated at evaluation time and recorded beside the corresponding results
rather than stored here.
