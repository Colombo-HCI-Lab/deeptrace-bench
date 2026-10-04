"""Manifest builders: one module per dataset, each turning its layout into a manifest.

Every builder has the signature ``build_manifest(root: Path) -> pandas.DataFrame``, where
``root`` is ``DTB_ROOT/datasets/<id>``, and returns rows in the schema of
``deeptrace_bench.manifest``. Builders are written against the real files once a dataset is
downloaded; until then each module documents the layout as far as it is known.
"""
