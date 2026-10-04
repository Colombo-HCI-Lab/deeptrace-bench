"""Pipeline tests and smoke runs can't be published as results."""

from __future__ import annotations

from deeptrace_bench.eval.publish import publish_refusal


def test_pipeline_tests_are_never_published(registry):
    assert "pipeline test" in publish_refusal(registry.evalset("unidatapro_videos"), None)


def test_nothing_in_a_namespace_is_published(registry):
    assert "smoke" in publish_refusal(registry.evalset("urdu_csalt"), "smoke")


def test_a_real_run_may_be_published(registry):
    assert publish_refusal(registry.evalset("urdu_csalt"), None) is None
