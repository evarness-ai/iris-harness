"""Tests for the deterministic task-spec layer."""

from __future__ import annotations

import pytest

from iris_harness.agent.task_spec import (
    TaskSpec,
    Verdict,
    artifacts_satisfy_request,
    build_missing_artifact_answer,
    build_task_spec,
    describe_expected_artifact,
    expected_artifact_extensions,
    render_code_exec_system_prompt_for_small,
    user_visible_artifacts,
    verify,
)


class TestExpectedArtifactExtensions:
    @pytest.mark.parametrize(
        "query,expected",
        [
            (
                "write a one-page document about Agentic Harness",
                {".md", ".markdown", ".pdf", ".html", ".htm", ".txt"},
            ),
            ("create a one-pager for me", {".md", ".markdown", ".pdf", ".html", ".htm", ".txt"}),
            ("draft an article about LLMs", {".md", ".markdown", ".pdf", ".html", ".htm", ".txt"}),
            ("generate a report", {".md", ".markdown", ".pdf", ".html", ".htm", ".txt"}),
            ("export the data as a PDF", {".pdf"}),
            ("save the table to csv", {".csv"}),
            ("export to excel", {".xlsx", ".xls"}),
            ("write the result as markdown", {".md", ".markdown"}),
            ("render an html page", {".html", ".htm"}),
        ],
    )
    def test_document_keywords_imply_extensions(self, query: str, expected: set[str]) -> None:
        assert expected_artifact_extensions(query) == expected

    @pytest.mark.parametrize(
        "query",
        [
            "what is 2 + 2",
            "summarize this conversation",
            "list files in cwd",
        ],
    )
    def test_no_deliverable_request_returns_empty(self, query: str) -> None:
        assert expected_artifact_extensions(query) == set()


class TestDescribeExpectedArtifact:
    def test_pdf_request(self) -> None:
        assert describe_expected_artifact("export to pdf") == "a PDF artifact"

    def test_generic_document_request(self) -> None:
        result = describe_expected_artifact("write a one-page article")
        assert "document artifact" in result

    def test_no_request(self) -> None:
        assert describe_expected_artifact("what time is it") == "the requested output"


class TestBuildTaskSpec:
    def test_document_request_requires_artifact(self) -> None:
        spec = build_task_spec("write a one-page article on Agentic Harness")
        assert isinstance(spec, TaskSpec)
        assert spec.must_create_artifact is True
        assert ".md" in spec.required_extensions
        assert ".pdf" in spec.required_extensions

    def test_chat_request_does_not_require_artifact(self) -> None:
        spec = build_task_spec("how does autoscaling work")
        assert spec.must_create_artifact is False
        assert spec.required_extensions == frozenset()

    def test_pdf_request_is_narrow(self) -> None:
        spec = build_task_spec("save invoice as pdf")
        assert spec.required_extensions == frozenset({".pdf"})

    def test_query_preserved_verbatim(self) -> None:
        q = "Write a one-page document about Agentic Harness"
        assert build_task_spec(q).query == q


class TestVerify:
    def test_no_artifact_required_is_satisfied(self) -> None:
        spec = build_task_spec("what is 2 + 2")
        assert verify(spec, []) is Verdict.SATISFIED
        assert verify(spec, ["/workspace/script.py"]) is Verdict.SATISFIED

    def test_document_request_with_only_helper_script_is_missing(self) -> None:
        spec = build_task_spec("write a one-page article")
        assert verify(spec, ["/workspace/script.py"]) is Verdict.MISSING_ARTIFACT

    def test_document_request_with_markdown_file_is_satisfied(self) -> None:
        spec = build_task_spec("write a one-page article")
        assert verify(spec, ["/workspace/script.py", "/workspace/article.md"]) is Verdict.SATISFIED

    def test_pdf_request_needs_pdf(self) -> None:
        spec = build_task_spec("export as pdf")
        assert verify(spec, ["/workspace/result.md"]) is Verdict.MISSING_ARTIFACT
        assert verify(spec, ["/workspace/result.pdf"]) is Verdict.SATISFIED


class TestSmallTierPrompt:
    def test_pdf_request_lists_required_extensions(self) -> None:
        spec = build_task_spec("export to pdf")
        out = render_code_exec_system_prompt_for_small(spec)

        assert "REQUIRED DELIVERABLE" in out
        assert ".pdf" in out
        assert "run_shell" in out
        assert '{"tool":"run_shell"' in out

    def test_chat_request_omits_deliverable_block(self) -> None:
        spec = build_task_spec("what is 2 + 2")
        out = render_code_exec_system_prompt_for_small(spec)

        assert "REQUIRED DELIVERABLE" not in out
        assert "TASK" in out
        assert "No specific file deliverable" in out

    def test_includes_heredoc_rule_no_os_package_manager_and_ask_user_policy(self) -> None:
        spec = build_task_spec("write an article")
        out = render_code_exec_system_prompt_for_small(spec)

        assert "PYEOF" in out
        assert "NEVER run apt/yum/brew" in out
        assert "ask_user is disabled" in out


class TestUserVisibleArtifacts:
    def test_filters_helper_scripts_for_document_request(self) -> None:
        visible = user_visible_artifacts(
            "write a one-page article",
            ["/workspace/script.py", "/workspace/article.md"],
        )
        assert visible == ["/workspace/article.md"]

    def test_keeps_everything_when_no_deliverable(self) -> None:
        artifacts = ["/workspace/a.py", "/workspace/b.txt"]
        assert user_visible_artifacts("just run it", artifacts) == artifacts

    def test_dedupes(self) -> None:
        result = user_visible_artifacts(
            "no deliverable",
            ["/workspace/a.py", "/workspace/a.py", "/workspace/b.py"],
        )
        assert result == ["/workspace/a.py", "/workspace/b.py"]


class TestBackCompatHelpers:
    """The legacy free-function names still work; bootstrap depends on them."""

    def test_artifacts_satisfy_request_no_requirement(self) -> None:
        assert artifacts_satisfy_request("hello", []) is True

    def test_artifacts_satisfy_request_missing(self) -> None:
        assert artifacts_satisfy_request("write an article", ["/w/script.py"]) is False

    def test_build_missing_artifact_answer_mentions_query_and_workspace(self) -> None:
        msg = build_missing_artifact_answer(
            task_query="write a one-page article",
            artifacts=["/w/script.py"],
            workspace_path="/w",
            iterations=4,
        )
        assert "write a one-page article" in msg
        assert "/w" in msg
        assert "4 iteration" in msg
        # script.py is not a visible artifact for a document request
        assert "script.py" not in msg
