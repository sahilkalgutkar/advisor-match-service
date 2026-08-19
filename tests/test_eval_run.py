import eval.run as eval_module
from eval.run import load_queries


class FakeEmbedder:
    def embed_text(self, text):
        return text  # identity - the fake search below doesn't care about the value


def test_load_queries(tmp_path, monkeypatch):
    queries_file = tmp_path / "queries.yaml"
    queries_file.write_text('queries:\n  - query: "heart"\n    expected_name: "Jane Doe"\n')
    monkeypatch.setattr(eval_module, "QUERIES_PATH", queries_file)

    assert load_queries() == [{"query": "heart", "expected_name": "Jane Doe"}]


def test_run_reports_hit_rate_and_mrr_across_hits_and_misses(tmp_path, monkeypatch, capsys):
    queries_file = tmp_path / "queries.yaml"
    queries_file.write_text(
        "queries:\n"
        '  - query: "first place hit"\n    expected_name: "Jane Doe"\n'
        '  - query: "second place hit"\n    expected_name: "John Smith"\n'
        '  - query: "total miss"\n    expected_name: "Nobody"\n'
    )
    monkeypatch.setattr(eval_module, "QUERIES_PATH", queries_file)
    monkeypatch.setattr(eval_module, "get_embedder", lambda: FakeEmbedder())

    def fake_hybrid_search(client, query_text, query_embedding, k=5, index_name=None):
        return {
            "first place hit": [{"name": "Jane Doe"}, {"name": "Other"}],
            "second place hit": [{"name": "Other"}, {"name": "John Smith"}],
            "total miss": [{"name": "Someone Else"}],
        }[query_text]

    monkeypatch.setattr(eval_module, "hybrid_search", fake_hybrid_search)

    report = eval_module.run(client=object())

    out = capsys.readouterr().out
    assert "[HIT ] 'first place hit'" in out
    assert "[HIT ] 'second place hit'" in out
    assert "[MISS] 'total miss'" in out
    assert "hit@5: 2/3 (67%)" in out

    assert report["hit_rate"] == 2 / 3
    # reciprocal ranks: 1/1, 1/2, 0
    assert report["mrr"] == (1 + 0.5 + 0) / 3
    assert report["n"] == 3
