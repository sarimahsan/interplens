"""Unit tests for Phase 6 Induction Head Auto-Detector Engine."""

from interplens.adapters.custom import CustomModelAdapter
from interplens.analysis.induction_heads import detect_induction_heads
from interplens.schema import InductionDetectorResponse
from tests.test_phase1 import DummyTransformer, DummyTokenizer


def test_induction_head_detector():
    """Tests repeated sequence (S_1 S_2) induction head detection algorithm."""
    model = DummyTransformer(vocab_size=100, hidden_dim=32, num_layers=4)
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model=model, tokenizer=tokenizer, model_name="dummy_induction")

    res = detect_induction_heads(
        adapter=adapter,
        sequence_length=15,
        threshold=0.10,
        top_k=5
    )

    assert isinstance(res, InductionDetectorResponse)
    assert res.num_layers == adapter.num_layers
    assert res.num_heads == 12
    assert res.total_heads_scanned == adapter.num_layers * 12
    assert len(res.matrix_scores) == adapter.num_layers
    assert len(res.matrix_scores[0]) == 12
    assert len(res.top_induction_heads) <= 5
