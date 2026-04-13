from multimodal_rag.ingestion.extractors import _dedupe_table_candidates, _word_overlaps_table


def test_word_overlap_detects_table_region():
    word_bbox = (12.0, 15.0, 25.0, 25.0)
    table_bboxes = [(10.0, 10.0, 100.0, 80.0)]
    assert _word_overlaps_table(word_bbox, table_bboxes, threshold=0.3)


def test_word_overlap_ignores_non_overlapping_region():
    word_bbox = (120.0, 200.0, 180.0, 240.0)
    table_bboxes = [(10.0, 10.0, 100.0, 80.0)]
    assert not _word_overlaps_table(word_bbox, table_bboxes, threshold=0.3)


def test_dedupe_table_candidates_by_bbox_iou():
    rowset = [["col1", "col2"], ["a", "b"]]
    candidates = [
        ((10.0, 10.0, 100.0, 80.0), rowset),
        ((11.0, 11.0, 99.0, 79.0), rowset),  # near-duplicate
        ((150.0, 30.0, 250.0, 90.0), rowset),
    ]
    deduped = _dedupe_table_candidates(candidates)
    assert len(deduped) == 2
