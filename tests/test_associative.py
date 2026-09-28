from datetime import date

from graph_memory.associative import AssociativeMemory, Extraction, Params, time_window


def test_time_window():
    now = date(2023, 5, 30)
    assert time_window("What did I do two months ago?", now) == (date(2023, 3, 23), date(2023, 4, 8))
    assert time_window("gym visits in the past month", now) == (date(2023, 4, 30), now)
    assert time_window("What did I buy in March?", now) == (date(2023, 3, 1), date(2023, 3, 31))
    assert time_window("the trip in June", now) == (date(2022, 6, 1), date(2022, 6, 30))
    assert time_window("two days before the wedding", now) is None
    assert time_window("How many weeks ago did I start?", now) is None


def extraction(memories, entities):
    return Extraction.model_validate({
        "memories": [{"text": t, "state": s, "round": 0, "date": "", "entities": e, "confidence": 1} for t, s, e in memories],
        "entities": [{"name": n, "is_a": c} for n, c in entities]})


def test_activation_reaches_memories_the_input_only_implies():
    m = AssociativeMemory()
    unrelated = [1.0, 0.0]  # every memory is orthogonal to the input: only association can recall it
    texts = {"Urine culture: ESBL E. coli": unrelated, "Patient received cefotaxime": unrelated,
             "Patient likes jazz": unrelated}
    for sid, day, mem, ents in [
        ("s1", date(2023, 1, 10), ("Urine culture: ESBL E. coli", "stated", ["ESBL E. coli"]), [("ESBL E. coli", ["E. coli"])]),
        ("s2", date(2023, 3, 1), ("Patient received cefotaxime", "stated", ["cefotaxime"]), [("cefotaxime", ["antibiotic"])]),
        ("s3", date(2023, 3, 2), ("Patient likes jazz", "stated", ["jazz"]), [("jazz", ["music"])]),
    ]:
        m.add_round(f"{sid}#0", mem[0], unrelated, day, sid)
        m.add_session(sid, day, [f"{sid}#0"], extraction([mem], ents), texts)

    ranked = m.activate("UTI caused by E. coli treated with cefotaxime", [0.0, 1.0], date(2023, 5, 30))
    memories = [r for r in ranked if r["kind"] == "memory"]
    assert [r["memory"] for r in memories] == ["Patient received cefotaxime", "Urine culture: ESBL E. coli"]
    assert memories[1]["activation_path"] == ["E. coli", "ESBL E. coli", "Urine culture: ESBL E. coli"]
    assert memories[0]["score"] > memories[1]["score"]  # named directly: one hop closer than the culture

    no_spread = m.activate("UTI caused by E. coli treated with cefotaxime", [0.0, 1.0], date(2023, 5, 30), Params(hops=0))
    assert not [r for r in no_spread if r["kind"] == "memory"]


def test_generic_nodes_spread_less():
    m = AssociativeMemory()
    for i in range(40):
        m.add_session(f"s{i}", date(2023, 1, 1), [], extraction([(f"memory {i}", "stated", ["infection"])], []), {})
    m.add_session("x", date(2023, 1, 1), [], extraction([("ESBL", "stated", ["ESBL E. coli"])], []), {})
    p = Params()
    assert m.specificity("e:infection", p) < 0.5 < m.specificity("e:esbl e coli", p) == 1
