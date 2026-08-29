import knowledge_base as kb


def test_maintenance_actions_covers_all_tiers():
    assert set(kb.MAINTENANCE_ACTIONS.keys()) == {"emergency", "elevated", "moderate", "low"}
    for actions in kb.MAINTENANCE_ACTIONS.values():
        assert len(actions) > 0


def test_theft_actions_covers_all_tiers():
    assert set(kb.THEFT_ACTIONS.keys()) == {"emergency", "elevated", "moderate", "low"}
    for actions in kb.THEFT_ACTIONS.values():
        assert len(actions) > 0


def test_symptom_and_glossary_keys_do_not_collide():
    # Dispatch in chatbot.py checks GLOSSARY then SYMPTOM_CAUSES by substring
    # match - if a phrase could match both, the outcome would depend on
    # dispatch order rather than being well-defined.
    for symptom_key in kb.SYMPTOM_CAUSES:
        for glossary_key in kb.GLOSSARY:
            assert symptom_key not in glossary_key
            assert glossary_key not in symptom_key


def test_symptom_entries_have_causes_and_steps():
    for entry in kb.SYMPTOM_CAUSES.values():
        assert entry["causes"]
        assert entry["inspection_steps"]


def test_glossary_entries_are_complete():
    for entry in kb.GLOSSARY.values():
        assert entry["meaning"]
        assert "recommended_actions" in entry and entry["recommended_actions"]
