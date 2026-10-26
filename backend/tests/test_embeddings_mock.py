import numpy as np

from app.services.embeddings import set_mock_embeddings, embed_texts


def test_mock_embeddings_similar_for_related_text():
    set_mock_embeddings(True)
    a, b, c = embed_texts(
        [
            "Hypertension is high blood pressure",
            "What is hypertension and blood pressure?",
            "The weather is sunny today in the park",
        ]
    )
    sim_related = float(np.dot(a, b))
    sim_unrelated = float(np.dot(a, c))
    assert sim_related > sim_unrelated
    assert sim_related > 0.2
