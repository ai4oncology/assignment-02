import pytest

def test_build_demo_corpus_shape_and_labels():
    pytest.importorskip("lime")
    from lime_exercise import LIME_CLASS_NAMES, build_demo_corpus

    df = build_demo_corpus()
    assert list(df.columns) == ["text", "label"]
    assert len(df) >= 12
    assert set(df["label"]) == {0, 1}
    assert LIME_CLASS_NAMES == ["ham", "spam"]


def test_split_corpus_is_not_implemented_yet():
    pytest.importorskip("lime")
    from lime_exercise import split_corpus

    with pytest.raises(NotImplementedError):
        split_corpus(build_demo_corpus())


def test_build_text_pipeline_is_not_implemented_yet():
    pytest.importorskip("lime")
    from lime_exercise import build_text_pipeline

    with pytest.raises(NotImplementedError):
        build_text_pipeline()


def test_load_breast_cancer_dataframe_is_not_implemented_yet():
    pytest.importorskip("shap_exercise")
    from shap_exercise import load_breast_cancer_dataframe

    with pytest.raises(NotImplementedError):
        load_breast_cancer_dataframe()


def test_mean_absolute_shap_importance_is_not_implemented_yet():
    pytest.importorskip("shap")
    from shap_exercise import mean_absolute_shap_importance

    class DummyValues:
        def __init__(self):
            import numpy as np

            self.values = np.array([[1.0, -2.0], [3.0, 4.0]])

    with pytest.raises(NotImplementedError):
        mean_absolute_shap_importance(DummyValues(), ["a", "b"])


def test_explain_one_instance_is_not_implemented_yet():
    pytest.importorskip("shap")
    from shap_exercise import explain_one_instance

    class DummyValues:
        def __init__(self):
            import numpy as np

            self.values = np.array([[1.0, -2.0], [3.0, 4.0]])

    with pytest.raises(NotImplementedError):
        explain_one_instance(DummyValues(), row_index=0)
