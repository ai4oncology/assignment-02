import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class TestNotebookAnswers:
    def test_submission_complete(self):
        path = ROOT / "submission.json"
        assert path.exists(), (
            "submission.json not found -- open notebook.py in marimo, "
            "fill in the widgets, and the export cell will write the file."
        )
        data = json.loads(path.read_text())
        unanswered = sorted(k for k, v in data.items() if v is None)
        assert not unanswered, (
            f"{len(unanswered)} question(s) unanswered in submission.json: "
            f"{unanswered}. Open the notebook and click through every widget."
        )


def test_build_demo_corpus_shape_and_labels():
    pytest.importorskip("lime")
    from lime_exercise import LIME_CLASS_NAMES, build_demo_corpus

    df = build_demo_corpus()
    assert list(df.columns) == ["text", "label"]
    assert len(df) >= 12
    assert set(df["label"]) == {0, 1}
    assert LIME_CLASS_NAMES == ["benign", "malignant"]


def test_split_corpus_returns_stratified_partitions():
    pytest.importorskip("lime")
    from lime_exercise import build_demo_corpus, split_corpus

    df = build_demo_corpus()
    X_train, X_test, y_train, y_test = split_corpus(df, test_size=0.25, seed=42)
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_build_text_pipeline_returns_pipeline():
    pytest.importorskip("lime")
    from lime_exercise import build_text_pipeline

    model = build_text_pipeline()
    assert list(model.named_steps) == ["tfidf", "clf"]
    assert model.named_steps["tfidf"].lowercase is True
    assert model.named_steps["tfidf"].stop_words == "english"
    assert model.named_steps["clf"].max_iter == 1000
    assert model.named_steps["clf"].random_state == 42


def test_fit_text_pipeline_returns_fitted_model():
    pytest.importorskip("lime")
    from lime_exercise import build_demo_corpus, build_text_pipeline, fit_text_pipeline, split_corpus

    X_train, _, y_train, _ = split_corpus(build_demo_corpus(), test_size=0.25, seed=42)
    model = build_text_pipeline(seed=42)

    fitted = fit_text_pipeline(model, X_train, y_train)

    assert fitted is model
    assert hasattr(fitted.named_steps["tfidf"], "vocabulary_")
    assert hasattr(fitted.named_steps["clf"], "classes_")


def test_evaluate_accuracy_rounds_to_4_decimals():
    pytest.importorskip("lime")
    from lime_exercise import evaluate_accuracy

    class DummyModel:
        def predict(self, X):
            return [0, 1, 1]

    accuracy = evaluate_accuracy(DummyModel(), ["a", "b", "c"], [0, 1, 0])
    assert accuracy == 0.6667


def test_explain_text_prediction_returns_list_for_predicted_class():
    pytest.importorskip("lime")
    from lime_exercise import explain_text_prediction

    class DummyExplanation:
        def as_list(self, label):
            assert label == 1
            return [("malignant", 0.8), ("cells", 0.3)]

    class DummyExplainer:
        def __init__(self, class_names):
            assert class_names == ["benign", "malignant"]

        def explain_instance(self, text_instance, classifier_fn, num_features):
            assert text_instance == "suspicious malignant cells"
            assert num_features == 4
            probs = classifier_fn([text_instance])
            assert probs == [[0.1, 0.9]]
            return DummyExplanation()

    class DummyModel:
        def predict_proba(self, texts):
            assert texts == ["suspicious malignant cells"]
            return [[0.1, 0.9]]

        def predict(self, texts):
            assert texts == ["suspicious malignant cells"]
            return [1]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("lime_exercise.LimeTextExplainer", DummyExplainer)
    try:
        explanation = explain_text_prediction(
            DummyModel(),
            "suspicious malignant cells",
            class_names=["benign", "malignant"],
            num_features=4,
        )
    finally:
        monkeypatch.undo()

    assert explanation == [("malignant", 0.8), ("cells", 0.3)]


def test_top_supporting_words_keeps_positive_weights_in_descending_order():
    pytest.importorskip("lime")
    from lime_exercise import top_supporting_words

    explanation = [("necrosis", -0.2), ("malignant", 0.8), ("cells", 0.3), ("benign", 0.0)]
    assert top_supporting_words(explanation) == ["malignant", "cells"]


def test_text_pipeline_accuracy_exceeds_0_9():
    pytest.importorskip("lime")
    from lime_exercise import (
        build_demo_corpus,
        build_text_pipeline,
        evaluate_accuracy,
        fit_text_pipeline,
        split_corpus,
    )

    df = build_demo_corpus()
    X_train, X_test, y_train, y_test = split_corpus(df, test_size=0.25, seed=42)
    model = build_text_pipeline(seed=42)
    model = fit_text_pipeline(model, X_train, y_train)

    accuracy = evaluate_accuracy(model, X_test, y_test)
    assert accuracy > 0.9


def test_biopsy_carcinoma_receives_highest_lime_score():
    pytest.importorskip("lime")
    from lime_exercise import (
        build_demo_corpus,
        build_text_pipeline,
        explain_text_prediction,
        fit_text_pipeline,
        split_corpus,
    )

    df = build_demo_corpus()
    X_train, _, y_train, _ = split_corpus(df, test_size=0.25, seed=42)
    model = build_text_pipeline(seed=42)
    model = fit_text_pipeline(model, X_train, y_train)

    explanation = explain_text_prediction(
        model,
        "Biopsy confirms anaplastic carcinoma.",
        num_features=6,
    )
    top_feature, _ = max(explanation, key=lambda item: item[1])
    assert top_feature == "carcinoma"


def test_load_breast_cancer_dataframe_shape():
    pytest.importorskip("shap")
    from shap_exercise import load_breast_cancer_dataframe

    X, y = load_breast_cancer_dataframe()
    assert X.shape[0] == len(y)
    assert X.shape[1] == 30
    assert y.name == "target"


def test_mean_absolute_shap_importance_2d_values():
    pytest.importorskip("shap")
    from shap_exercise import mean_absolute_shap_importance

    class DummyValues:
        def __init__(self):
            import numpy as np

            self.values = np.array([[1.0, -2.0], [3.0, 4.0]])

    importance = mean_absolute_shap_importance(DummyValues(), ["a", "b"])
    assert list(importance.index) == ["b", "a"]
    assert importance["a"] == 2.0
    assert importance["b"] == 3.0


def test_split_dataset_returns_stratified_partitions():
    pytest.importorskip("shap")
    from shap_exercise import load_breast_cancer_dataframe, split_dataset

    X, y = load_breast_cancer_dataframe()
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, seed=42)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_train_random_forest_returns_fitted_estimator():
    pytest.importorskip("shap")
    from shap_exercise import load_breast_cancer_dataframe, split_dataset, train_random_forest

    X, y = load_breast_cancer_dataframe()
    X_train, _, y_train, _ = split_dataset(X, y, test_size=0.2, seed=42)
    model = train_random_forest(X_train, y_train, seed=42)

    assert model.n_estimators == 200
    assert model.random_state == 42
    assert hasattr(model, "classes_")


def test_evaluate_model_rounds_to_4_decimals():
    pytest.importorskip("shap")
    from shap_exercise import evaluate_model

    class DummyModel:
        def predict(self, X):
            return [1, 0, 1]

    accuracy = evaluate_model(DummyModel(), [[1], [2], [3]], [1, 1, 1])
    assert accuracy == 0.6667


def test_make_tree_explainer_returns_tree_explainer():
    pytest.importorskip("shap")
    from shap_exercise import load_breast_cancer_dataframe, split_dataset, train_random_forest, make_tree_explainer

    X, y = load_breast_cancer_dataframe()
    X_train, _, y_train, _ = split_dataset(X, y, test_size=0.2, seed=42)
    model = train_random_forest(X_train, y_train, seed=42)
    explainer = make_tree_explainer(model)

    import shap

    assert isinstance(explainer, shap.TreeExplainer)


def test_compute_shap_values_returns_values_object():
    pytest.importorskip("shap")
    from shap_exercise import (
        compute_shap_values,
        load_breast_cancer_dataframe,
        make_tree_explainer,
        split_dataset,
        train_random_forest,
    )

    X, y = load_breast_cancer_dataframe()
    X_train, X_test, y_train, _ = split_dataset(X, y, test_size=0.2, seed=42)
    model = train_random_forest(X_train, y_train, seed=42)
    explainer = make_tree_explainer(model)
    shap_values = compute_shap_values(explainer, X_test.head(5))

    assert hasattr(shap_values, "values")
    assert shap_values.values.shape[0] == 5


def test_random_forest_accuracy_exceeds_0_9():
    pytest.importorskip("shap")
    from shap_exercise import evaluate_model, load_breast_cancer_dataframe, split_dataset, train_random_forest

    X, y = load_breast_cancer_dataframe()
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, seed=42)
    model = train_random_forest(X_train, y_train, seed=42)

    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy > 0.9


def test_mean_absolute_shap_importance_3d_values_uses_positive_class():
    pytest.importorskip("shap")
    from shap_exercise import mean_absolute_shap_importance

    class DummyValues:
        def __init__(self):
            import numpy as np

            self.values = np.array(
                [
                    [[1.0, 10.0], [2.0, 20.0]],
                    [[3.0, 30.0], [4.0, 40.0]],
                ]
            )

    importance = mean_absolute_shap_importance(DummyValues(), ["a", "b"])
    assert list(importance.index) == ["b", "a"]
    assert importance["a"] == 20.0
    assert importance["b"] == 30.0


def test_explain_one_instance_3d_values_uses_positive_class():
    pytest.importorskip("shap")
    from shap_exercise import explain_one_instance

    class DummyValues:
        def __init__(self):
            import numpy as np

            self.values = np.array(
                [
                    [[1.0, 10.0], [2.0, 20.0]],
                    [[3.0, 30.0], [4.0, 40.0]],
                ]
            )

    explanation = explain_one_instance(DummyValues(), row_index=0)
    assert explanation.tolist() == [10.0, 20.0]


def test_explain_one_instance_2d_values_returns_requested_row():
    pytest.importorskip("shap")
    from shap_exercise import explain_one_instance

    class DummyValues:
        def __init__(self):
            import numpy as np

            self.values = np.array([[1.0, -2.0], [3.0, 4.0]])

    explanation = explain_one_instance(DummyValues(), row_index=1)
    assert explanation.tolist() == [3.0, 4.0]


@pytest.fixture(scope="module")
def submission():
    path = ROOT / "submission.json"
    if not path.exists():
        pytest.skip(
            "submission.json not found -- open notebook.py in marimo, fill "
            "in the widgets, and the export cell will write the file."
        )
    return json.loads(path.read_text())


def _require(submission, key):
    assert key in submission, f"{key} missing from submission.json"
    val = submission[key]
    assert val is not None, (
        f"{key} is unanswered in submission.json (None). "
        "Fill in the corresponding widget in the notebook."
    )
    return val


class TestPenAndPaperAnswers:
    def test_q_pp_lin_pred(self, submission):
        assert abs(_require(submission, "Q_PP_LIN_PRED") - 8.45) <= 0.05

    def test_q_pp_lin_eff_age(self, submission):
        assert abs(_require(submission, "Q_PP_LIN_EFF_AGE") - 3.25) <= 0.05

    def test_q_pp_lin_eff_comorb(self, submission):
        assert abs(_require(submission, "Q_PP_LIN_EFF_COMORB") - 3.6) <= 0.05

    def test_q_pp_lin_eff_fit(self, submission):
        assert abs(_require(submission, "Q_PP_LIN_EFF_FIT") - (-1.2)) <= 0.05

    def test_q_pp_lin_eff_surg(self, submission):
        assert abs(_require(submission, "Q_PP_LIN_EFF_SURG") - 0.7) <= 0.05

    def test_q_pp_lin_major(self, submission):
        assert _require(submission, "Q_PP_LIN_MAJOR") == "b"

    def test_q_pp_lin_fit_delta(self, submission):
        assert abs(_require(submission, "Q_PP_LIN_FIT_DELTA") - (-0.9)) <= 0.05

    def test_q_pp_log_or_bmi(self, submission):
        assert abs(_require(submission, "Q_PP_LOG_OR_BMI") - 2.226) <= 0.02

    def test_q_pp_log_prob(self, submission):
        assert abs(_require(submission, "Q_PP_LOG_PROB") - 0.574) <= 0.02

    def test_q_pp_log_bmi_factor(self, submission):
        assert abs(_require(submission, "Q_PP_LOG_BMI_FACTOR") - 0.449) <= 0.02

    def test_q_pp_lime_a0(self, submission):
        assert abs(_require(submission, "Q_PP_LIME_A0") - 1.00) <= 0.01

    def test_q_pp_lime_a1(self, submission):
        assert abs(_require(submission, "Q_PP_LIME_A1") - 0.61) <= 0.01

    def test_q_pp_lime_a2(self, submission):
        assert abs(_require(submission, "Q_PP_LIME_A2") - 0.61) <= 0.01

    def test_q_pp_lime_a3(self, submission):
        assert abs(_require(submission, "Q_PP_LIME_A3") - 0.37) <= 0.01

    def test_q_pp_lime_b0(self, submission):
        assert abs(_require(submission, "Q_PP_LIME_B0") - 0.85) <= 0.01

    def test_q_pp_lime_b1(self, submission):
        assert abs(_require(submission, "Q_PP_LIME_B1") - 0.60) <= 0.01

    def test_q_pp_lime_b2(self, submission):
        assert abs(_require(submission, "Q_PP_LIME_B2") - 0.55) <= 0.01

    def test_q_pp_lime_b3(self, submission):
        assert abs(_require(submission, "Q_PP_LIME_B3") - 0.20) <= 0.01

    def test_q_pp_lime_hr_present(self, submission):
        assert abs(_require(submission, "Q_PP_LIME_HR_PRESENT") - 0.25) <= 0.02

    def test_q_pp_lime_hr_absent(self, submission):
        assert abs(_require(submission, "Q_PP_LIME_HR_ABSENT") - 0.35) <= 0.02

    def test_q_pp_lime_limitation(self, submission):
        assert _require(submission, "Q_PP_LIME_LIMITATION") == "a"

    def test_q_pp_ins_auc(self, submission):
        assert abs(_require(submission, "Q_PP_INS_AUC") - 0.537) <= 0.01

    def test_q_pp_del_auc(self, submission):
        assert abs(_require(submission, "Q_PP_DEL_AUC") - 0.31) <= 0.01

    def test_q_pp_insdel_order(self, submission):
        assert _require(submission, "Q_PP_INSDEL_ORDER") == "a"

    def test_q_pp_shap_lactate(self, submission):
        assert abs(_require(submission, "Q_PP_SHAP_LACTATE") - 0.245) <= 0.01

    def test_q_pp_shap_age(self, submission):
        assert abs(_require(submission, "Q_PP_SHAP_AGE") - 0.105) <= 0.01

    def test_q_pp_shap_map(self, submission):
        assert abs(_require(submission, "Q_PP_SHAP_MAP") - 0.070) <= 0.01

    def test_q_pp_shap_efficiency(self, submission):
        assert abs(_require(submission, "Q_PP_SHAP_EFFICIENCY") - 0.42) <= 0.01

    def test_q_pp_closed_bmi(self, submission):
        assert abs(_require(submission, "Q_PP_CLOSED_BMI") - 1.05) <= 0.02

    def test_q_pp_closed_efficiency_holds(self, submission):
        assert _require(submission, "Q_PP_CLOSED_EFFICIENCY_HOLDS") == "a"

    def test_q_pp_closed_modifiable(self, submission):
        assert _require(submission, "Q_PP_CLOSED_MODIFIABLE") == "a"

    def test_q_pp_kernel_w1(self, submission):
        assert abs(_require(submission, "Q_PP_KERNEL_W1") - 0.25) <= 0.01

    def test_q_pp_kernel_w2(self, submission):
        assert abs(_require(submission, "Q_PP_KERNEL_W2") - 0.125) <= 0.01

    def test_q_pp_kernel_w3(self, submission):
        assert abs(_require(submission, "Q_PP_KERNEL_W3") - 0.25) <= 0.01

    def test_q_pp_kernel_highest(self, submission):
        assert _require(submission, "Q_PP_KERNEL_HIGHEST") == "b"

    def test_q_pp_kernel_exclude(self, submission):
        assert _require(submission, "Q_PP_KERNEL_EXCLUDE") == "a"

    def test_q_pp_kernel_intuition(self, submission):
        assert _require(submission, "Q_PP_KERNEL_INTUITION") == "a"

    def test_q_pp_gradcam_a1(self, submission):
        assert abs(_require(submission, "Q_PP_GRADCAM_A1") - 0.50) <= 0.02

    def test_q_pp_gradcam_a2(self, submission):
        assert abs(_require(submission, "Q_PP_GRADCAM_A2") - (-0.10)) <= 0.02

    def test_q_pp_gradcam_a3(self, submission):
        assert abs(_require(submission, "Q_PP_GRADCAM_A3") - 0.20) <= 0.02

    def test_q_pp_gradcam_l11(self, submission):
        assert abs(_require(submission, "Q_PP_GRADCAM_L11") - 1.5) <= 0.02

    def test_q_pp_gradcam_l12(self, submission):
        assert abs(_require(submission, "Q_PP_GRADCAM_L12") - 0.0) <= 0.02

    def test_q_pp_gradcam_l21(self, submission):
        assert abs(_require(submission, "Q_PP_GRADCAM_L21") - 0.9) <= 0.02

    def test_q_pp_gradcam_l22(self, submission):
        assert abs(_require(submission, "Q_PP_GRADCAM_L22") - 1.3) <= 0.02

    def test_q_pp_gradcam_toploc(self, submission):
        assert _require(submission, "Q_PP_GRADCAM_TOPLOC") == "a"

    def test_q_pp_method_tree(self, submission):
        assert _require(submission, "Q_PP_METHOD_TREE") == "a"

    def test_q_pp_method_image(self, submission):
        assert _require(submission, "Q_PP_METHOD_IMAGE") == "b"

    def test_q_pp_method_genes(self, submission):
        assert _require(submission, "Q_PP_METHOD_GENES") == "b"

    def test_q_pp_method_fairness(self, submission):
        assert _require(submission, "Q_PP_METHOD_FAIRNESS") == "b"
