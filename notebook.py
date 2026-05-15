import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Assignment-02: Model Explanations with LIME and SHAP
        **ML4Health 2026**

        ---

        This notebook includes the first two parts of Assignment 02:

        1. **Section A - LIME for text classification** (`lime_exercise.py`)
        2. **Section B - SHAP on the Breast Cancer Wisconsin dataset** (`shap_exercise.py`)
        3. **Section C - Pen and paper exercises** (to be added later)

        Run cells top-to-bottom. Coding tasks live in the two Python modules.

        ### How to work
        - Implement the `TODO` functions in **`lime_exercise.py`** and **`shap_exercise.py`**.
        - Use this notebook to load data, train models, and inspect explanations.
        - Run `pytest tests/ -v` locally before pushing.
        - Record difficulties or open questions in `experiences.md`.
        """
    )
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import shap
    from lime import lime_text

    from lime_exercise import (
        LIME_CLASS_NAMES,
        build_demo_corpus,
        build_text_pipeline,
        evaluate_accuracy as evaluate_text_accuracy,
        explain_text_prediction,
        fit_text_pipeline,
        split_corpus,
        top_supporting_words,
    )
    from shap_exercise import (
        compute_shap_values,
        evaluate_model,
        explain_one_instance,
        load_breast_cancer_dataframe,
        make_tree_explainer,
        mean_absolute_shap_importance,
        split_dataset,
        train_random_forest,
    )

    ROOT = Path(".")
    return (
        LIME_CLASS_NAMES,
        ROOT,
        build_demo_corpus,
        build_text_pipeline,
        compute_shap_values,
        evaluate_model,
        evaluate_text_accuracy,
        explain_one_instance,
        explain_text_prediction,
        fit_text_pipeline,
        lime_text,
        load_breast_cancer_dataframe,
        make_tree_explainer,
        mean_absolute_shap_importance,
        np,
        pd,
        plt,
        shap,
        split_corpus,
        split_dataset,
        top_supporting_words,
        train_random_forest,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Section A - LIME for Text Classification

        ## Learning goals
        By the end of this exercise, you should be able to:
        1. train a simple text classifier,
        2. explain an individual text prediction with **LIME**,
        3. identify which words push the prediction toward a class,
        4. critically assess the usefulness and limits of local explanations in NLP.
        """
    )
    return


@app.cell
def _(build_demo_corpus):
    lime_df = build_demo_corpus()
    lime_df.head()
    return (lime_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The LIME section uses the local movie-review dataset
        `movie_sentiment_200.csv`, with columns `text` and `label`.
        The goal here is not benchmark performance; it is to inspect how a
        text model justifies one decision.
        """
    )
    return


@app.cell
def _(build_text_pipeline, fit_text_pipeline, lime_df, split_corpus):
    lime_X_train, lime_X_test, lime_y_train, lime_y_test = split_corpus(lime_df)
    lime_model = build_text_pipeline()
    lime_model = fit_text_pipeline(lime_model, lime_X_train, lime_y_train)
    return lime_X_test, lime_model, lime_y_test


@app.cell
def _(evaluate_text_accuracy, lime_X_test, lime_model, lime_y_test):
    lime_test_accuracy = evaluate_text_accuracy(lime_model, lime_X_test, lime_y_test)
    print("LIME section test accuracy:", lime_test_accuracy)
    return (lime_test_accuracy,)


@app.cell
def _(lime_X_test, lime_model):
    sample_text = lime_X_test.iloc[0]
    sample_prediction = lime_model.predict([sample_text])[0]
    sample_probabilities = lime_model.predict_proba([sample_text])[0]
    print("Sample text:", sample_text)
    print("Predicted class:", sample_prediction)
    print("Probabilities:", sample_probabilities)
    return sample_prediction, sample_text


@app.cell
def _(LIME_CLASS_NAMES, explain_text_prediction, sample_text, lime_model):
    lime_explanation = explain_text_prediction(
        lime_model,
        sample_text,
        class_names=LIME_CLASS_NAMES,
        num_features=6,
    )
    lime_explanation
    return (lime_explanation,)


@app.cell
def _(lime_explanation, top_supporting_words):
    print("Top supporting words:", top_supporting_words(lime_explanation))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Questions to discuss while running this section:

        - Which words have the strongest positive contribution?
        - Do those words make semantic sense?
        - What does LIME miss when it explains text one instance at a time?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Section B - SHAP on the Breast Cancer Wisconsin Dataset

        ## Learning goals
        By the end of this notebook, you should be able to:
        1. train a simple classification model,
        2. compute SHAP explanations,
        3. interpret **global** explanations,
        4. interpret **local** explanations for one prediction,
        5. create your own SHAP visualizations.
        """
    )
    return


@app.cell
def _(load_breast_cancer_dataframe):
    bc_X, bc_y = load_breast_cancer_dataframe()
    bc_X.head()
    return bc_X, bc_y


@app.cell
def _(bc_X, bc_y, split_dataset, train_random_forest):
    bc_X_train, bc_X_test, bc_y_train, bc_y_test = split_dataset(bc_X, bc_y)
    bc_model = train_random_forest(bc_X_train, bc_y_train)
    return bc_X_test, bc_model, bc_y_test


@app.cell
def _(bc_X_test, bc_model, bc_y_test, evaluate_model):
    bc_test_accuracy = evaluate_model(bc_model, bc_X_test, bc_y_test)
    print("SHAP section test accuracy:", bc_test_accuracy)
    return (bc_test_accuracy,)


@app.cell
def _(bc_model, make_tree_explainer):
    bc_explainer = make_tree_explainer(bc_model)
    return (bc_explainer,)


@app.cell
def _(bc_X_test, bc_explainer, compute_shap_values):
    bc_shap_values = compute_shap_values(bc_explainer, bc_X_test)
    print("SHAP values shape:", bc_shap_values.values.shape)
    return (bc_shap_values,)


@app.cell
def _(bc_X_test, bc_shap_values, mean_absolute_shap_importance):
    global_importance = mean_absolute_shap_importance(
        bc_shap_values,
        bc_X_test.columns.tolist(),
    )
    global_importance.head(10)
    return (global_importance,)


@app.cell
def _(global_importance, plt):
    global_importance.head(10).sort_values().plot(kind="barh", figsize=(8, 5))
    plt.title("Top-10 mean(|SHAP|) features")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(bc_shap_values, explain_one_instance):
    local_shap_values = explain_one_instance(bc_shap_values, row_index=0)
    print("First 10 local SHAP values:", local_shap_values[:10])
    return (local_shap_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Suggested follow-up:

        - Use `shap.summary_plot(...)` for a global view.
        - Use `shap.plots.waterfall(...)` or `shap.force_plot(...)` for one case.
        - Compare the global ranking with the local explanation of a single patient.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Section C - Pen and Paper

        This section will be added later.
        """
    )
    return


if __name__ == "__main__":
    app.run()
