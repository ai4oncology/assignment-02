import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import json
    from pathlib import Path as _Path

    _submission_path = _Path(__file__).with_name("submission.json")
    try:
        submission_data = json.loads(_submission_path.read_text()) if _submission_path.exists() else {}
    except json.JSONDecodeError:
        submission_data = {}

    def submission_default(key, default=None):
        return submission_data.get(key, default)

    def submission_radio_default(key, options, default=None):
        saved_value = submission_data.get(key, default)
        if saved_value in options:
            return saved_value
        for option_key, option_value in options.items():
            if option_value == saved_value:
                return option_key
        return default

    return submission_default, submission_radio_default


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Assignment-02: Model Explanations
        **ML4Health 2026**

        ---

        This notebook includes the first two parts of Assignment 02:

        1. **Section A - LIME for text classification** (`lime_exercise.py`)
        2. **Section B - SHAP on the Breast Cancer Wisconsin dataset** (`shap_exercise.py`)
        3. **Section C - Pen and paper exercises**

        Run cells top-to-bottom. Coding tasks live in the two Python modules.

        ### How to work
        - Implement the `TODO` functions in **`lime_exercise.py`** and **`shap_exercise.py`**.
        - Multiple-choice and short-numeric questions are answered with the radio /
          number widgets in this notebook. A hidden cell at the bottom auto-saves
          every widget value to **`submission.json`** whenever anything changes.
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

        This section collects the week 4 and week 5 pen-and-paper exercises on
        explainability foundations. Work through the questions directly in the
        notebook and make sure every widget has an answer before submitting.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Part 1 - Week 4 Foundations
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            We include the week 4 exercises on:

            1. interpreting a clinical linear model,
            2. odds ratios in logistic regression,
            3. a step-by-step LIME example.

            The decision-tree split exercise is intentionally omitted.
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise C1 - Interpreting a Clinical Linear Model

        A hospital builds a linear regression model to predict length of stay (days) after surgery:

        $\hat{y} = 2.1 + 0.05 \cdot \text{age} + 1.8 \cdot \text{comorbidity\_index} - 0.3 \cdot \text{fitness\_score} + 0.7 \cdot \text{surgery\_type}$

        where `surgery_type = 1` for major surgery and `0` for minor surgery.

        Patient A has:
        - `age = 65`
        - `comorbidity_index = 2`
        - `fitness_score = 4`
        - `surgery_type = 1`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lin_pred = mo.ui.number(
        start=-100.0,
        stop=100.0,
        step=0.01,
        label="C1(a) Compute the predicted length of stay for Patient A (in days):",
        value=submission_default("Q_PP_LIN_PRED"),
    )
    q_pp_lin_pred
    return (q_pp_lin_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        C1(b) Compute the feature effect $\beta_j \cdot x_j$ for each feature for
        Patient A.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lin_eff_age = mo.ui.number(
        start=-100.0,
        stop=100.0,
        step=0.01,
        label='$\\beta_{age} \\cdot x_{age}$',
        value=submission_default("Q_PP_LIN_EFF_AGE"),
    )
    q_pp_lin_eff_age
    return (q_pp_lin_eff_age,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lin_eff_comorb = mo.ui.number(
        start=-100.0,
        stop=100.0,
        step=0.01,
        label='$\\beta_{comorbidity} \\cdot x_{comorbidity}$',
        value=submission_default("Q_PP_LIN_EFF_COMORB"),
    )
    q_pp_lin_eff_comorb
    return (q_pp_lin_eff_comorb,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lin_eff_fit = mo.ui.number(
        start=-100.0,
        stop=100.0,
        step=0.01,
        label='$\\beta_{fitness} \\cdot x_{fitness}$',
        value=submission_default("Q_PP_LIN_EFF_FIT"),
    )
    q_pp_lin_eff_fit
    return (q_pp_lin_eff_fit,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lin_eff_surg = mo.ui.number(
        start=-100.0,
        stop=100.0,
        step=0.01,
        label='$\\beta_{surgery} \\cdot x_{surgery}$',
        value=submission_default("Q_PP_LIN_EFF_SURG"),
    )
    q_pp_lin_eff_surg
    return (q_pp_lin_eff_surg,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_lin_major_options = {
            "Age": "a",
            "Comorbidity index": "b",
            "Fitness score": "c",
            "Surgery type": "d",
    }
    q_pp_lin_major = mo.ui.radio(
        options=q_pp_lin_major_options,
        label="C1(c) Select the feature with the largest contribution to Patient A's prediction:",
        value=submission_radio_default("Q_PP_LIN_MAJOR", q_pp_lin_major_options),
    )
    q_pp_lin_major
    return (q_pp_lin_major,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lin_fit_delta = mo.ui.number(
        start=-10.0,
        stop=10.0,
        step=0.01,
        label="C1(d) Compute the change in predicted length of stay if fitness_score changes from 4 to 7 (in days):",
        value=submission_default("Q_PP_LIN_FIT_DELTA"),
    )
    q_pp_lin_fit_delta
    return (q_pp_lin_fit_delta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise C2 - Odds Ratios in Disease Prediction

        A logistic regression predicts diabetes risk:

        $$
        \log \left( \frac{P(Y=1)}{1-P(Y=1)} \right)
        = -4.0 + 0.03 \cdot \text{age} + 0.8 \cdot \text{BMI\_category} + 1.2 \cdot \text{family\_history}
        $$

        Variable definitions:
        - $\text{BMI\_category} \in \{0, 1, 2\}$
        - $\text{family\_history} \in \{0, 1\}$

        Patient values:
        - $\text{age} = 50$
        - $\text{BMI\_category} = 2$
        - $\text{family\_history} = 1$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_log_or_bmi = mo.ui.number(
        start=0.0,
        stop=10.0,
        step=0.001,
        label='C2(a) Compute the odds ratio for a one-step increase in "BMI_category":',
        value=submission_default("Q_PP_LOG_OR_BMI"),
    )
    q_pp_log_or_bmi
    return (q_pp_log_or_bmi,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_log_prob = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=0.001,
        label="C2(b) Compute the predicted probability P(Y = 1) for the given patient:",
        value=submission_default("Q_PP_LOG_PROB"),
    )
    q_pp_log_prob
    return (q_pp_log_prob,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_log_bmi_factor = mo.ui.number(
        start=0.0,
        stop=5.0,
        step=0.001,
        label="C2(c) Compute the multiplicative change in the odds if BMI_category changes from 2 to 1:",
        value=submission_default("Q_PP_LOG_BMI_FACTOR"),
    )
    q_pp_log_bmi_factor
    return (q_pp_log_bmi_factor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise C3 - LIME Step by Step

        A black-box sepsis-risk model is probed with 4 perturbations:

        | temp | HR | model output |
        |---|---|---|
        | 1 | 1 | 0.85 |
        | 1 | 0 | 0.60 |
        | 0 | 1 | 0.55 |
        | 0 | 0 | 0.20 |

        C3(a) Complete the weighted least-squares LIME objective

        $$
        \begin{aligned}
        \min_{w_0,w_1,w_2} \quad
        &a_0 \cdot (b_0 - w_0 - w_1 - w_2)^2 \\
        &+ a_1 \cdot (b_1 - w_0 - w_1)^2 \\
        &+ a_2 \cdot (b_2 - w_0 - w_2)^2 \\
        &+ a_3 \cdot (b_3 - w_0)^2
        \end{aligned}
        $$

        by identifying the values of $a_i$ and $b_i$ from the table.

        Interpret the effect of heart rate under different temperature settings.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lime_a0 = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.01,
        label='Enter the coefficient $a_0$:',
        value=submission_default("Q_PP_LIME_A0"),
    )
    q_pp_lime_a0
    return (q_pp_lime_a0,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lime_a1 = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.01,
        label='Enter the coefficient $a_1$:',
        value=submission_default("Q_PP_LIME_A1"),
    )
    q_pp_lime_a1
    return (q_pp_lime_a1,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lime_a2 = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.01,
        label='Enter the coefficient $a_2$:',
        value=submission_default("Q_PP_LIME_A2"),
    )
    q_pp_lime_a2
    return (q_pp_lime_a2,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lime_a3 = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.01,
        label='Enter the coefficient $a_3$:',
        value=submission_default("Q_PP_LIME_A3"),
    )
    q_pp_lime_a3
    return (q_pp_lime_a3,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lime_b0 = mo.ui.number(
        start=-1.0,
        stop=1.0,
        step=0.01,
        label='Enter the coefficient $b_0$:',
        value=submission_default("Q_PP_LIME_B0"),
    )
    q_pp_lime_b0
    return (q_pp_lime_b0,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lime_b1 = mo.ui.number(
        start=-1.0,
        stop=1.0,
        step=0.01,
        label='Enter the coefficient $b_1$:',
        value=submission_default("Q_PP_LIME_B1"),
    )
    q_pp_lime_b1
    return (q_pp_lime_b1,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lime_b2 = mo.ui.number(
        start=-1.0,
        stop=1.0,
        step=0.01,
        label='Enter the coefficient $b_2$:',
        value=submission_default("Q_PP_LIME_B2"),
    )
    q_pp_lime_b2
    return (q_pp_lime_b2,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lime_b3 = mo.ui.number(
        start=-1.0,
        stop=1.0,
        step=0.01,
        label='Enter the coefficient $b_3$:',
        value=submission_default("Q_PP_LIME_B3"),
    )
    q_pp_lime_b3
    return (q_pp_lime_b3,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lime_hr_present = mo.ui.number(
        start=-2.0,
        stop=2.0,
        step=0.01,
        label="C3(b) Compute the effect of heart rate when temperature is present: f(1,1) - f(1,0)",
        value=submission_default("Q_PP_LIME_HR_PRESENT"),
    )
    q_pp_lime_hr_present
    return (q_pp_lime_hr_present,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lime_hr_absent = mo.ui.number(
        start=-2.0,
        stop=2.0,
        step=0.01,
        label="C3(c) Compute the effect of heart rate when temperature is absent: f(0,1) - f(0,0)",
        value=submission_default("Q_PP_LIME_HR_ABSENT"),
    )
    q_pp_lime_hr_absent
    return (q_pp_lime_hr_absent,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_lime_limitation_options = {
            "LIME's local linear surrogate cannot represent the interaction exactly and must average the effect.": "a",
            "LIME proves the model is globally linear.": "b",
            "LIME shows that heart rate has zero effect once temperature is included.": "c",
            "LIME guarantees causal interpretation of the interaction.": "d",
    }
    q_pp_lime_limitation = mo.ui.radio(
        options=q_pp_lime_limitation_options,
        label="C3(d) Select the correct key insight:",
        value=submission_radio_default("Q_PP_LIME_LIMITATION", q_pp_lime_limitation_options),
    )
    q_pp_lime_limitation
    return (q_pp_lime_limitation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise C4 - Insertion and Deletion

        Consider a single test sample with 3 features:
        - `lactate` with importance score `0.60`
        - `mean_arterial_pressure` with importance score `0.30`
        - `heart_rate` with importance score `0.10`

        The model prediction for this sample is available for every feature subset:

        - $f(\{\}) = 0.10$
        - $f(\{\text{lactate}\}) = 0.45$
        - $f(\{\text{mean\_arterial\_pressure}\}) = 0.25$
        - $f(\{\text{heart\_rate}\}) = 0.15$
        - $f(\{\text{lactate}, \text{mean\_arterial\_pressure}\}) = 0.70$
        - $f(\{\text{lactate}, \text{heart\_rate}\}) = 0.55$
        - $f(\{\text{mean\_arterial\_pressure}, \text{heart\_rate}\}) = 0.32$
        - $f(\{\text{lactate}, \text{mean\_arterial\_pressure}, \text{heart\_rate}\}) = 0.82$

        Compute each score as the trapezoidal AUC over the 4 equally spaced
        steps. If the curve values are $y_0, y_1, y_2, y_3$, use

        $\mathrm{AUC} = \frac{1}{3}\left(\frac{y_0 + y_1}{2} + \frac{y_1 + y_2}{2} + \frac{y_2 + y_3}{2}\right)$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_ins_auc = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=0.001,
        label="C4(a) Compute the insertion score (trapezoidal AUC):",
        value=submission_default("Q_PP_INS_AUC"),
    )
    q_pp_ins_auc
    return (q_pp_ins_auc,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_del_auc = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=0.001,
        label="C4(b) Compute the deletion score (trapezoidal AUC):",
        value=submission_default("Q_PP_DEL_AUC"),
    )
    q_pp_del_auc
    return (q_pp_del_auc,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_insdel_order_options = {
            "lactate -> mean_arterial_pressure -> heart_rate": "a",
            "heart_rate -> mean_arterial_pressure -> lactate": "b",
            "mean_arterial_pressure -> lactate -> heart_rate": "c",
            "The ordering cannot be determined from the attribution scores": "d",
    }
    q_pp_insdel_order = mo.ui.radio(
        options=q_pp_insdel_order_options,
        label="C4(c) Which feature order should be used for both insertion and deletion?",
        value=submission_radio_default("Q_PP_INSDEL_ORDER", q_pp_insdel_order_options),
    )
    q_pp_insdel_order
    return (q_pp_insdel_order,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Part 2 - Week 5 Foundations
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            We include the week 5 exercises on:

            1. Shapley values,
            2. closed-form SHAP for a linear model,
            3. the KernelSHAP kernel,
            4. Grad-CAM,
            5. xAI method selection.
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise C5 - Shapley Values for a Clinical Model

        A model predicts ICU mortality risk using 3 features with the following
        coalition values:

        $\mathrm{val}(\{\}) = 0.30$

        $\mathrm{val}(\{\text{age}\}) = 0.38$

        $\mathrm{val}(\{\text{lactate}\}) = 0.50$

        $\mathrm{val}(\{\text{MAP}\}) = 0.35$

        $\mathrm{val}(\{\text{age}, \text{lactate}\}) = 0.62$

        $\mathrm{val}(\{\text{age}, \text{MAP}\}) = 0.42$

        $\mathrm{val}(\{\text{lactate}, \text{MAP}\}) = 0.58$

        $\mathrm{val}(\{\text{age}, \text{lactate}, \text{MAP}\}) = 0.72$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        C5(a) Calculate the Shapley value for each feature.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_shap_lactate = mo.ui.number(
        start=-1.0,
        stop=1.0,
        step=0.001,
        label='$\\phi(\\mathrm{lactate})$',
        value=submission_default("Q_PP_SHAP_LACTATE"),
    )
    q_pp_shap_lactate
    return (q_pp_shap_lactate,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_shap_age = mo.ui.number(
        start=-1.0,
        stop=1.0,
        step=0.001,
        label='$\\phi(\\mathrm{age})$',
        value=submission_default("Q_PP_SHAP_AGE"),
    )
    q_pp_shap_age
    return (q_pp_shap_age,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_shap_map = mo.ui.number(
        start=-1.0,
        stop=1.0,
        step=0.001,
        label='$\\phi(\\mathrm{MAP})$',
        value=submission_default("Q_PP_SHAP_MAP"),
    )
    q_pp_shap_map
    return (q_pp_shap_map,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_shap_efficiency = mo.ui.number(
        start=-1.0,
        stop=1.0,
        step=0.001,
        label='C5(b) Compute the efficiency-check quantity: val({age, lactate, MAP}) - val({}):',
        value=submission_default("Q_PP_SHAP_EFFICIENCY"),
    )
    q_pp_shap_efficiency
    return (q_pp_shap_efficiency,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise C6 - Closed-Form Shapley Values

        A linear model predicts HbA1c:

        $$
        \hat{f}(x) = 5.0 + 0.02 \cdot \text{age} + 0.15 \cdot \text{BMI} - 0.1 \cdot \text{exercise\_hours}
        $$

        Population means:
        - $E[\text{age}] = 50$
        - $E[\text{BMI}] = 25$
        - $E[\text{exercise}] = 3$

        Patient values:
        - $\text{age} = 60$
        - $\text{BMI} = 32$
        - $\text{exercise} = 1$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        C6(a) Compute the Shapley value for each feature using the closed-form.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_closed_bmi = mo.ui.number(
        start=-5.0,
        stop=5.0,
        step=0.01,
        label='$\\phi(\\mathrm{BMI})$',
        value=submission_default("Q_PP_CLOSED_BMI"),
    )
    q_pp_closed_bmi
    return (q_pp_closed_bmi,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_closed_efficiency_holds_options = {
            "Yes, the efficiency criterion holds for this patient example.": "a",
            "No, the efficiency criterion does not hold for this patient example.": "b",
    }
    q_pp_closed_efficiency_holds = mo.ui.radio(
        options=q_pp_closed_efficiency_holds_options,
        label="C6(b) Does the efficiency criterion hold for this patient example?",
        value=submission_radio_default("Q_PP_CLOSED_EFFICIENCY_HOLDS", q_pp_closed_efficiency_holds_options),
    )
    q_pp_closed_efficiency_holds
    return (q_pp_closed_efficiency_holds,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_closed_modifiable_options = {
            "BMI contributes most, and it is a modifiable risk factor.": "a",
            "Age contributes most, and it is a modifiable risk factor.": "b",
            "Exercise contributes most, and it is not modifiable.": "c",
            "All three contribute equally, so no intervention is suggested.": "d",
    }
    q_pp_closed_modifiable = mo.ui.radio(
        options=q_pp_closed_modifiable_options,
        label="C6(c) Which statement is correct?",
        value=submission_radio_default("Q_PP_CLOSED_MODIFIABLE", q_pp_closed_modifiable_options),
    )
    q_pp_closed_modifiable
    return (q_pp_closed_modifiable,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise C7 - KernelSHAP Kernel

        For a model with `M = 4` features, the KernelSHAP kernel is:

        $$
        \pi_x(z') = \frac{M-1}{\binom{M}{|z'|}\, |z'| \, (M-|z'|)}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        C7(a) Compute the KernelSHAP weight $\pi_x(z')$ for each coalition size.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_kernel_w1 = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.001,
        label="For coalition size |z'| = 1:",
        value=submission_default("Q_PP_KERNEL_W1"),
    )
    q_pp_kernel_w1
    return (q_pp_kernel_w1,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_kernel_w2 = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.001,
        label="For coalition size |z'| = 2:",
        value=submission_default("Q_PP_KERNEL_W2"),
    )
    q_pp_kernel_w2
    return (q_pp_kernel_w2,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_kernel_w3 = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.001,
        label="For coalition size |z'| = 3:",
        value=submission_default("Q_PP_KERNEL_W3"),
    )
    q_pp_kernel_w3
    return (q_pp_kernel_w3,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_kernel_highest_options = {
            "Coalitions of size 2 only": "a",
            "Coalitions of size 1 and 3": "b",
            "Coalitions of size 0 and 4": "c",
            "All coalition sizes receive the same weight": "d",
    }
    q_pp_kernel_highest = mo.ui.radio(
        options=q_pp_kernel_highest_options,
        label="Which coalition sizes receive the highest weight?",
        value=submission_radio_default("Q_PP_KERNEL_HIGHEST", q_pp_kernel_highest_options),
    )
    q_pp_kernel_highest
    return (q_pp_kernel_highest,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_kernel_exclude_options = {
            "Because |z'|=0 and |z'|=M are fixed baseline/full-prediction reference points and are not informative for attribution sampling.": "a",
            "Because KernelSHAP only works for binary classifiers.": "b",
            "Because those coalition sizes make the Shapley values sum to zero.": "c",
            "Because the kernel is undefined for all other coalition sizes.": "d",
    }
    q_pp_kernel_exclude = mo.ui.radio(
        options=q_pp_kernel_exclude_options,
        label="C7(b) Why are |z'| = 0 and |z'| = M excluded?",
        value=submission_radio_default("Q_PP_KERNEL_EXCLUDE", q_pp_kernel_exclude_options),
    )
    q_pp_kernel_exclude
    return (q_pp_kernel_exclude,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_kernel_intuition_options = {
            "Because small and large coalitions reveal a feature's individual contribution more clearly, while middle-sized coalitions contain more confounding interactions.": "a",
            "Because KernelSHAP is only valid when coalition sizes are close to M/2.": "b",
            "Because coalitions of size 1 and M-1 always make the model linear.": "c",
            "Because the Shapley axioms require equal weights only for middle-sized coalitions.": "d",
    }
    q_pp_kernel_intuition = mo.ui.radio(
        options=q_pp_kernel_intuition_options,
        label="C7(c) Why do coalitions of size 1 and M - 1 receive higher weights than coalitions of size M/2?",
        value=submission_radio_default("Q_PP_KERNEL_INTUITION", q_pp_kernel_intuition_options),
    )
    q_pp_kernel_intuition
    return (q_pp_kernel_intuition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise C8 - Grad-CAM Computation

        A CNN classifies chest X-rays. For the class `pneumonia`, the final
        convolutional layer has 3 channels, each with a `2 x 2` activation map.
        Compute the Grad-CAM channel weights and then compute

        $$
        L^c = \mathrm{ReLU}\left(\sum_k \alpha_k^c A_k\right)
        $$

        using the matrix form

        $$
        L^c =
        \begin{pmatrix}
        \ell_{11} & \ell_{12} \\
        \ell_{21} & \ell_{22}
        \end{pmatrix}.
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        C8(a) Compute the Grad-CAM channel weights.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_gradcam_a1 = mo.ui.number(
        start=-2.0,
        stop=2.0,
        step=0.01,
        label=r"For channel weight $\alpha_1^c$:",
        value=submission_default("Q_PP_GRADCAM_A1"),
    )
    q_pp_gradcam_a1
    return (q_pp_gradcam_a1,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_gradcam_a2 = mo.ui.number(
        start=-2.0,
        stop=2.0,
        step=0.01,
        label=r"For channel weight $\alpha_2^c$:",
        value=submission_default("Q_PP_GRADCAM_A2"),
    )
    q_pp_gradcam_a2
    return (q_pp_gradcam_a2,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_gradcam_a3 = mo.ui.number(
        start=-2.0,
        stop=2.0,
        step=0.01,
        label=r"For channel weight $\alpha_3^c$:",
        value=submission_default("Q_PP_GRADCAM_A3"),
    )
    q_pp_gradcam_a3
    return (q_pp_gradcam_a3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        C8(b) Compute the four entries of the Grad-CAM heatmap matrix $L^c$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_gradcam_l11 = mo.ui.number(
        start=-5.0,
        stop=5.0,
        step=0.01,
        label=r"For $\ell_{11}$:",
        value=submission_default("Q_PP_GRADCAM_L11"),
    )
    q_pp_gradcam_l11
    return (q_pp_gradcam_l11,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_gradcam_l12 = mo.ui.number(
        start=-5.0,
        stop=5.0,
        step=0.01,
        label=r"For $\ell_{12}$:",
        value=submission_default("Q_PP_GRADCAM_L12"),
    )
    q_pp_gradcam_l12
    return (q_pp_gradcam_l12,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_gradcam_l21 = mo.ui.number(
        start=-5.0,
        stop=5.0,
        step=0.01,
        label=r"For $\ell_{21}$:",
        value=submission_default("Q_PP_GRADCAM_L21"),
    )
    q_pp_gradcam_l21
    return (q_pp_gradcam_l21,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_gradcam_l22 = mo.ui.number(
        start=-5.0,
        stop=5.0,
        step=0.01,
        label=r"For $\ell_{22}$:",
        value=submission_default("Q_PP_GRADCAM_L22"),
    )
    q_pp_gradcam_l22
    return (q_pp_gradcam_l22,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_gradcam_toploc_options = {
            r"$\ell_{11}$": "a",
            r"$\ell_{12}$": "b",
            r"$\ell_{21}$": "c",
            r"$\ell_{22}$": "d",
    }
    q_pp_gradcam_toploc = mo.ui.radio(
        options=q_pp_gradcam_toploc_options,
        label="C8(c) Which spatial location has the highest activation after ReLU?",
        value=submission_radio_default("Q_PP_GRADCAM_TOPLOC", q_pp_gradcam_toploc_options),
    )
    q_pp_gradcam_toploc
    return (q_pp_gradcam_toploc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise C9 - Method Selection

        Match each scenario to the most appropriate xAI method.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_method_tree_options = {
            "TreeSHAP": "a",
            "Grad-CAM": "b",
            "KernelSHAP only": "c",
            "No xAI method is appropriate": "d",
    }
    q_pp_method_tree = mo.ui.radio(
        options=q_pp_method_tree_options,
        label="C9(a) Random forest for 30-day readmission risk; explain each patient's risk:",
        value=submission_radio_default("Q_PP_METHOD_TREE", q_pp_method_tree_options),
    )
    q_pp_method_tree
    return (q_pp_method_tree,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_method_image_options = {
            "TreeSHAP": "a",
            "Grad-CAM": "b",
            "Permutation importance": "c",
            "LIME for tabular data only": "d",
    }
    q_pp_method_image = mo.ui.radio(
        options=q_pp_method_image_options,
        label="C9(b) Deep model on retinal fundus images; clinician wants to see where the model looks:",
        value=submission_radio_default("Q_PP_METHOD_IMAGE", q_pp_method_image_options),
    )
    q_pp_method_image
    return (q_pp_method_image,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_method_genes_options = {
            "Grad-CAM": "a",
            "SHAP summary plot based on global |phi_j| values": "b",
            "Only confusion matrices": "c",
            "Nearest-neighbor search": "d",
    }
    q_pp_method_genes = mo.ui.radio(
        options=q_pp_method_genes_options,
        label="C9(c) Find the most important genes across all patients:",
        value=submission_radio_default("Q_PP_METHOD_GENES", q_pp_method_genes_options),
    )
    q_pp_method_genes
    return (q_pp_method_genes,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    q_pp_method_fairness_options = {
            "Grad-CAM on ethnicity": "a",
            "SHAP dependence plots for ethnicity, combined with fairness metrics": "b",
            "Only training loss curves": "c",
            "k-means clustering of predictions": "d",
    }
    q_pp_method_fairness = mo.ui.radio(
        options=q_pp_method_fairness_options,
        label="C9(d) Regulator asks whether an ICU mortality model discriminates based on ethnicity:",
        value=submission_radio_default("Q_PP_METHOD_FAIRNESS", q_pp_method_fairness_options),
    )
    q_pp_method_fairness
    return (q_pp_method_fairness,)


@app.cell(hide_code=True)
def _(
    mo,
    q_pp_closed_bmi,
    q_pp_closed_efficiency_holds,
    q_pp_closed_modifiable,
    q_pp_gradcam_a1,
    q_pp_gradcam_a2,
    q_pp_gradcam_a3,
    q_pp_gradcam_l11,
    q_pp_gradcam_l12,
    q_pp_gradcam_l21,
    q_pp_gradcam_l22,
    q_pp_gradcam_toploc,
    q_pp_ins_auc,
    q_pp_del_auc,
    q_pp_insdel_order,
    q_pp_kernel_exclude,
    q_pp_lime_a0,
    q_pp_lime_a1,
    q_pp_lime_a2,
    q_pp_lime_a3,
    q_pp_lime_b0,
    q_pp_lime_b1,
    q_pp_lime_b2,
    q_pp_lime_b3,
    q_pp_kernel_highest,
    q_pp_kernel_intuition,
    q_pp_kernel_w1,
    q_pp_kernel_w2,
    q_pp_kernel_w3,
    q_pp_lime_hr_absent,
    q_pp_lime_hr_present,
    q_pp_lime_limitation,
    q_pp_lin_eff_age,
    q_pp_lin_eff_comorb,
    q_pp_lin_eff_fit,
    q_pp_lin_eff_surg,
    q_pp_lin_fit_delta,
    q_pp_lin_major,
    q_pp_lin_pred,
    q_pp_log_bmi_factor,
    q_pp_log_or_bmi,
    q_pp_log_prob,
    q_pp_method_fairness,
    q_pp_method_genes,
    q_pp_method_image,
    q_pp_method_tree,
    q_pp_shap_age,
    q_pp_shap_efficiency,
    q_pp_shap_lactate,
    q_pp_shap_map,
):
    import json as _json
    from pathlib import Path as _Path

    submission = {
        "Q_PP_LIN_PRED": q_pp_lin_pred.value,
        "Q_PP_LIN_EFF_AGE": q_pp_lin_eff_age.value,
        "Q_PP_LIN_EFF_COMORB": q_pp_lin_eff_comorb.value,
        "Q_PP_LIN_EFF_FIT": q_pp_lin_eff_fit.value,
        "Q_PP_LIN_EFF_SURG": q_pp_lin_eff_surg.value,
        "Q_PP_LIN_MAJOR": q_pp_lin_major.value,
        "Q_PP_LIN_FIT_DELTA": q_pp_lin_fit_delta.value,
        "Q_PP_LOG_OR_BMI": q_pp_log_or_bmi.value,
        "Q_PP_LOG_PROB": q_pp_log_prob.value,
        "Q_PP_LOG_BMI_FACTOR": q_pp_log_bmi_factor.value,
        "Q_PP_LIME_A0": q_pp_lime_a0.value,
        "Q_PP_LIME_A1": q_pp_lime_a1.value,
        "Q_PP_LIME_A2": q_pp_lime_a2.value,
        "Q_PP_LIME_A3": q_pp_lime_a3.value,
        "Q_PP_LIME_B0": q_pp_lime_b0.value,
        "Q_PP_LIME_B1": q_pp_lime_b1.value,
        "Q_PP_LIME_B2": q_pp_lime_b2.value,
        "Q_PP_LIME_B3": q_pp_lime_b3.value,
        "Q_PP_LIME_HR_PRESENT": q_pp_lime_hr_present.value,
        "Q_PP_LIME_HR_ABSENT": q_pp_lime_hr_absent.value,
        "Q_PP_LIME_LIMITATION": q_pp_lime_limitation.value,
        "Q_PP_INS_AUC": q_pp_ins_auc.value,
        "Q_PP_DEL_AUC": q_pp_del_auc.value,
        "Q_PP_INSDEL_ORDER": q_pp_insdel_order.value,
        "Q_PP_SHAP_LACTATE": q_pp_shap_lactate.value,
        "Q_PP_SHAP_AGE": q_pp_shap_age.value,
        "Q_PP_SHAP_MAP": q_pp_shap_map.value,
        "Q_PP_SHAP_EFFICIENCY": q_pp_shap_efficiency.value,
        "Q_PP_CLOSED_BMI": q_pp_closed_bmi.value,
        "Q_PP_CLOSED_EFFICIENCY_HOLDS": q_pp_closed_efficiency_holds.value,
        "Q_PP_CLOSED_MODIFIABLE": q_pp_closed_modifiable.value,
        "Q_PP_KERNEL_W1": q_pp_kernel_w1.value,
        "Q_PP_KERNEL_W2": q_pp_kernel_w2.value,
        "Q_PP_KERNEL_W3": q_pp_kernel_w3.value,
        "Q_PP_KERNEL_HIGHEST": q_pp_kernel_highest.value,
        "Q_PP_KERNEL_EXCLUDE": q_pp_kernel_exclude.value,
        "Q_PP_KERNEL_INTUITION": q_pp_kernel_intuition.value,
        "Q_PP_GRADCAM_A1": q_pp_gradcam_a1.value,
        "Q_PP_GRADCAM_A2": q_pp_gradcam_a2.value,
        "Q_PP_GRADCAM_A3": q_pp_gradcam_a3.value,
        "Q_PP_GRADCAM_L11": q_pp_gradcam_l11.value,
        "Q_PP_GRADCAM_L12": q_pp_gradcam_l12.value,
        "Q_PP_GRADCAM_L21": q_pp_gradcam_l21.value,
        "Q_PP_GRADCAM_L22": q_pp_gradcam_l22.value,
        "Q_PP_GRADCAM_TOPLOC": q_pp_gradcam_toploc.value,
        "Q_PP_METHOD_TREE": q_pp_method_tree.value,
        "Q_PP_METHOD_IMAGE": q_pp_method_image.value,
        "Q_PP_METHOD_GENES": q_pp_method_genes.value,
        "Q_PP_METHOD_FAIRNESS": q_pp_method_fairness.value,
    }
    _submission_path = _Path(__file__).with_name("submission.json")
    _submission_path.write_text(_json.dumps(submission, indent=2))
    n_unanswered = sum(1 for v in submission.values() if v is None)
    mo.md(
        f"**Submission auto-saved to `{_submission_path.name}`** "
        f"({len(submission)} questions, {n_unanswered} unanswered)."
    )
    return


if __name__ == "__main__":
    app.run()


