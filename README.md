# Assignment 02 - Model Explanations

**ML4Health 2026**

This assignment introduces two complementary explanation methods:

1. **LIME for NLP** - local explanations for a text classifier trained on
   `medical_pathology_200.csv`.
2. **SHAP for tabular data** - global and local explanations on the
   **Breast Cancer Wisconsin** dataset from scikit-learn.
3. **Pen and paper exercises** - week 4 and week 5 xAI foundations.

The structure matches `assignment-01`: one notebook drives the assignment, and
the implementation work lives in separate Python modules.

---

## Prerequisites

- **git** - `git --version`
- **GitHub account** with access to the assignment
- **miniconda** recommended for environment management

## 1. Set up your environment

```bash
conda create -n ml4health python=3.11
conda activate ml4health
pip install -r requirements.txt
```

## 2. Open the notebook

```bash
marimo edit notebook.py
```

Run cells from top to bottom.

## 3. Implement the TODOs

As you reach each section, fill in the functions marked `TODO`:

- **Section A** -> [`lime_exercise.py`](lime_exercise.py)
- **Section B** -> [`shap_exercise.py`](shap_exercise.py)

Section C is answered directly in the notebook via radio and number widgets.
Those answers are auto-saved to `submission.json`.

Do not change function signatures. The tests import these functions directly.

## 4. Self-check before submitting

Run the local test suite:

```bash
pytest tests/ -v
```

Because this repository is an assignment template, tests will fail until you
implement the missing functions.

## 5. Submit

Push to `main` when you are ready:

```bash
git add -A
git commit -m "final submission"
git push origin main
```

## Project layout

```text
.
|-- .github/
|   `-- workflows/
|       |-- classroom.yml
|       `-- tests.yml
|-- notebook.py              # single marimo notebook
|-- lime_exercise.py         # Section A: text classification + LIME
|-- shap_exercise.py         # Section B: breast-cancer classification + SHAP
|-- medical_pathology_200.csv # local dataset for Section A
|-- submission.json          # auto-saved notebook answers (generated after running)
|-- tests/
|   `-- test_assignment.py   # local + autograder tests
|-- requirements.txt
|-- experiences.md
`-- __init__.py
```

## Feedback and help

If you could not solve part of the exercise, describe what you tried in
[`experiences.md`](experiences.md). General questions go to the course forum on
ILIAS.
