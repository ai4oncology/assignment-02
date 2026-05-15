# ML for Health 2026 — Assignment Template

## Overview

This repository is the template for assignments in the **ML for Health 2026** course.
Each week a new assignment will be released. You will receive a link via ILIAS to accept it.

---

## Prerequisites

You will need the following installed on your machine:

- **git** — check with `git --version`
- **GitHub account** — required to access and submit assignments
- **miniconda** — recommended for managing Python environments

## Setting Up Your Environment

Create a dedicated conda environment for this course:

```bash
conda create -n ml4health python=3.11
conda activate ml4health
pip install -r requirements.txt
```

---

## Accepting and Working on an Assignment

1. Open the assignment link posted on **ILIAS** each week
2. On first use, link your GitHub account to your **university ID** from the provided list
3. A personal repository `<assignment-name>-<your-github-username>` will be created for you
4. Clone it locally:
   ```bash
   git clone git@github.com:ml4health-2026/<assignment-name>-<your-github-username>.git
   ```
5. Activate your environment and install dependencies:
   ```bash
   conda activate ml4health
   pip install -r requirements.txt
   ```
6. Open `assignment.py` and complete the `TODO` sections

---

## Testing Your Solution Locally

Run the test suite locally before submitting:

```bash
pytest tests/ -v
```

Fix any failing tests before pushing.

---

## Submitting

Push to the `main` branch to submit:

```bash
git add assignment.py experiences.md
git commit -m "Submit assignment"
git push origin main
```

Pushing to `main` triggers an automated test run via GitHub Actions.
**Only push to `main` when you are ready for final submission.**
For saving work-in-progress remotely, create a separate branch.

**Deadline:** Tuesday, 5 May 2026, end of day.

---

## Feedback

For each assignment, a pull request on a `feedback` branch will be created automatically.
Your instructors will leave inline comments there. **Do not merge this pull request.**

---

## Reporting Issues

If you could not solve part of the exercise, describe the problem in `experiences.md`.
General questions should be posted to the course forum on **ILIAS**.
