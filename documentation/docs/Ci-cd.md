---
id: Ci-cd
title: Continuous Integration & Deployment
sidebar_position: 4
---

# Continuous Integration & Deployment (CI/CD)

Modern software engineering relies on automation. The **CI/CD Pipeline** is the backbone that ensures our project remains stable, compilable, and correct at all times, despite frequent changes. We utilize **GitHub Actions** to orchestrate this process.

---

## 1. The Philosophy: "Fail Fast"

The goal of our CI is to catch errors **before** a human reviews the code.
*   If a developer pushes broken code, the pipeline turns **Red**.
*   Blocking rules prevent merging red code into the main branch.
*   This enforces a "Green Build Policy": the `main` branch is always stable, always deployable.

---

## 2. The Pipeline Workflow

Our workflow is defined in `.github/workflows/`. It triggers on two events: `push` (every save) and `pull_request`.

It consists of parallel jobs that execute in isolated Linux Containers (Runners).

### Job A: Compilation Check (`build_middleware`)
This is the health check.
*   **Environment**: `ubuntu-latest`.
*   **Action**: It checks out the code and runs `make`.
*   **Verification**: It specifically checks that the Makefile exists, that `make` produces the expected binary `pbrain-gomoku-ai`, and that no compilation errors occur for C++ source files.
*   *Why?* C++ is sensitive. A missing semicolon or a linker error breaks the app entirely.

### Job B: Coding Style Enforcer (`coding-style`)
We adhere to the strict **Epitech Coding Style**.
*   **Tool**: We use a specialized Docker container (`ghcr.io/epitech/coding-style-checker`).
*   **Action**: It scans every `.cpp` and `.hpp` file.
*   **Rules**: It checks for indentation, naming conventions, function length, and forbidden headers.
*   **Outcome**: If a single style error is found, the job fails and outputs a report detailing the file and line number (e.g., `Parser.cpp:42: Trailing whitespace`).

### Job C: Automated Testing (`run_tests`)
Compilation is not enough; the code must be correct.
*   **Tool**: We use the **Criterion** testing framework for C++.
*   **Action**: Runs `make tests_run`.
*   **Scope**:
    *   **Unit Tests**: Checks individual functions (e.g., `Board::checkWin` must return true for 5 aligned stones).
    *   **Integration Tests**: Checks the Protocol (e.g., Sending `START` returns `OK`).
*   **Outcome**: All asserts must pass.

### Job D: The Mirror Sync (`push_to_mirror`)
*   **Condition**: This job only runs if previous jobs (Build, Tests, Style) succeeded **AND** the push is on the `main` branch.
*   **Action**: It mirrors (copies) the repository content to a secondary remote repository (Epitech's submission server).
*   **Security**: Uses an SSH Private Key stored in GitHub Secrets (`SSH_PRIVATE_KEY`) to authenticate the push without exposing credentials.

---

## 3. Deployment Artifacts

While this is a game engine, "Deployment" in our context means generating a release-ready binary.
*   The pipeline ensures that the binary produced is static or has linked dependencies correctly.
*   It ensures the `gomoku_model.nn` weight file format is compatible with the engine version being built (versioning consistency).

---

## 4. Visual Feedback

Developers receive immediate feedback directly in the GitHub UI and via email/Slack notifications.
*   **Green Checkmark**: "Safe to merge".
*   **Red Cross**: "Fix required". Detailed logs are available to debug the exact compiler error or failed test case.
