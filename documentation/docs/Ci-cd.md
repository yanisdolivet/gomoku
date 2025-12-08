

---
id: Ci-cd
title: CI/CD
sidebar_position: 9
---

# CI/CD

The Gomoku AI project uses a Continuous Integration and Continuous Deployment (CI/CD) pipeline based on **GitHub Actions** to ensure code quality and automate delivery.

## Pipeline Architecture

The pipeline is orchestrated by the `.github/workflows/ci-orchestrator.yml` file. It coordinates the execution of several specialized modules.

### 1. Linting (Code Quality)

This step runs first to verify code compliance and commit messages.

- **Commit Linting**: Checks that commit messages follow the convention (e.g., `feat:`, `fix:`, `docs:`).
- **CPP Linting**: Analyzes source code with `cppcheck` to detect bugs and bad practices.

### 2. Build & Test

If linting passes, the project is compiled and tested.

- **Compilation**: Uses `make` to build the `pbrain-gomoku-ai` binary.
- **Clean Check**: Ensures `make clean` and `make fclean` work correctly.
- **Unit Tests**: Executed via `make tests_run` with coverage reporting.

### 3. Deployment (Main Branch Only)

These steps are triggered only on pushes to the `main` branch, and only if the previous steps succeeded.

#### Mirroring
The code is automatically pushed to the Epitech repository via `pixta-dev/repository-mirroring-action`.

#### Documentation
1.  **Doxygen**: Generates C++ API documentation from source comments.
2.  **Docusaurus**: Builds the user documentation site.
3.  **Deployment**: Automatically deploys the combined documentation to **GitHub Pages**.

## Configuration

Workflows are defined in `.github/workflows/`:

- `ci-orchestrator.yml`: Main entry point.
- `linting.yml`: Style checks.
- `build-and-test.yml`: Build and test procedures.
- `mirroring.yml`: Sync with Epitech.
- `documentation.yml`: Documentation generation and publishing.

