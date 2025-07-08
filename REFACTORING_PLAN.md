# Refactoring Plan: From Scripts to a Unified Analysis Framework

## 1. Executive Summary & Diagnosis

This document outlines a detailed plan to refactor the existing collection of analysis scripts into a professional, maintainable, and scalable Python project named **`research_analysis`**.

**Current State Diagnosis:**
-   **Fragmented Workflow:** The analysis is a three-stage pipeline executed by three separate Python scripts (`bib_analyzer.py`, `advanced_systematic_analyzer.py`, `create_enhanced_research_visualization.py`). This is fragile and hard to manage.
-   **Confusing Module Structure:** Functionality is split across many small files with overlapping and unclear responsibilities (e.g., `data_loader.py` vs. `data_loader_viz.py`; `report_generator.py` vs. `report_generation.py`).
-   **Monolithic Configuration:** A single, massive `config.yaml` makes it difficult to manage parameters for different stages of the pipeline.
-   **Scattered Outputs:** Artifacts are saved in two different top-level directories (`bertopic_analysis/` and `results/`), making it difficult to track the outputs of a single pipeline run.
-   **Lack of a Coherent Entry Point:** There is no single way to run the entire pipeline or its individual stages, which hinders usability and automation.

**The Vision: A Unified Framework**
The goal is to transform this project into an installable Python package with a clear command-line interface (CLI). A user should be able to install the package and run commands like `research-analysis run-pipeline` or `research-analysis run-stage --stage=bertopic`. This will make the entire process more robust, reproducible, and user-friendly.

---

## 2. Proposed Architecture

### 2.1. New Project Name
The project will be renamed to **`research_analysis`**.

### 2.2. New Directory Structure

The new structure will follow Python best practices, using a `src`-layout and organizing modules by function. Each stage of the pipeline will be its own submodule, preserving the modularity from the initial refactoring while providing high-level organization.

```
research_analysis/
├── configs/
│   ├── pipeline.yaml              # High-level settings, paths, logging
│   ├── stage_1_topic_model.yaml   # BERTopic, UMAP, HDBSCAN params
│   └── stage_2_selection.yaml     # Paper selection & metrics params
├── data/
│   └── merged.bib                 # Source bibliography data
├── pyproject.toml                 # Project definition and dependencies
├── README.md                      # Updated documentation
├── src/
│   └── research_analysis/
│       ├── __init__.py
│       ├── main.py                # Main CLI entry point logic
│       ├── config/
│       │   ├── __init__.py
│       │   ├── loader.py          # Pydantic-based config loading
│       │   └── models.py          # Pydantic models for config files
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── orchestrator.py    # Main pipeline execution logic
│       ├── stages/
│       │   ├── __init__.py
│       │   ├── stage_1_topic_model/
│       │   │   ├── __init__.py          # Orchestrates stage 1, calling other modules
│       │   │   ├── bib_parser.py
│       │   │   ├── embeddings.py
│       │   │   ├── model_setup.py
│       │   │   ├── training.py
│       │   │   └── outlier_reduction.py
│       │   ├── stage_2_paper_selection/
│       │   │   ├── __init__.py          # Orchestrates stage 2
│       │   │   ├── data_loader.py
│       │   │   ├── metrics.py
│       │   │   ├── selection.py
│       │   │   └── report.py
│       │   └── stage_3_visualization/
│       │       ├── __init__.py          # Orchestrates stage 3
│       │       ├── data_loader.py
│       │       ├── plots.py
│       │       └── report.py
│       └── utils/
│           ├── __init__.py
│           ├── files.py           # File I/O, caching, path helpers
│           └── logging.py         # Logging setup
└── outputs/                       # Unified output directory
    └── 20250708_103000/           # Example timestamped run directory
        ├── logs/
        │   └── app.log
        ├── stage_1_topic_model/
        │   ├── bertopic_model
        │   └── topic_info.csv
        ├── stage_2_paper_selection/
        │   ├── comprehensive_analysis.csv
        │   ├── selection_report.md
        │   └── ...
        └── stage_3_visualization/
            ├── metrics_overview.png
            ├── visualization_report.md
            └── ...
```

### 2.3. Key Architectural Decisions

1.  **Unified Output Directory (`outputs/`):** All pipeline runs will generate a single, timestamped directory inside `outputs/`. This directory will contain subdirectories for each stage's artifacts, solving the `bertopic_analysis/` vs `results/` problem and making each run self-contained.
2.  **Staged Configuration (`configs/`):** The `config.yaml` will be split into files that correspond to pipeline stages. Pydantic models in `src/research_analysis/config/` will load and validate them into a single, strongly-typed object.
3.  **Pipeline-Oriented Code (`src/research_analysis/`):**
    -   `main.py` will define the CLI using `Typer`.
    -   `pipeline/orchestrator.py` will contain the main function that calls each stage in sequence, passing data between them in memory.
    -   `stages/` will contain three submodules, one for each stage of the pipeline. The `__init__.py` in each stage submodule will act as a simple orchestrator, calling the more specialized modules within its own directory. This preserves modularity and avoids creating new monoliths.
4.  **Dismantling `utils.py`:** The monolithic `utils.py` will be broken apart. Logging setup will go into `utils/logging.py`. Caching and file helpers will go into `utils/files.py`. Domain-specific computations (like metrics) will be moved directly into the stage that uses them.

---

## 3. Incremental Refactoring Plan

This plan is designed to be executed in small, safe, and verifiable steps. I will pause for your approval after each commit.

### **Milestone 1: Establish the Project Foundation**

**➡️ Step 1.1: Create `pyproject.toml` and New Directory Structure**
-   **Action:**
    1.  Create `pyproject.toml` to define the `research-analysis` project.
    2.  Create the `src/research_analysis` directory and the new `configs/` and `data/` directories.
    3.  Move `merged.bib` to `data/`.
-   **Rationale:** Establishes the modern Python package structure.
-   **Testing:** Run `pip install -e .` to confirm the package is installable.
-   **Commit:** `refactor: Introduce pyproject.toml and new project structure`

**➡️ Step 1.2: Set Up Logging and Utilities**
-   **Action:**
    1.  Create `src/research_analysis/utils/logging.py` and move `setup_logging` from the old `utils.py` into it.
    2.  Create `src/research_analysis/utils/files.py` and move file-related helpers (caching, path creation) into it.
-   **Rationale:** Begins the careful dismantling of `utils.py` into organized utility modules.
-   **Testing:** We will test this implicitly in the next steps as we integrate it.
-   **Commit:** `refactor: Create dedicated logging and file utility modules`

---

### **Milestone 2: Overhaul Configuration**

**➡️ Step 2.1: Split `config.yaml` and Implement Pydantic Models**
-   **Action:**
    1.  Split `config.yaml` into the new files within the `configs/` directory.
    2.  Create `src/research_analysis/config/models.py` to define Pydantic classes for each new YAML file.
    3.  Create `src/research_analysis/config/loader.py` with a function that loads all YAMLs and merges them into a single, validated Pydantic object.
-   **Rationale:** Introduces a robust, self-documenting configuration system that is easy to manage and extend.
-   **Testing:** Write a temporary script to test that the new config loader works and correctly builds the config object.
-   **Commit:** `feat: Implement staged, Pydantic-based configuration system`

---

### **Milestone 3: Consolidate the Pipeline Stages**

**➡️ Step 3.1: Consolidate Stage 1 (Topic Modeling)**
-   **Action:**
    1.  Create the `src/research_analysis/stages/stage_1_topic_model/` subdirectory.
    2.  Move the logic from `bibliography.py`, `embeddings.py`, `bertopic_setup.py`, `topic_training.py`, and `outlier_reduction.py` into their corresponding new files within this directory.
    3.  The `__init__.py` file will contain a single function, `run_stage_1(config, output_dir)`, that orchestrates the calls to the other modules in its directory.
-   **Rationale:** Consolidates all related logic for the first stage into a cohesive, yet still modular, submodule.
-   **Testing:** We will write a temporary script that imports and calls `run_stage_1` to verify it produces the same results in the new `outputs/` structure.
-   **Commit:** `refactor: Consolidate topic modeling logic into stage_1 submodule`

**➡️ Step 3.2: Consolidate Stage 2 (Paper Selection)**
-   **Action:**
    1.  Create the `src/research_analysis/stages/stage_2_paper_selection/` subdirectory.
    2.  Move the logic from `advanced_systematic_analyzer.py`'s dependencies (`data_loader.py`, `topic_processor.py`, `paper_selection.py`, etc.) into new, aptly named modules inside this directory.
    3.  The `__init__.py` will contain a `run_stage_2(config, stage_1_results, output_dir)` function to orchestrate the stage.
-   **Rationale:** Follows the same consolidation pattern for the second stage, preserving modularity.
-   **Testing:** We will write a script to test `run_stage_2` using the artifacts from the previous test run.
-   **Commit:** `refactor: Consolidate paper selection logic into stage_2 submodule`

**➡️ Step 3.3: Consolidate Stage 3 (Visualization)**
-   **Action:**
    1.  Create the `src/research_analysis/stages/stage_3_visualization/` subdirectory.
    2.  Merge the logic from `create_enhanced_research_visualization.py` and its many `visualization_*.py` and `report_generation.py` dependencies into this new submodule.
    3.  The `__init__.py` will contain a `run_stage_3(config, stage_2_results, output_dir)` function.
-   **Rationale:** Completes the logical consolidation of the pipeline.
-   **Testing:** We will write a script to test `run_stage_3` and verify its output.
-   **Commit:** `refactor: Consolidate all visualization logic into stage_3 submodule`

---

### **Milestone 4: Create the Final Application**

**➡️ Step 4.1: Build the Pipeline Orchestrator and CLI**
-   **Action:**
    1.  Create `src/research_analysis/pipeline/orchestrator.py`. This will have a main `run_pipeline()` function that calls the three stage functions in order, passing data between them in memory and handling the creation of the timestamped output directory.
    2.  Create `src/research_analysis/main.py`. This will use `Typer` to create the CLI, with commands to run the full pipeline (`run-pipeline`) or individual stages.
    3.  Update `pyproject.toml` to link the CLI entry point to this file.
-   **Rationale:** This creates the final, user-facing application, providing a clean and powerful interface to the entire workflow.
-   **Testing:** Run the new CLI commands (e.g., `python -m research_analysis.main run-pipeline`) and verify they execute the full pipeline correctly.
-   **Commit:** `feat: Implement pipeline orchestrator and Typer CLI`

**➡️ Step 4.2: Final Cleanup and Documentation**
-   **Action:**
    1.  Delete all the old, now-unused scripts and modules from the root directory.
    2.  Thoroughly update `README.md` to reflect the new project name, structure, installation instructions, and CLI usage.
-   **Rationale:** Finalizes the refactoring, leaving a clean, professional, and well-documented project.
-   **Testing:** A final check of the repository to ensure no legacy files remain.
-   **Commit:** `chore: Remove legacy scripts and update documentation`

## 4. Next Steps

This plan is far more detailed and addresses the core architectural issues we've identified. Please review it. If you approve, I will delete the old plan and proceed with **Step 1.1: Create `pyproject.toml` and New Directory Structure**. 