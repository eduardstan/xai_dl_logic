#!/usr/bin/env python3
"""
Report generation for the paper selection stage.

This module creates a comprehensive Markdown report summarizing the outputs
of the paper selection and analysis process.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List

from research_analysis.config.models import AppConfig
from research_analysis.utils.logging import get_logger

logger = get_logger()


def generate_selection_report(
    config: AppConfig,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Generates and saves a comprehensive Markdown report for Stage 2.

    Args:
        config: The application configuration.
        results_df: DataFrame with all analyzed papers.
        summary_df: DataFrame with topic-level summary statistics.
        selected_df: DataFrame with only the selected representative papers.
        output_dir: The directory to save the report in.

    Returns:
        The path to the generated report file.
    """
    logger.info("Generating paper selection analysis report...")
    output_dir.mkdir(exist_ok=True, parents=True)
    report_path = output_dir / "selection_analysis_report.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        report_parts = [
            _generate_header(timestamp),
            _generate_executive_summary(results_df, selected_df, summary_df),
            _generate_selection_config(config),
            _generate_metrics_comparison(results_df, selected_df),
            _generate_research_alignment(selected_df),
            _generate_assignment_analysis(results_df),
            _generate_reading_recommendations(selected_df),
        ]
        report_content = "\n\n---\n\n".join(report_parts)

        with report_path.open("w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Successfully saved selection report to {report_path}")
        return report_path

    except Exception as e:
        logger.error(f"Failed to generate selection report: {e}", exc_info=True)
        raise


def _generate_header(timestamp: str) -> str:
    return (
        f"# Paper Selection & Analysis Report\n\n"
        f"**Generated on:** {timestamp}"
    )


def _generate_executive_summary(
    results_df: pd.DataFrame, selected_df: pd.DataFrame, summary_df: pd.DataFrame
) -> str:
    total_papers = len(results_df)
    total_selected = len(selected_df)
    selection_ratio = total_selected / total_papers if total_papers > 0 else 0
    total_topics = summary_df["topic_id"].nunique()

    return f"""## Executive Summary
- **Total Papers Analyzed:** {total_papers:,}
- **Representative Papers Selected:** {total_selected:,}
- **Selection Ratio:** {selection_ratio:.2%}
- **Topics Covered:** {total_topics}"""


def _generate_selection_config(config: AppConfig) -> str:
    cfg = config.stage_2
    return f"""## Selection Configuration
- **Selection Strategy:** `{cfg.selection_strategy.method}`
- **Count Method:** `{cfg.selection_strategy.count_method}`
- **Diversity Weight:** {cfg.metrics.diversity_weight}
- **Similarity Threshold:** {cfg.metrics.similarity_threshold}
- **Use Iterative Selection:** {cfg.metrics.use_iterative_selection}"""


def _generate_metrics_comparison(
    results_df: pd.DataFrame, selected_df: pd.DataFrame
) -> str:
    sections = ["## Metrics Comparison"]
    metrics = ["similarity_to_centroid", "diversity_score", "representativeness_score"]
    
    for metric in metrics:
        if metric not in results_df.columns or metric not in selected_df.columns:
            continue
            
        all_mean = results_df[metric].mean()
        all_std = results_df[metric].std()
        sel_mean = selected_df[metric].mean()
        sel_std = selected_df[metric].std()
        
        improvement = (sel_mean - all_mean) / all_mean if all_mean != 0 else 0
        
        sections.append(f"### {metric.replace('_', ' ').title()}")
        sections.append(
            "| Population | Mean | Std. Dev. |\n"
            "|------------|------|-----------|\n"
            f"| All Papers | {all_mean:.4f} | {all_std:.4f} |\n"
            f"| Selected   | {sel_mean:.4f} | {sel_std:.4f} |\n"
        )
        sections.append(f"**Improvement:** `{improvement:+.2%}`")

    return "\n\n".join(sections)


def _generate_research_alignment(selected_df: pd.DataFrame) -> str:
    alignment_cols = [c for c in selected_df.columns if "alignment" in c]
    if not alignment_cols:
        return ""

    rows = ["## Research Alignment (Selected Papers)"]
    for col in sorted(alignment_cols):
        avg_score = selected_df[col].mean()
        rows.append(f"- **{col.replace('_', ' ').title()}:** {avg_score:.2%}")

    return "\n".join(rows)


def _generate_assignment_analysis(results_df: pd.DataFrame) -> str:
    if "representative_title" not in results_df.columns:
        return ""

    non_selected = results_df[~results_df["is_selected_representative"]]
    assigned = non_selected[non_selected["representative_title"].notna()]
    
    if len(non_selected) == 0:
        return ""

    coverage = len(assigned) / len(non_selected) if len(non_selected) > 0 else 0
    avg_sim = assigned["similarity_to_representative"].mean() if len(assigned) > 0 else 0

    return f"""## Paper Assignment Analysis
- **Non-Selected Papers:** {len(non_selected):,}
- **Assigned to a Representative:** {len(assigned):,} ({coverage:.2%})
- **Average Assignment Similarity:** {avg_sim:.4f}"""


def _generate_reading_recommendations(selected_df: pd.DataFrame) -> str:
    sections = ["## Reading Recommendations"]

    # Priority 1: High Representativeness
    sections.append("### Priority 1: Highest Representativeness Score")
    top_5_rep = selected_df.nlargest(5, "representativeness_score")
    sections.append(_format_paper_list(top_5_rep))

    # Priority 2: Highest Representativeness per Topic
    sections.append("\n### Priority 2: Top Paper per Topic (Top 5 Topics by Size)")
    top_topics = selected_df['topic_id'].value_counts().nlargest(5).index
    
    top_papers_by_topic = (
        selected_df[selected_df['topic_id'].isin(top_topics)]
        .sort_values("representativeness_score", ascending=False)
        .groupby("topic_id")
        .first()
        .reset_index()
    )
    sections.append(_format_paper_list(top_papers_by_topic, by_topic=True))
    
    return "\n\n".join(sections)


def _format_paper_list(df: pd.DataFrame, by_topic: bool = False) -> str:
    if df.empty:
        return "No papers to recommend in this category."

    lines = []
    for _, paper in df.iterrows():
        title = paper.get("title", "N/A")
        authors = paper.get("author", "N/A")
        rep_score = paper.get('representativeness_score', 0)
        topic_id = paper.get('topic_id', -1)
        
        line = f"- **{title}**"
        if by_topic:
            line += f" (Topic {topic_id})"
        
        lines.append(line)
        lines.append(f"  - *Authors:* {authors}")
        lines.append(f"  - *Representativeness Score:* {rep_score:.4f}")
        
    return "\n".join(lines) 