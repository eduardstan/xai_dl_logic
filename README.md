# Research Analysis Framework

A unified, professional framework for systematic bibliography analysis using advanced topic modeling, paper selection, and visualization techniques.

## 🚀 Features

- **Professional CLI Interface**: Comprehensive command-line interface for all operations
- **Modular Pipeline Architecture**: Three-stage pipeline with consistent APIs
- **Advanced Topic Modeling**: BERTopic with UMAP, HDBSCAN, and outlier reduction
- **Intelligent Paper Selection**: Representative paper selection with diversity metrics
- **Rich Visualizations**: Interactive dashboards, network graphs, and statistical reports
- **GPU Acceleration**: Automatic GPU detection and optimization
- **Smart Caching**: Persistent caching for fast re-runs
- **Configuration Management**: Pydantic-based configuration with validation
- **Reproducible Research**: Complete configuration tracking and timestamped outputs

## 📦 Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, but recommended)

### Install from Source
```bash
git clone <repository-url>
cd research_analysis
conda create -n research_analysis python=3.11
conda activate research_analysis
pip install -e .
```

### Verify Installation
```bash
research-analysis --help
research-analysis validate-config
```

## 🎯 Quick Start

### Run Complete Pipeline
```bash
# Run all three stages in sequence
research-analysis run-pipeline

# With custom configuration directory
research-analysis run-pipeline --config-dir custom_configs

# With custom output directory
research-analysis run-pipeline --output-dir my_analysis
```

### Run Individual Stages
```bash
# Stage 1: Topic Modeling
research-analysis run-stage --stage=1 --output-dir outputs/my_run

# Stage 2: Paper Selection
research-analysis run-stage --stage=2 --output-dir outputs/my_run

# Stage 3: Visualization
research-analysis run-stage --stage=3 --output-dir outputs/my_run
```

### Configuration Management
```bash
# Validate configuration files
research-analysis validate-config

# Check version
research-analysis version
```

## 🏗️ Architecture

### Pipeline Stages

**Stage 1: Topic Modeling**
- Parse bibliography data (BibTeX format)
- Generate document embeddings using sentence transformers
- Apply UMAP dimensionality reduction
- Perform HDBSCAN clustering
- Train BERTopic model with representation learning
- Apply intelligent outlier reduction strategies

**Stage 2: Paper Selection**
- Load topic modeling artifacts
- Calculate paper metrics (centrality, diversity, research alignment)
- Apply selection strategies (ratio-based or threshold-based)
- Generate comprehensive analysis reports

**Stage 3: Visualization**
- Create static visualizations (metrics overview, selection analysis)
- Generate interactive dashboards (papers explorer, topic dashboard)
- Build network visualizations (paper assignments)
- Produce enhanced statistical reports

### Directory Structure
```
research_analysis/
├── configs/                    # Configuration files
│   ├── pipeline.yaml          # Global pipeline settings
│   ├── stage_1_topic_model.yaml
│   └── stage_2_selection.yaml
├── data/
│   └── merged.bib             # Source bibliography
├── src/research_analysis/
│   ├── main.py                # CLI entry point
│   ├── config/                # Configuration management
│   ├── pipeline/              # Stage orchestrators
│   ├── stages/                # Stage implementations
│   │   ├── stage_1_topic_model/
│   │   ├── stage_2_paper_selection/
│   │   └── stage_3_visualization/
│   └── utils/                 # Shared utilities
├── cache/                     # Persistent cache
└── outputs/                   # Timestamped run results
    └── YYYYMMDD_HHMMSS/       # Single run output
        ├── pipeline_config_used.yaml
        ├── stage_1_topic_model/
        ├── stage_2_paper_selection/
        └── stage_3_visualization/
```

## ⚙️ Configuration

The framework uses a three-level configuration system:

### Pipeline Configuration (`configs/pipeline.yaml`)
```yaml
paths:
  bibliography_file: "data/merged.bib"
  outputs: "outputs"
  cache: "cache"
  stage_1_path: "stage_1_topic_model"
  stage_2_path: "stage_2_paper_selection"
  stage_3_path: "stage_3_visualization"

logging:
  level: "INFO"
  rotation: "10 MB"
  retention: "1 week"

reproducibility:
  random_seed: 42
  timestamp_format: "%Y%m%d_%H%M%S"
```

### Stage 1 Configuration (`configs/stage_1_topic_model.yaml`)
```yaml
embedding_model:
  name: "all-MiniLM-L6-v2"
  device: "auto"  # auto-detects GPU
  batch_size: 64

umap_params:
  n_neighbors: 15
  n_components: 128
  min_dist: 0.0
  metric: "cosine"

hdbscan_params:
  min_cluster_size: 16
  cluster_selection_epsilon: 0.1

outlier_reduction:
  enabled: true
  strategies:
    - strategy: "c-tf-idf"
      threshold: 0.1
    - strategy: "embeddings"
      threshold: 0.7
```

### Stage 2 Configuration (`configs/stage_2_selection.yaml`)
```yaml
selection_strategy:
  method: "diverse_representative"
  count_method: "ratio_based"
  max_papers_per_cluster: 200

metrics:
  diversity_weight: 0.75
  similarity_threshold: 0.8
  use_iterative_selection: true
```

## 📊 Output Artifacts

### Stage 1 Outputs
- `bertopic_model/` - Trained BERTopic model
- `bibliography_with_topics.csv` - Documents with topic assignments
- `topic_info.csv` - Topic information and statistics
- `embeddings.pkl` - Document embeddings
- `topic_analysis_report.md` - Summary report

### Stage 2 Outputs
- `comprehensive_analysis.csv` - Complete paper analysis with metrics
- `selected_representatives.csv` - Selected representative papers
- `selection_summary.csv` - Topic-level selection statistics
- `selection_analysis_report.md` - Selection strategy report

### Stage 3 Outputs
- `metrics_overview.png` - Comprehensive metrics visualization
- `selection_analysis.png` - Selection strategy analysis
- `paper_assignment_networks.png` - Network visualizations
- `interactive_papers_explorer.html` - Interactive paper dashboard
- `topic_dashboard.html` - Topic comparison dashboard
- `enhanced_analysis_report.md` - Statistical analysis report

## 🚀 Performance

**Hardware Requirements:**
- **Recommended**: GPU with 4GB+ VRAM (RTX 2050 or better)
- **Minimum**: 8GB RAM, modern CPU

**Performance Benchmarks:**
- **With GPU**: ~5 minutes total (first run), ~30 seconds (cached reruns)
- **CPU Only**: ~20 minutes total (first run), ~2 minutes (cached reruns)

**Caching Strategy:**
- Global cache for embeddings and parsed data
- Stage-specific caching for artifacts
- Hash-based cache validation
- Automatic cache invalidation on config changes

## 🔧 Development

### Adding New Stages
1. Create stage directory in `src/research_analysis/stages/`
2. Implement stage modules following existing patterns
3. Add stage orchestrator in `src/research_analysis/pipeline/`
4. Update configuration models and CLI

### Running Tests
```bash
# Validate configuration
research-analysis validate-config

# Test individual stages
research-analysis run-stage --stage=1 --output-dir test_output

# Full pipeline test
research-analysis run-pipeline --output-dir test_full_pipeline
```

### Code Quality Standards
- ≤ 200 lines per file
- Type hints for all functions
- Comprehensive docstrings
- Pydantic models for configuration
- Structured logging throughout
- Error handling with user-friendly messages

## 📈 Research Applications

This framework is designed for:

- **Systematic Literature Reviews**: Automated topic discovery and paper selection
- **Research Trend Analysis**: Topic evolution and clustering insights
- **Bibliography Management**: Intelligent organization of large paper collections
- **Meta-Analysis Preparation**: Representative paper selection for detailed analysis
- **Research Landscape Mapping**: Visual exploration of research domains

## 🛠️ Troubleshooting

### GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# Force CPU mode
# Set device: "cpu" in configs/stage_1_topic_model.yaml
```

### Memory Issues
```bash
# Reduce batch size in configuration
embedding_model:
  batch_size: 32  # or 16 for very limited memory
```

### Cache Issues
```bash
# Clear cache for fresh start
rm -rf cache/

# Validate configuration
research-analysis validate-config
```

### Common Error Solutions
- **Config validation errors**: Check YAML syntax and required fields
- **Missing artifacts**: Ensure previous stages completed successfully
- **Memory errors**: Reduce batch sizes or use CPU mode
- **Permission errors**: Check file/directory permissions

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{research_analysis_framework,
  title = {Research Analysis Framework: A Unified System for Bibliography Analysis},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo/research_analysis}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BERTopic**: Maarten Grootendorst's excellent topic modeling library
- **UMAP**: McInnes, Healy, and Melville's dimensionality reduction algorithm
- **HDBSCAN**: Campello, Moulavi, and Sander's clustering algorithm
- **Sentence Transformers**: Nils Reimers' semantic embeddings library

---

**Questions or Issues?** Please check the troubleshooting section or open an issue on GitHub. 