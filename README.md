# BERTopic Analysis for Academic Bibliography

A comprehensive topic modeling analysis of your `merged.bib` file using BERTopic with GPU acceleration and intelligent caching.

## 🚀 Features

- **GPU-Accelerated**: Automatically uses your GPU for faster embedding computation
- **Smart Caching**: Never recompute embeddings or models - everything is cached!
- **Interactive Visualizations**: Including the amazing DataMapPlot for exploring documents
- **Academic Optimized**: Configured specifically for academic literature analysis
- **Jupyter-Friendly**: Run step-by-step without losing progress

## 📊 Your Dataset

- **Total entries**: 3,364 bibliography entries
- **File size**: ~6.9 MB
- **Fields available**: title, abstract, keywords, author, year, etc.

## 🛠️ Quick Start

### 1. Activate Environment
```bash
conda activate xai_dl_logic
```

### 2. Verify Configuration
```bash
python -c "from bib_analyzer import load_config; print('✅ Configuration valid')"
```

### 3. Run Analysis

**Main Script**
```bash
python bib_analyzer.py
```

**Advanced Systematic Review**
```bash
python advanced_systematic_analyzer.py
```

**Enhanced Visualizations**
```bash
python create_enhanced_research_visualization.py
```

## 💡 Smart Caching System

The analysis uses intelligent caching for maximum efficiency:

1. **⚡ Fast Reruns**: Once you compute embeddings (2-3 minutes), they're cached forever
2. **🔄 Iterative**: Modify parameters and re-run only what you need
3. **🎯 GPU Accelerated**: Uses your RTX 2050 automatically
4. **📊 Immediate Results**: Cached results load in seconds
5. **🛡️ Reliable**: Hash-based validation ensures cache integrity

## 🎯 What You'll Get

### Visualizations
- **Topic Overview**: Interactive scatter plot of topic relationships
- **DataMapPlot**: Zoomable map of ALL 3,364 documents colored by topic
- **Hierarchy**: How topics relate to each other
- **Heatmap**: Topic similarity matrix
- **Bar Charts**: Most important terms per topic

### Data Files
- `bibliography_with_topics.csv`: Your original data + topic assignments
- `topic_info.csv`: Detailed information about each discovered topic
- Cached embeddings and models for fast re-runs

### Interactive Features
- Hover over documents to see content
- Zoom into topic clusters
- Explore topic relationships
- Filter and analyze specific topics

## ⚙️ Configuration

All settings are in `config.yaml`:

```yaml
# GPU settings
embedding_model:
  device: "cuda"  # Uses GPU automatically
  batch_size: 64  # Optimized for GPU

# Topic modeling parameters
data:
  min_topic_size: 10      # Minimum documents per topic
  nr_topics: "auto"       # Let BERTopic decide optimal number

# Caching (saves tons of time!)
output:
  cache_embeddings: true
  cache_dir: "cache"
```

## 🔧 Customization

### Change Topic Granularity
```yaml
# More topics (smaller, more specific)
min_topic_size: 5

# Fewer topics (larger, more general) 
min_topic_size: 20
```

### Try Different Models
```yaml
embedding_model:
  name: "all-mpnet-base-v2"  # Better quality, slower
  # or "all-MiniLM-L6-v2"   # Faster, good quality (default)
```

## 📁 Output Structure

```
bertopic_analysis/
├── results/
│   ├── bibliography_with_topics_YYYYMMDD_HHMMSS.csv
│   ├── topic_info_YYYYMMDD_HHMMSS.csv
│   └── analysis_summary_YYYYMMDD_HHMMSS.md
├── models/
│   └── bertopic_model_YYYYMMDD_HHMMSS/
├── plots/
│   ├── topics_overview.html
│   ├── documents_interactive_datamap.html
│   └── ...
└── cache/
    ├── embeddings_*.pkl
    ├── trained_model_*.pkl
    └── ...
```

## 🎯 Best Practices Followed

- **≤ 200 lines per function** ✅
- **Type hints and docstrings** ✅
- **Configuration-driven** ✅
- **Comprehensive logging** ✅
- **Error handling** ✅
- **Reproducible results** ✅
- **Modular design** ✅

## 🚨 Troubleshooting

### GPU Not Detected?
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Out of Memory?
Reduce batch size in `config.yaml`:
```yaml
embedding_model:
  batch_size: 32  # or 16
```

### Want to Start Fresh?
Delete the cache directory:
```bash
rm -rf cache/
```

## 📈 Performance

**With GPU (Recommended)**:
- Embeddings: ~2-3 minutes for 3,364 documents
- Model training: ~1-2 minutes
- Total first run: ~5 minutes
- **Subsequent runs: ~30 seconds** (thanks to caching!)

**CPU Only**:
- Embeddings: ~10-15 minutes
- Model training: ~3-5 minutes
- Total: ~20 minutes

## 🎉 Happy Topic Modeling!

Once you run the analysis, you'll discover the hidden topics in your academic bibliography and gain insights into the research landscape of your field!

**Questions?** Check the comments in the code or run the test script first. 