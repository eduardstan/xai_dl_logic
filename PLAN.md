# Topic Modeling Enhancement Plan
*Addressing skewed cluster sizes and injecting domain knowledge*

## Current Situation

- **Problem**: Large heterogeneous cluster (891 papers) dominating the analysis
- **Cluster sizes**: 891, 174, 172, etc. - highly skewed distribution
- **Need**: Inject domain expertise to better separate XAI, symbolic, and sub-symbolic approaches
- **Constraint**: BERTopic seed words only support single-word exact matches

## Available Alternatives

### 1. **Seed Words with ClassTfidfTransformer** ⭐ *RECOMMENDED START*

**Description**: Use `ClassTfidfTransformer` to boost importance of domain-specific single words during topic representation.

**How it works**:
- Add `bertopic.vectorizers.ClassTfidfTransformer` to model setup
- Define seed words in config.yaml (separate from search keywords)
- Multiply importance of these words by configurable factor

**Pros**:
- ✅ Simple implementation - modify existing `bib_analyzer.py`
- ✅ Non-invasive - preserves current clustering algorithm
- ✅ Easily reversible with config flag
- ✅ Improves topic interpretability
- ✅ Can target specific domain concepts
- ✅ Low computational overhead

**Cons**:
- ❌ Limited to single-word exact matches only
- ❌ Indirect influence on clustering (mainly affects representation)
- ❌ May not fully solve large cluster problem
- ❌ Cannot use phrases like "decision tree" as single seed

**Implementation complexity**: **Low** (1-2 hours)

**Config additions needed**:
```yaml
domain_guidance:
  seed_words:
    enabled: true
    multiplier: 2.0
    words: ["explainable", "interpretable", "transparent", "symbolic", "logic", "formal", "neural", "deep", "network", "reasoning", "rule", "tree", "forest", "attention", "transformer"]
```

---

### 2. **Guided Topic Modeling**

**Description**: Use BERTopic's guided topic modeling to influence both clustering and representation using seed words.

**How it works**:
- Provide seed words that guide cluster formation
- Model learns additional topics beyond guided ones
- Directly influences embedding space during clustering

**Pros**:
- ✅ Direct influence on cluster formation
- ✅ Can break up large heterogeneous clusters
- ✅ Maintains discovery of new topics
- ✅ Balances human knowledge with data patterns

**Cons**:
- ❌ Medium implementation complexity
- ❌ Still limited to single-word seeds
- ❌ Requires careful parameter tuning
- ❌ May over-constrain natural topic emergence

**Implementation complexity**: **Medium** (4-6 hours)

**Required changes**: Significant modifications to model setup, new guided_topics parameter handling

---

### 3. **Semi-Supervised Topic Modeling**

**Description**: Pre-define topic categories and train model to assign documents while discovering additional topics.

**How it works**:
- Manually label subset of documents with known topics
- Model learns from these labels and discovers new patterns
- Hybrid supervised/unsupervised approach

**Pros**:
- ✅ Precise control over important domain topics
- ✅ Ensures critical categories are captured
- ✅ Can handle multi-word concepts through manual labeling
- ✅ High-quality topic separation

**Cons**:
- ❌ Requires manual document labeling (time-intensive)
- ❌ High implementation complexity
- ❌ Risk of bias toward predefined categories
- ❌ May miss unexpected patterns

**Implementation complexity**: **High** (1-2 days)

**Required changes**: New labeling workflow, model architecture changes, evaluation metrics

---

### 4. **Hierarchical Topic Modeling + Post-processing**

**Description**: Apply hierarchical clustering to break down large topics after initial analysis.

**How it works**:
- Run standard BERTopic analysis first
- Identify large clusters (>threshold size)
- Apply secondary clustering to split large topics
- Maintain topic hierarchy relationships

**Pros**:
- ✅ Targeted approach - only affects problematic clusters
- ✅ Preserves original analysis structure
- ✅ Flexible depth control
- ✅ Can be applied selectively

**Cons**:
- ❌ Two-step process increases complexity
- ❌ Doesn't inject domain knowledge directly
- ❌ May create artificial topic boundaries
- ❌ Harder to interpret hierarchical results

**Implementation complexity**: **Medium** (3-4 hours)

**Required changes**: Post-processing pipeline, new visualization methods

---

### 5. **Custom Representation Models**

**Description**: Use domain-specific representation models (KeyBERT, OpenAI, etc.) to improve topic naming and coherence.

**How it works**:
- Keep current clustering unchanged
- Apply domain-aware representation models
- Generate better topic labels and descriptions

**Pros**:
- ✅ Significant improvement in topic interpretability
- ✅ Can handle multi-word concepts in representations
- ✅ Non-invasive to clustering process
- ✅ Multiple representation methods available

**Cons**:
- ❌ Doesn't solve large cluster problem
- ❌ Mainly cosmetic improvements
- ❌ May require API keys (OpenAI) or additional models

**Implementation complexity**: **Medium** (2-3 hours)

**Required changes**: New representation model configuration, API handling

---

### 6. **Custom Embedding Fine-tuning**

**Description**: Fine-tune sentence transformer model on domain-specific academic literature.

**How it works**:
- Collect domain-specific training data
- Fine-tune embedding model on academic text pairs
- Use fine-tuned model for better semantic representations

**Pros**:
- ✅ Fundamental improvement in representation quality
- ✅ Better captures domain semantic relationships
- ✅ Potentially dramatic improvement in clustering

**Cons**:
- ❌ Very high complexity and time investment
- ❌ Requires significant computational resources
- ❌ Needs curated training data
- ❌ Risk of overfitting to training domain

**Implementation complexity**: **Very High** (1-2 weeks)

**Required changes**: Complete new training pipeline, model management, evaluation framework

---

## Recommended Implementation Strategy

### **Phase 1: Seed Words Implementation** 🎯

**Timeline**: 1-2 hours
**Goal**: Test if seed words can improve topic quality and potentially split large cluster

**Steps**:
1. Add seed word configuration to `config.yaml`
2. Implement `ClassTfidfTransformer` in `bib_analyzer.py`
3. Create easily toggleable feature
4. Test with current dataset
5. Evaluate impact on cluster sizes and topic quality

### **Phase 2: Evaluation & Next Steps** 📊

**Timeline**: 30 minutes analysis
**Goal**: Determine if additional approaches needed

**If Phase 1 successful**:
- Fine-tune seed words and multipliers
- Document best practices
- Consider custom representation models for better naming

**If Phase 1 insufficient**:
- Implement guided topic modeling (Phase 3)
- Consider hierarchical post-processing for large clusters

### **Phase 3: Advanced Techniques** (If needed) 🚀

**Timeline**: 4-6 hours
**Goal**: Stronger intervention if seed words don't solve the problem

**Options**:
- Guided topic modeling for direct clustering influence
- Hierarchical clustering for large topic breakdown
- Semi-supervised approach with manual labeling

---

## Success Metrics

### **Cluster Size Balance**:
- Target: No single cluster >30% of total documents
- Current: 891/3364 = 26.5% (already close, but could be better distributed)

### **Domain Topic Coverage**:
- Ensure distinct topics for: XAI, symbolic AI, sub-symbolic AI
- Clear separation between methodological approaches
- Interpretable topic representations

### **Model Quality**:
- Maintain or improve topic coherence scores
- Preserve meaningful topic relationships
- Keep computational efficiency

---

## Configuration Design

### New `config.yaml` Section:
```yaml
# Domain Knowledge Injection
domain_guidance:
  # Seed words for topic representation enhancement
  seed_words:
    enabled: true
    multiplier: 2.0
    # Single list of domain-relevant terms to boost in topic representations
    words: ["explainable", "interpretable", "transparent", "symbolic", "logic", "formal", "neural", "deep", "network", "reasoning", "rule", "tree", "forest", "attention", "transformer", "ontology", "knowledge", "embedding", "gradient", "svm", "linear", "ensemble"]
    
  # Future extensions
  guided_topics:
    enabled: false
    # Will be added in Phase 2 if needed
    
  representation_models:
    enabled: false
    # Will be added if custom representations needed
```

This approach allows for:
- ✅ Easy on/off toggle
- ✅ Fine-grained control over different domain areas
- ✅ Separation from search keywords
- ✅ Future extensibility
- ✅ Clear organization by investigation focus 