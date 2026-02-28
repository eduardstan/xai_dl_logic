# Enhanced Research Landscape Visualization Report

**Generated:** 2026-02-11 17:51:59  
**Pipeline Stage:** 3 - Visualization and Analysis  
**Report Type:** Enhanced Statistics and Insights  

## Overview

This report summarizes the comprehensive visualization analysis of the systematic literature review, 
providing statistical insights and research recommendations based on the selected representative papers 
and their relationship to the broader research landscape.

---

## Executive Summary

### Key Statistics

- **Total Papers Analyzed:** 3,760
- **Representative Papers Selected:** 535
- **Selection Ratio:** 14.23%
- **Topics Identified:** 58
- **Average Cluster Size:** 64.6

### Selection Quality Metrics

- **Average Centrality Score:** 0.680
- **Average Diversity Score:** 0.345
- **Average Representativeness Score:** 0.496

### Research Coverage

The selected papers provide comprehensive coverage across 58 distinct research topics, 
with an overall selection ratio of 14.23%. This ensures both broad coverage and 
manageable scope for detailed review.

---

## Generated Visualizations

### Static Visualizations (PNG)

1. **metrics_overview.png**
   - Distribution analysis of centrality, diversity, and representativeness scores
   - Comparative histograms showing all papers vs. selected representatives
   - Scatter plots revealing relationships between key metrics
   - Research focus alignment analysis

2. **selection_analysis.png**
   - Papers selected vs. cluster size analysis
   - Selection ratio distribution across topics
   - Centrality vs. diversity trade-off visualization
   - Topic size category distribution

3. **paper_assignment_networks.png**
   - Network visualization of top 5 topics
   - Representative papers as central nodes
   - Assignment relationships between papers
   - Spatial layout based on similarity metrics

4. **topic_hierarchy.html** (Enhanced)
   - Interactive BERTopic hierarchy (dendrogram) using Plotly
   - Allows zooming and exploration of topic relationships

5. **topic_similarity_matrix.html** (Enhanced)
   - Interactive heatmap showing cosine similarity between all topic pairs

6. **topic_tree.txt** (Enhanced)
   - Hierarchical textual representation of the topic structure


### Statistical Analysis Visualizations (PNG)

4. **distribution_comparison.png**
   - Statistical distribution comparisons between selected and non-selected papers
   - Histogram overlays with median lines for each variable
   - Density plots showing distribution shapes and differences

5. **effect_size_forest_plot.png**
   - Cohen's d effect sizes with 95% confidence intervals
   - Color-coded by effect size magnitude (small, medium, large)
   - Reference lines for effect size thresholds

6. **temporal_analysis.png**
   - Publication year distribution analysis
   - Kolmogorov-Smirnov test results for temporal bias
   - Selection rate trends over time

7. **statistical_summary.png**
   - Mann-Whitney U test p-values (log scale)
   - Effect sizes for all analyzed variables
   - Significance indicators and effect size categories

### Interactive Visualizations (HTML)

1. **interactive_papers_explorer.html**
   - Interactive scatter plot of selected representatives
   - Hover information with paper details
   - Filterable by metrics and research alignment
   - Suitable for detailed paper exploration

2. **topic_dashboard.html**
   - Multi-panel dashboard for topic analysis
   - Interactive charts for cluster analysis
   - Selection efficiency metrics
   - Research alignment distribution

---

## Centroid vs Medoid Investigation
    
### Methodological Rationale
To address reviewer critique regarding the extraction of paper representatives, we conducted a rigorous comparative analysis between **Centroid-based** and **Medoid-based** selection strategies.
- **Centroid (Theoretical Center):** The mathematical mean of all document embeddings in a topic cluster. While precise, it represents a "virtual" paper that may not exist in the corpus.
- **Medoid (Empirical Center):** The actual paper within the cluster whose embedding has the minimum distance to all other papers. By design, the medoid *is* a representative paper.

### Comparison Metrics (Cross-Topic Averages)
- **Rank Correlation (Spearman $\rho$):** 0.8707
- **Top-5 Selection Overlap (Jaccard):** 0.4500
- **Top-5 Overlap Percentage:** 59.7%

### Analysis & Interpretation
1. **Consistency of Centrality:** The high Spearman correlation (0.8707) indicates that both methods are highly consistent in how they rank papers by "centrality." Papers near the theoretical centroid are almost always those also identified as medoids or near-medoids.
2. **Selection Sensitivity:** The moderate overlap (59.7%) in the top-5 selection suggests that while the "global" ranking is stable, the specific "local" choice of the top-1 or top-5 representatives can vary depending on whether one prioritizes the mathematical mean or an existing exemplar.
3. **Validation of Representative Extraction:** This comparison validates the use of centroid-based selection in our pipeline. Since papers highly similar to the centroid are also those identified by the medoid approach, using centroids as a reference point successfully extracts papers that are "representative by design," as requested by the reviewers.

### Strategic Conclusion
Our multi-metric approach (combining Centrality + Diversity) effectively mitigates the risks of relying on a single centrality definition. By ensuring that selected representatives satisfy both central positioning and local uniqueness, we provide a robust and representative snapshot of the HDBSCAN-identified research landscape.

---

## Topic Representative Details
The following table lists the primary representatives selected for each topic based on the original 3749 paper pool.

| topic_id | title                                                 | year  | similarity_to_centroid | similarity_to_medoid | diversity_score | representativeness_score |
| -------- | ----------------------------------------------------- | ----- | ---------------------- | -------------------- | --------------- | ------------------------ |
| 0        | {Fuzzy Rule-Based Explainer Systems for Deep Neura... | 2023  | 0.7293                 | 0.6821               | 0.287           | 0.486                    |
| 0        | {Logic-oriented fuzzy neural networks: A survey}...   | 2024  | 0.757                  | 0.6601               | 0.2952          | 0.503                    |
| 0        | {Using Resistin, Glucose, Age and BMI and Pruning ... | 2019  | 0.6595                 | 0.5401               | 0.3545          | 0.4917                   |
| 0        | {A novel and fast MIMO fuzzy inference system base... | 2017  | 0.6383                 | 0.519                | 0.3341          | 0.471                    |
| 0        | {Residual Sketch Learning for a Feature-Importance... | 2024  | 0.7163                 | 0.5846               | 0.2847          | 0.4789                   |
| 0        | {FRRI: A novel algorithm for fuzzy-rough rule indu... | 2025  | 0.6724                 | 0.6726               | 0.2633          | 0.4474                   |
| 0        | {A Lightweight TSK Fuzzy Classifier with Quantitat... | 2025  | 0.7363                 | 0.5277               | 0.2951          | 0.4936                   |
| 0        | {Multigranularity Fuzzy Autoencoder for Discrimina... | 2025  | 0.6873                 | 0.5289               | 0.3355          | 0.4938                   |
| 0        | {How to implement MCDM tools and continuous logic ... | 2020  | 0.6287                 | 0.5953               | 0.3971          | 0.5013                   |
| 0        | {Optimize TSK Fuzzy Systems for Classification Pro... | 2020  | 0.6782                 | 0.5169               | 0.2977          | 0.4689                   |
| 0        | {IC-FNN: A Novel Fuzzy Neural Network With Interpr... | 2018  | 0.7404                 | 0.592                | 0.3312          | 0.5153                   |
| 0        | {An explainable semi-supervised self-organizing fu... | 2022  | 0.6388                 | 0.5121               | 0.3867          | 0.5002                   |
| 0        | {Autonomous learning for fuzzy systems: a review}...  | 2023  | 0.7305                 | 0.6203               | 0.3354          | 0.5132                   |
| 0        | {PIE-RSPOP: A brain-inspired pseudo-incremental en... | 2018  | 0.6663                 | 0.5221               | 0.3921          | 0.5155                   |
| 0        | {A Multi-Agent Architecture for the Design of Hier... | 2019  | 0.5614                 | 0.4637               | 0.3907          | 0.4675                   |
| 0        | {A Bayesian approach to consequent parameter estim... | 2017  | 0.6303                 | 0.5088               | 0.3853          | 0.4956                   |
| 0        | {SCINN: Semantic Concept-Based Inference Neural Ne... | 2024  | 0.751                  | 0.6649               | 0.311           | 0.509                    |
| 0        | {Convolutional fuzzy modules stacked deep residual... | 2025  | 0.7257                 | 0.5547               | 0.2867          | 0.4843                   |
| 0        | {Neo-fuzzy neuron learning using backfitting algor... | 2019  | 0.6775                 | 0.4984               | 0.2981          | 0.4689                   |
| 0        | {On the interpretability of Fuzzy Cognitive Maps}...  | 2023  | 0.6263                 | 0.5781               | 0.3178          | 0.4566                   |
| 0        | {Bayesian network parameter learning using fuzzy c... | 2023  | 0.6521                 | 0.5777               | 0.3862          | 0.5059                   |
| 0        | {Neuro-Fuzzy Random Vector Functional Link Neural ... | 2024  | 0.722                  | 0.539                | 0.3535          | 0.5193                   |
| 0        | {UNFIS: A Novel Neuro-Fuzzy Inference System with ... | 2024  | 0.7918                 | 0.7352               | 0.2648          | 0.502                    |
| 0        | {Information orientation-based modular Type-2 fuzz... | 2024  | 0.6982                 | 0.5276               | 0.3926          | 0.5301                   |
| 0        | {Deep Neuro-Fuzzy System application trends, chall... | 2023  | 0.6381                 | 0.495                | 0.379           | 0.4956                   |
| 0        | {Interpretability Constraints for Fuzzy Modeling I... | 2018  | 0.6573                 | 0.6147               | 0.3853          | 0.5077                   |
| 0        | {Deep Fuzzy Rule-Based Classification System With ... | 2022  | 0.7672                 | 0.6797               | 0.27            | 0.4937                   |
| 0        | {Disjunctive Fuzzy Neural Networks: A New Splittin... | 2022  | 0.7987                 | 0.6393               | 0.3046          | 0.527                    |
| 0        | {IT2-ENFIS: Interval Type-2 Exclusionary Neuro-Fuz... | 2025  | 0.676                  | 0.5794               | 0.3526          | 0.4981                   |
| 0        | {A Linguistically Interpretable Deep Fuzzy Classif... | 2024  | 0.7058                 | 0.5793               | 0.2861          | 0.4749                   |
| 0        | {A Fully Interpretable First-Order TSK Fuzzy Syste... | 2023  | 0.7112                 | 0.5722               | 0.3275          | 0.5002                   |
| 1        | {MUTAN: Multimodal Tucker Fusion for Visual Questi... | 2017  | 0.6972                 | 0.6672               | 0.2993          | 0.4784                   |
| 1        | {Cognitive Vision and Perception}...                  | 2020  | 0.7432                 | 0.6519               | 0.2992          | 0.499                    |
| 1        | {Modularized Zero-shot VQA with Pre-trained Models... | 2023  | 0.7699                 | 0.7099               | 0.2732          | 0.4967                   |
| 1        | {Generic Attention-model Explainability for Interp... | 2021  | 0.7104                 | 0.6988               | 0.3012          | 0.4853                   |
| 1        | {Explainable Video Entailment with Grounded Visual... | 2021  | 0.7261                 | 0.5817               | 0.3349          | 0.511                    |
| 1        | {Visual Relation Detection Using Hybrid Analogical... | 2021  | 0.7211                 | 0.6454               | 0.3075          | 0.4936                   |
| 1        | {VSS: A Storage System for Video Analytics}...        | 2021  | 0.3486                 | 0.2491               | 0.5744          | 0.4728                   |
| 1        | {Learning the Dynamics of Visual Relational Reason... | 2022  | 0.7152                 | 0.6522               | 0.2848          | 0.4785                   |
| 1        | {Visual Graph Question Answering with ASP and LLMs... | 2025  | 0.7608                 | 0.6506               | 0.2803          | 0.4966                   |
| 1        | {Learning dynamics of attention: Human prior for i... | 2019  | 0.75                   | 0.6933               | 0.2613          | 0.4812                   |
| 1        | {Multi-modal Action Chain Abductive Reasoning}...     | 2023  | 0.7846                 | 0.6764               | 0.2816          | 0.508                    |
| 1        | {Doubly Right Object Recognition: A Why Prompt for... | 2023  | 0.7011                 | 0.5945               | 0.3536          | 0.5099                   |
| 1        | {Spatio-Temporal Graph for Video Captioning With K... | 2020  | 0.7579                 | 0.5756               | 0.3001          | 0.5061                   |
| 1        | {Integrating Language Guidance into Vision-based D... | 2022  | 0.705                  | 0.5561               | 0.3562          | 0.5131                   |
| 1        | {Inferring context from pixels for multimodal imag... | 2019  | 0.7019                 | 0.5702               | 0.3271          | 0.4958                   |
| 1        | {Raven's Progressive Matrices Completion with Late... | 2021  | 0.6842                 | 0.5997               | 0.3194          | 0.4836                   |
| 1        | {ProbVLM: Probabilistic Adapter for Frozen Vison-L... | 2023  | 0.6761                 | 0.5293               | 0.3306          | 0.486                    |
| 1        | {Hierarchical Human Parsing with Typed Part-Relati... | 2020  | 0.7144                 | 0.5753               | 0.2816          | 0.4764                   |
| 1        | {Are You Talking to Me? Reasoned Visual Dialog Gen... | 2018  | 0.7254                 | 0.6612               | 0.268           | 0.4738                   |
| 1        | {Control Image Captioning Spatially and Temporally... | 2021  | 0.6657                 | 0.5761               | 0.3208          | 0.476                    |
| 1        | {Interpretable and Globally Optimal Prediction for... | 2017  | 0.7637                 | 0.593                | 0.2694          | 0.4919                   |
| 1        | {Large Language Model with Curriculum Reasoning fo... | 2024  | 0.7505                 | 0.6406               | 0.3108          | 0.5087                   |
| 1        | {Neural module networks: A review}...                 | 2023  | 0.6078                 | 0.5315               | 0.3511          | 0.4666                   |
| 1        | {R2G: Reasoning to ground in 3D scenes}...            | 2025  | 0.7798                 | 0.6488               | 0.2744          | 0.5018                   |
| 1        | {Causal Reasoning Meets Visual Representation Lear... | 2022  | 0.7373                 | 0.6066               | 0.3065          | 0.5004                   |
| 1        | {AFSPrompt: An Axiomatic Fuzzy Set Prompt Pipeline... | 2025  | 0.7403                 | 0.6595               | 0.2813          | 0.4879                   |
| 1        | {Text-guided neural network training for image rec... | 2021  | 0.7323                 | 0.679                | 0.297           | 0.4928                   |
| 2        | {TRIPLETREE: A Versatile Interpretable Representat... | 2021  | 0.7362                 | 0.6418               | 0.3014          | 0.4971                   |
| 2        | {Differentiable Synthesis of Program Architectures... | 2021  | 0.5238                 | 0.4415               | 0.3684          | 0.4383                   |
| 2        | {Neurosymbolic Reinforcement Learning: Playing Min... | 2025  | 0.7607                 | 0.6063               | 0.31            | 0.5128                   |
| 2        | {Stochastic neural networks for hierarchical reinf... | 2017  | 0.7517                 | 0.6709               | 0.2887          | 0.4971                   |
| 2        | {$\pi$-Light: Programmatic Interpretable Reinforce... | 2024  | 0.7395                 | 0.6855               | 0.2755          | 0.4843                   |
| 2        | {Creativity of AI: Automatic Symbolic Option Disco... | 2022  | 0.7835                 | 0.6704               | 0.263           | 0.4972                   |
| 2        | {Reinforcement Learning for Control with Multiple ... | 2020  | 0.6333                 | 0.6472               | 0.3528          | 0.479                    |
| 2        | {Teaching Humans When To Defer to a Classifier via... | 2022  | 0.6736                 | 0.5509               | 0.3811          | 0.5127                   |
| 2        | {Inherently Explainable Reinforcement Learning in ... | 2022  | 0.7345                 | 0.5491               | 0.2706          | 0.4793                   |
| 2        | {GeoDRL: A Self-Learning Framework for Geometry Pr... | 2023  | 0.6127                 | 0.4535               | 0.3968          | 0.4939                   |
| 2        | {Learning-Aided Heuristics Design for Storage Syst... | 2021  | 0.6428                 | 0.5285               | 0.4113          | 0.5155                   |
| 2        | {Learning strategic network emergence games}...       | 2020  | 0.6844                 | 0.516                | 0.3777          | 0.5157                   |
| 2        | {Safe Exploration for Interactive Machine Learning... | 2019  | 0.6264                 | 0.5297               | 0.3578          | 0.4787                   |
| 2        | {Constrained Update Projection Approach to Safe Po... | 2022  | 0.6868                 | 0.5777               | 0.3578          | 0.5058                   |
| 2        | {Deep reinforcement learning with relational induc... | 2019  | 0.8098                 | 0.6088               | 0.2742          | 0.5152                   |
| 2        | {Advancing Sample Efficiency and Explainability in... | 2024  | 0.7796                 | 0.648                | 0.3102          | 0.5214                   |
| 2        | {XCS with opponent modelling for concurrent reinfo... | 2020  | 0.6945                 | 0.5863               | 0.3968          | 0.5307                   |
| 2        | {Explainable reinforcement learning for broad-XAI:... | 2023  | 0.7653                 | 0.574                | 0.3096          | 0.5146                   |
| 2        | {MIXRTs: Toward Interpretable Multi-Agent Reinforc... | 2025  | 0.7466                 | 0.6354               | 0.316           | 0.5098                   |
| 2        | {Hierarchical goals contextualize local reward dec... | 2023  | 0.7289                 | 0.5978               | 0.3117          | 0.4995                   |
| 3        | {Grad-SAM: Explaining Transformers via Gradient Se... | 2021  | 0.7476                 | 0.5939               | 0.3232          | 0.5142                   |
| 3        | {XplainLLM: A Knowledge-Augmented Dataset for Reli... | 2024  | 0.7427                 | 0.5445               | 0.299           | 0.4986                   |
| 3        | {Transfer learning for neural semantic parsing}...    | 2017  | 0.7386                 | 0.5671               | 0.2752          | 0.4837                   |
| 3        | {Transformer Feed-Forward Layers Build Predictions... | 2022  | 0.7635                 | 0.5885               | 0.329           | 0.5245                   |
| 3        | {Trigger-Argument based Explanation for Event Dete... | 2023  | 0.765                  | 0.6368               | 0.3035          | 0.5112                   |
| 3        | {Neural machine translation with synchronous laten... | 2021  | 0.7345                 | 0.5288               | 0.362           | 0.5296                   |
| 3        | {Toward controlled generation of text}...             | 2017  | 0.6996                 | 0.5409               | 0.3258          | 0.494                    |
| 3        | {A MULTI-GRAINED SELF-INTERPRETABLE SYMBOLIC-NEURA... | 2023  | 0.7898                 | 0.6662               | 0.3285          | 0.5361                   |
| 3        | {Do neural language models show preferences for sy... | 2020  | 0.7027                 | 0.5643               | 0.3235          | 0.4941                   |
| 3        | {Minimum Description Length Recurrent Neural Netwo... | 2022  | 0.7133                 | 0.6045               | 0.3784          | 0.5291                   |
| 3        | {Unsupervised Does Not Mean Uninterpretable: The C... | 2017  | 0.7243                 | 0.6085               | 0.3574          | 0.5225                   |
| 3        | {Do Transformer Interpretability Methods Transfer ... | 2025  | 0.7112                 | 0.6674               | 0.323           | 0.4977                   |
| 3        | {Learning explainable linguistic expressions with ... | 2020  | 0.7936                 | 0.6833               | 0.2988          | 0.5214                   |
| 3        | {Retrieve-and-Fill for Scenario-based Task-Oriente... | 2023  | 0.6831                 | 0.5179               | 0.2752          | 0.4587                   |
| 3        | {ReFT: Representation Finetuning for Language Mode... | 2024  | 0.7502                 | 0.5533               | 0.35            | 0.5301                   |
| 3        | {Interpretable and low-resource entity matching vi... | 2021  | 0.6335                 | 0.4053               | 0.4066          | 0.5087                   |
| 3        | {Interventional Rationalization}...                   | 2023  | 0.4774                 | 0.3963               | 0.4786          | 0.478                    |
| 3        | {A Generative Approach for Script Event Prediction... | 2023  | 0.7282                 | 0.5832               | 0.2883          | 0.4862                   |
| 3        | {AI-Generated vs. Human Text: Introducing a New Da... | 2025  | 0.6749                 | 0.5699               | 0.3776          | 0.5114                   |
| 3        | {ProtoryNet - Interpretable Text Classification Vi... | 2023  | 0.8058                 | 1.0                  | 0.2778          | 0.5154                   |
| 3        | {Holographic Declarative Memory: Distributional Se... | 2020  | 0.6241                 | 0.471                | 0.3662          | 0.4822                   |
| 3        | {Bag-of-Concepts representation for document class... | 2020  | 0.6085                 | 0.4607               | 0.4001          | 0.4939                   |
| 3        | {Integrating regular expressions into neural netwo... | 2024  | 0.7353                 | 0.6309               | 0.2605          | 0.4741                   |
| 4        | {Sparse Attacks for Manipulating Explanations in D... | 2023  | 0.7918                 | 0.6355               | 0.2829          | 0.5119                   |
| 4        | {Steganographic Passport: An Owner and User Verifi... | 2024  | 0.5981                 | 0.5076               | 0.4094          | 0.4943                   |
| 4        | {Minimax Risks and Optimal Procedures for Estimati... | 2023  | 0.5102                 | 0.3426               | 0.2761          | 0.3814                   |
| 4        | {ERM-KTP: Knowledge-Level Machine Unlearning via K... | 2023  | 0.71                   | 0.5128               | 0.3438          | 0.5086                   |
| 4        | {Attacks Meet Interpretability: Attribute-steered ... | 2018  | 0.7753                 | 0.7144               | 0.275           | 0.5002                   |
| 4        | {Interpret Neural Networks by Identifying Critical... | 2018  | 0.6594                 | 0.5796               | 0.3659          | 0.4979                   |
| 4        | {Understanding Adversarial Robustness of Vision Tr... | 2023  | 0.6819                 | 0.6815               | 0.3185          | 0.482                    |
| 4        | {Fast, Robust and Interpretable Participant Contri... | 2024  | 0.6746                 | 0.5083               | 0.3253          | 0.4825                   |
| 4        | {Sharing deep neural network models with interpret... | 2018  | 0.7548                 | 0.5756               | 0.3586          | 0.5369                   |
| 4        | {Evaluating the Validity of Word-level Adversarial... | 2024  | 0.7163                 | 0.6728               | 0.2718          | 0.4718                   |
| 4        | {Differentially Private Graph Neural Networks for ... | 2023  | 0.6656                 | 0.5079               | 0.2639          | 0.4447                   |
| 4        | {Wavelet regularization benefits adversarial train... | 2023  | 0.6842                 | 0.6636               | 0.3364          | 0.4929                   |
| 4        | {ARDST: An Adversarial-Resilient Deep Symbolic Tre... | 2024  | 0.7629                 | 0.661                | 0.31            | 0.5138                   |
| 5        | {DETERRENT: Knowledge Guided Graph Attention Netwo... | 2020  | 0.6583                 | 0.6198               | 0.2956          | 0.4589                   |
| 5        | {Interpretable Off-Policy Evaluation in Reinforcem... | 2020  | 0.5531                 | 0.4122               | 0.3418          | 0.4369                   |
| 5        | {Towards Interpretability and Personalization: A P... | 2021  | 0.6753                 | 0.511                | 0.3069          | 0.4727                   |
| 5        | {Retrieval-Based Gradient Boosting Decision Trees ... | 2022  | 0.8157                 | 0.7188               | 0.2812          | 0.5217                   |
| 5        | {Concurrent Multi-Label Prediction in Event Stream... | 2023  | 0.6792                 | 0.5956               | 0.3619          | 0.5047                   |
| 5        | {Recent Advances on Graph Analytics and Its Applic... | 2020  | 0.7393                 | 0.6964               | 0.2768          | 0.4849                   |
| 5        | {Automatic Emergency Diagnosis with Knowledge-Base... | 2020  | 0.7213                 | 0.5441               | 0.3077          | 0.4938                   |
| 5        | {Predicting sequenced dental treatment plans from ... | 2024  | 0.7459                 | 0.6646               | 0.2883          | 0.4942                   |
| 5        | {Learning using privileged information with logist... | 2024  | 0.6609                 | 0.5153               | 0.4503          | 0.5451                   |
| 5        | {A case-based ensemble learning system for explain... | 2020  | 0.7255                 | 0.5722               | 0.3741          | 0.5322                   |
| 5        | {X-SCSANet: Explainable Stack Convolutional Self-A... | 2025  | 0.6777                 | 0.5144               | 0.3077          | 0.4742                   |
| 5        | {Fully automated diagnosis of thyroid nodule ultra... | 2024  | 0.6963                 | 0.5343               | 0.3676          | 0.5155                   |
| 5        | {A dual medical ontology and relational graph fram... | 2025  | 0.7295                 | 0.6288               | 0.2786          | 0.4815                   |
| 5        | {Analyzing the Impact of Data Augmentation on the ... | 2024  | 0.6785                 | 0.5768               | 0.3062          | 0.4738                   |
| 5        | {Deep Reinforcement Learning for personalized diag... | 2024  | 0.7139                 | 0.6115               | 0.3095          | 0.4915                   |
| 5        | {Empowering early predictions: A paradigm shift in... | 2025  | 0.7044                 | 0.6283               | 0.2937          | 0.4785                   |
| 5        | {Human-guided deep learning with ante-hoc explaina... | 2023  | 0.7605                 | 0.6444               | 0.3022          | 0.5084                   |
| 5        | {Interpretable Machine Learning for COVID-19: An E... | 2023  | 0.6872                 | 0.5313               | 0.3378          | 0.495                    |
| 5        | {TGFN-SD: A text-guided multimodal fusion network ... | 2025  | 0.5798                 | 0.3942               | 0.4203          | 0.4921                   |
| 5        | {Detecting mental and physical disorders using mul... | 2024  | 0.599                  | 0.4862               | 0.3761          | 0.4764                   |
| 6        | {Vec2Node: Self-Training with Tensor Augmentation ... | 2023  | 0.5783                 | 0.3941               | 0.4336          | 0.4987                   |
| 6        | {Supervised Clustering for Subgroup Discovery: An ... | 2021  | 0.5406                 | 0.3386               | 0.3587          | 0.4406                   |
| 6        | {Clustering on Sparse Data in Non-overlapping Feat... | 2018  | 0.6643                 | 0.5414               | 0.4053          | 0.5219                   |
| 6        | {Interpretable Clustering via Multi-Polytope Machi... | 2022  | 0.6572                 | 0.4388               | 0.2953          | 0.4582                   |
| 6        | {Tensorized Label Learning on Anchor Graph}...        | 2024  | 0.7423                 | 0.5807               | 0.3025          | 0.5004                   |
| 6        | {Multi-Dictionary Tensor Decomposition}...            | 2023  | 0.6994                 | 0.5662               | 0.2875          | 0.4728                   |
| 6        | {Recent Developments in Boolean Matrix Factorizati... | 2020  | 0.6305                 | 0.5235               | 0.326           | 0.463                    |
| 6        | {Structurally Regularized Non-negative Tensor Fact... | 2017  | 0.6446                 | 0.5166               | 0.2756          | 0.4416                   |
| 6        | {Exploiting Combination Effect for Unsupervised Fe... | 2019  | 0.7626                 | 0.6425               | 0.3427          | 0.5317                   |
| 6        | {Multi-view unsupervised feature selection with te... | 2023  | 0.7102                 | 0.6294               | 0.3239          | 0.4977                   |
| 6        | {Robust Discriminant Subspace Clustering With Adap... | 2023  | 0.7509                 | 0.6581               | 0.3202          | 0.514                    |
| 6        | {Online Nonnegative CP-dictionary Learning for Mar... | 2022  | 0.739                  | 0.6294               | 0.295           | 0.4948                   |
| 6        | {Neural Implicit Flow: a mesh-agnostic dimensional... | 2023  | 0.5423                 | 0.4223               | 0.3784          | 0.4522                   |
| 6        | {Deep two-way matrix reordering for relational dat... | 2022  | 0.6068                 | 0.4951               | 0.4269          | 0.5079                   |
| 6        | {Sparse discriminant PCA based on contrastive lear... | 2023  | 0.6009                 | 0.5431               | 0.2963          | 0.4334                   |
| 7        | {HOOPS (sic): Human-in-the-Loop Graph Reasoning fo... | 2021  | 0.7376                 | 0.6086               | 0.3124          | 0.5037                   |
| 7        | {You Are What and Where You Are: Graph Enhanced At... | 2021  | 0.7427                 | 0.5667               | 0.2809          | 0.4887                   |
| 7        | {Automatically Discovering User Consumption Intent... | 2022  | 0.6892                 | 0.5438               | 0.3116          | 0.4815                   |
| 7        | {Automatic Meta-Path Discovery for Effective Graph... | 2022  | 0.7767                 | 0.676                | 0.2787          | 0.5028                   |
| 7        | {A Deep Markov Model for Clickstream Analytics in ... | 2022  | 0.5558                 | 0.4212               | 0.3221          | 0.4273                   |
| 7        | {Make It a Chorus: Knowledge- and Time-aware Item ... | 2020  | 0.6789                 | 0.5712               | 0.3253          | 0.4844                   |
| 7        | {Spatial-Temporal Interval Aware Sequential POI Re... | 2022  | 0.5605                 | 0.3811               | 0.3018          | 0.4182                   |
| 7        | {Towards Deeper, Lighter and Interpretable Cross N... | 2023  | 0.6614                 | 0.5547               | 0.2801          | 0.4517                   |
| 7        | {A Generic Reinforced Explainable Framework with K... | 2023  | 0.7745                 | 0.6798               | 0.2958          | 0.5112                   |
| 7        | {Knowledge-refined Denoising Network for Robust Re... | 2023  | 0.7419                 | 0.6778               | 0.2814          | 0.4887                   |
| 7        | {Relational social recommendation: Application to ... | 2019  | 0.6923                 | 0.5962               | 0.3328          | 0.4945                   |
| 7        | {KGTN: Knowledge Graph Transformer Network for exp... | 2023  | 0.7536                 | 0.7072               | 0.2864          | 0.4967                   |
| 7        | {Two-layer knowledge graph transformer network-bas... | 2025  | 0.7628                 | 0.684                | 0.2759          | 0.495                    |
| 7        | {Graph-ICF: Item-based collaborative filtering bas... | 2022  | 0.7917                 | 0.7282               | 0.2718          | 0.5057                   |
| 7        | {A survey on heterogeneous information network bas... | 2022  | 0.7763                 | 0.6448               | 0.2781          | 0.5023                   |
| 7        | {Knowledge&Social-based collaborative method with ... | 2025  | 0.7276                 | 0.6937               | 0.2734          | 0.4778                   |
| 7        | {Reinforced logical reasoning over KGs for interpr... | 2025  | 0.754                  | 0.657                | 0.2662          | 0.4857                   |
| 7        | {Accurate and Explainable Recommendation via Hiera... | 2021  | 0.6781                 | 0.5496               | 0.2883          | 0.4637                   |
| 8        | {Open Learner Models for Multi-activity Educationa... | 2021  | 0.66                   | 0.4853               | 0.3583          | 0.4941                   |
| 8        | {Knowledge Query Network for Knowledge Tracing}...    | 2019  | 0.7813                 | 0.6862               | 0.2954          | 0.514                    |
| 8        | {Interpretable Embedding Procedure Knowledge Trans... | 2021  | 0.6116                 | 0.5216               | 0.3457          | 0.4654                   |
| 8        | {PYKT: A Python Library to Benchmark Deep Learning... | 2022  | 0.6744                 | 0.6508               | 0.3492          | 0.4955                   |
| 8        | {Learning Process-consistent Knowledge Tracing}...    | 2021  | 0.7823                 | 0.6846               | 0.2822          | 0.5073                   |
| 8        | {Structure-based knowledge tracing: An influence p... | 2020  | 0.6384                 | 0.5063               | 0.3582          | 0.4843                   |
| 8        | {Item Response Ranking for Cognitive Diagnosis}...    | 2021  | 0.5397                 | 0.4239               | 0.4004          | 0.4631                   |
| 8        | {Boosting Neural Cognitive Diagnosis with Student'... | 2024  | 0.6918                 | 0.6406               | 0.2659          | 0.4575                   |
| 8        | {Explainable exercise recommendation with knowledg... | 2025  | 0.7014                 | 0.5701               | 0.3129          | 0.4877                   |
| 8        | {Augmenting assessment with AI coding of online st... | 2024  | 0.6414                 | 0.4497               | 0.3393          | 0.4753                   |
| 8        | {A Unified Interpretable Intelligent Learning Diag... | 2023  | 0.7342                 | 0.6928               | 0.3059          | 0.4987                   |
| 9        | {DeepHawkes: Bridging the gap between prediction a... | 2017  | 0.7335                 | 0.5751               | 0.283           | 0.4857                   |
| 9        | {dEFEND: A system for explainable fake news detect... | 2019  | 0.7617                 | 0.5567               | 0.2687          | 0.4906                   |
| 9        | {The Future is Different: Predicting Reddits Popul... | 2024  | 0.7096                 | 0.5273               | 0.2961          | 0.4822                   |
| 9        | {Weaving a Semantic Web of Credibility Reviews for... | 2021  | 0.7012                 | 0.5402               | 0.3311          | 0.4977                   |
| 9        | {Simplistic Collection and Labeling Practices Limi... | 2023  | 0.6372                 | 0.4895               | 0.3555          | 0.4823                   |
| 9        | {Why Should This Article Be Deleted? Transparent S... | 2023  | 0.6694                 | 0.4517               | 0.3907          | 0.5161                   |
| 9        | {The power of summarization in graph mining and le... | 2021  | 0.566                  | 0.6178               | 0.3822          | 0.4649                   |
| 9        | {FactCatch: Incremental Pay-as-You-Go Fact Checkin... | 2020  | 0.5321                 | 0.4312               | 0.3193          | 0.4151                   |
| 9        | {TOT: Topology-Aware Optimal Transport for Multimo... | 2023  | 0.6167                 | 0.448                | 0.3863          | 0.49                     |
| 9        | {PROMO for Interpretable Personalized Social Emoti... | 2021  | 0.6928                 | 0.5655               | 0.4265          | 0.5463                   |
| 9        | {A quantitative argumentation-based Automated eXpl... | 2022  | 0.7534                 | 0.5753               | 0.2687          | 0.4868                   |
| 9        | {Cognitive Network Science for Understanding Onlin... | 2022  | 0.5971                 | 0.416                | 0.4672          | 0.5256                   |
| 9        | {Irony Detection, Reasoning and Understanding in Z... | 2025  | 0.7202                 | 0.5214               | 0.2657          | 0.4702                   |
| 10       | {Fault Detection in Wastewater Treatment Plants: A... | 2023  | 0.5648                 | 0.4353               | 0.4801          | 0.5182                   |
| 10       | {Knowledge-enhanced spatial–temporal multi-frequen... | 2025  | 0.6883                 | 0.536                | 0.3889          | 0.5236                   |
| 10       | {Interpretable physics-informed domain adaptation ... | 2024  | 0.6722                 | 0.6119               | 0.3421          | 0.4906                   |
| 10       | {Explainable AI for domain experts: a post Hoc ana... | 2022  | 0.6913                 | 0.612                | 0.3053          | 0.479                    |
| 10       | {Multi-fidelity sub-label-guided transfer network ... | 2025  | 0.7753                 | 0.7364               | 0.2636          | 0.4939                   |
| 10       | {Priori-distribution-guided adaptive sparse attent... | 2024  | 0.7139                 | 0.6283               | 0.3644          | 0.5217                   |
| 10       | {One-shot learning for acoustic diagnosis of indus... | 2021  | 0.7047                 | 0.6061               | 0.3939          | 0.5338                   |
| 10       | {Deep understanding in industrial processes by com... | 2019  | 0.707                  | 0.6154               | 0.3022          | 0.4844                   |
| 10       | {MPGE and RootRank: A sufficient root cause charac... | 2023  | 0.5514                 | 0.4176               | 0.3547          | 0.4432                   |
| 10       | {A Copula network deconvolution-based direct corre... | 2024  | 0.5656                 | 0.4178               | 0.4563          | 0.5055                   |
| 10       | {Knowledge extraction and insertion to deep belief... | 2020  | 0.753                  | 0.6822               | 0.3177          | 0.5136                   |
| 10       | {Learning the manufacturing capabilities of machin... | 2024  | 0.6417                 | 0.5506               | 0.2889          | 0.4477                   |
| 10       | {AITL-Net: An adaptive interpretable transfer lear... | 2025  | 0.6272                 | 0.5334               | 0.356           | 0.4781                   |
| 11       | {Mitigating the Effect of Incidental Correlations ... | 2023  | 0.59                   | 0.5428               | 0.3956          | 0.4831                   |
| 11       | {B-cos Networks: Alignment is All We Need for Inte... | 2022  | 0.6988                 | 0.6068               | 0.3582          | 0.5115                   |
| 11       | {Concept Distillation: Leveraging Human-Centered E... | 2023  | 0.7529                 | 0.6551               | 0.2946          | 0.5008                   |
| 11       | {Visual interpretability analysis of Deep CNNs usi... | 2021  | 0.7428                 | 0.6188               | 0.2715          | 0.4836                   |
| 11       | {HiLite: Hierarchical Level-implemented Architectu... | 2024  | 0.7027                 | 0.6044               | 0.369           | 0.5192                   |
| 11       | {Neural Prototype Trees for Interpretable Fine-gra... | 2021  | 0.7281                 | 0.7333               | 0.2667          | 0.4743                   |
| 11       | {NeSyFOLD: A Framework for Interpretable Image Cla... | 2024  | 0.7782                 | 0.6581               | 0.2964          | 0.5132                   |
| 11       | {X-Vision: Explainable Image Retrieval by Re-Ranki... | 2022  | 0.6879                 | 0.5473               | 0.3153          | 0.483                    |
| 11       | {ProtoPShare: Prototypical Parts Sharing for Simil... | 2021  | 0.6456                 | 0.7253               | 0.2747          | 0.4416                   |
| 11       | {Lipschitz Continuity Guided Knowledge Distillatio... | 2021  | 0.6679                 | 0.5341               | 0.2946          | 0.4626                   |
| 11       | {Multi-label Image Recognition by Recurrently Disc... | 2017  | 0.6468                 | 0.5618               | 0.338           | 0.4769                   |
| 11       | {Geo-SIC: Learning Deformable Geometric Shapes in ... | 2022  | 0.7253                 | 0.6153               | 0.3441          | 0.5156                   |
| 11       | {VISUAL RECOGNITION WITH DEEP NEAREST CENTROIDS}...   | 2023  | 0.7621                 | 0.6592               | 0.3044          | 0.5104                   |
| 11       | {Towards Trustable Skin Cancer Diagnosis via Rewri... | 2023  | 0.7383                 | 0.6059               | 0.3316          | 0.5146                   |
| 11       | {Interpreting CNNs via Decision Trees}...             | 2019  | 0.7194                 | 0.5713               | 0.288           | 0.4821                   |
| 11       | {IDEAL: Interpretable-by-Design ALgorithms for lea... | 2025  | 0.74                   | 0.6744               | 0.3149          | 0.5062                   |
| 11       | {Positional normalization-based mixed-image data a... | 2024  | 0.7045                 | 0.6007               | 0.2965          | 0.4801                   |
| 11       | {MGRW-Transformer: Multigranularity Random Walk Tr... | 2025  | 0.7773                 | 0.6388               | 0.2855          | 0.5068                   |
| 11       | {Toward extracting and exploiting generalizable kn... | 2023  | 0.7685                 | 0.6904               | 0.303           | 0.5125                   |
| 11       | {SIGN: Statistical Inference Graphs Based on Proba... | 2023  | 0.6494                 | 0.5742               | 0.4112          | 0.5184                   |
| 11       | {What is a Tabby? Interpretable Model Decisions by... | 2021  | 0.729                  | 0.6128               | 0.3672          | 0.53                     |
| 11       | {An expert knowledge-empowered CNN approach for we... | 2023  | 0.6884                 | 0.544                | 0.3763          | 0.5168                   |
| 11       | {Medical Image Classifications Using Convolutional... | 2024  | 0.7362                 | 0.6431               | 0.2751          | 0.4826                   |
| 11       | {Detecting Local Insights from Global Labels: Supe... | 2021  | 0.7155                 | 0.6878               | 0.3122          | 0.4937                   |
| 11       | {A Diverse Knowledge Perception and Fusion network... | 2025  | 0.7048                 | 0.5844               | 0.3009          | 0.4827                   |
| 11       | {GO-MAE: Self-supervised pre-training via masked a... | 2025  | 0.4564                 | 0.3403               | 0.5048          | 0.483                    |
| 11       | {A novel explainable neural network for Alzheimer'... | 2022  | 0.7439                 | 0.5643               | 0.2993          | 0.4993                   |
| 12       | {Trading Complexity for Sparsity in Random Forest ... | 2022  | 0.7455                 | 0.6114               | 0.2934          | 0.4968                   |
| 12       | {Explaining a Random Survival Forest by Extracting... | 2021  | 0.7699                 | 0.6274               | 0.2658          | 0.4926                   |
| 12       | {Connecting Interpretability and Robustness in Dec... | 2021  | 0.6976                 | 0.6011               | 0.2778          | 0.4667                   |
| 12       | {A cautionary tale on fitting decision trees to da... | 2022  | 0.7131                 | 0.5692               | 0.3456          | 0.511                    |
| 12       | {Linear TreeShap}...                                  | 2022  | 0.749                  | 0.6609               | 0.2805          | 0.4913                   |
| 12       | {RuDi: Explaining Behavior Sequence Models by Auto... | 2022  | 0.6458                 | 0.3971               | 0.371           | 0.4946                   |
| 12       | {Semi-Supervised Learning with Decision Trees: Gra... | 2022  | 0.6707                 | 0.5682               | 0.3438          | 0.4909                   |
| 12       | {Graph embedded rules for explainable predictions ... | 2020  | 0.7369                 | 0.6321               | 0.2916          | 0.492                    |
| 12       | {The voice of optimization}...                        | 2021  | 0.6674                 | 0.5034               | 0.3465          | 0.4909                   |
| 12       | {Toward trustworthy and sustainable clinical decis... | 2025  | 0.7526                 | 0.661                | 0.3114          | 0.51                     |
| 12       | {A user-guided Bayesian framework for ensemble fea... | 2022  | 0.6025                 | 0.5067               | 0.4063          | 0.4946                   |
| 12       | {A Survey of Neural Trees: Co-Evolving Neural Netw... | 2024  | 0.7886                 | 0.7005               | 0.2995          | 0.5196                   |
| 12       | {Explaining neural networks without access to trai... | 2024  | 0.7773                 | 0.6321               | 0.3087          | 0.5196                   |
| 12       | {Interpretable surrogate models to approximate the... | 2023  | 0.5596                 | 0.5177               | 0.4138          | 0.4794                   |
| 12       | {Manifoldron: Direct Space Partition via Manifold ... | 2024  | 0.6118                 | 0.4942               | 0.4466          | 0.5209                   |
| 12       | {One-Stage Tree: end-to-end tree builder and prune... | 2022  | 0.7081                 | 0.6018               | 0.3632          | 0.5184                   |
| 13       | {Entropy-Based Logic Explanations of Neural Networ... | 2022  | 0.7354                 | 0.6103               | 0.3027          | 0.4974                   |
| 13       | {Reasoning-Based Learning of Interpretable ML Mode... | 2021  | 0.7426                 | 0.6066               | 0.2648          | 0.4798                   |
| 13       | {Skin Cancer Detection: A Hybrid Approach Combinin... | 2025  | 0.5653                 | 0.5336               | 0.2987          | 0.4187                   |
| 13       | {A Classification of Anomaly Explanation Methods}...  | 2021  | 0.7556                 | 0.6842               | 0.2626          | 0.4844                   |
| 13       | {Local Explanations via Necessity and Sufficiency:... | 2021  | 0.7079                 | 0.6544               | 0.2855          | 0.4756                   |
| 13       | {BayLIME: Bayesian Local Interpretable Model-Agnos... | 2021  | 0.7695                 | 0.7059               | 0.2682          | 0.4938                   |
| 13       | {Editorial: From Explainable Artificial Intelligen... | 2024  | 0.6709                 | 0.6582               | 0.3302          | 0.4835                   |
| 13       | {An end-to-end explainability framework for spatio... | 2025  | 0.727                  | 0.7248               | 0.2752          | 0.4785                   |
| 13       | {The performance-interpretability trade-off: a com... | 2025  | 0.7355                 | 0.6118               | 0.2672          | 0.4779                   |
| 13       | {AI's 10 to Watch, 2022}...                           | 2023  | 0.6071                 | 0.478                | 0.4082          | 0.4977                   |
| 13       | {Exploring explainable AI: category theory insight... | 2023  | 0.7141                 | 0.6424               | 0.2723          | 0.4711                   |
| 13       | {A typology for exploring the mitigation of shortc... | 2023  | 0.626                  | 0.5278               | 0.4088          | 0.5065                   |
| 13       | {State-of-the-art review and synthesis: A requirem... | 2024  | 0.407                  | 0.2947               | 0.4711          | 0.4423                   |
| 13       | {AI, Explainability and Public Reason: The Argumen... | 2021  | 0.6633                 | 0.5848               | 0.3218          | 0.4755                   |
| 13       | {Decoding Mental States in Social Cognition: Insig... | 2025  | 0.5676                 | 0.5197               | 0.4482          | 0.5019                   |
| 13       | {Post-hoc explanation of black-box classifiers usi... | 2021  | 0.7297                 | 0.6996               | 0.3004          | 0.4936                   |
| 13       | {Explainable AI for the Choquet Integral}...          | 2021  | 0.8194                 | 0.7278               | 0.2722          | 0.5184                   |
| 13       | {Explainable Artificial Intelligence for Developin... | 2020  | 0.7091                 | 0.6046               | 0.3558          | 0.5148                   |
| 13       | {A Survey on Neural Network Interpretability}...      | 2021  | 0.6842                 | 0.6274               | 0.2856          | 0.465                    |
| 14       | {HYPER: Learned Hybrid Trajectory Prediction via F... | 2022  | 0.7299                 | 0.609                | 0.3124          | 0.5003                   |
| 14       | {Receding Horizon Planning with Rule Hierarchies f... | 2023  | 0.6597                 | 0.5922               | 0.3344          | 0.4808                   |
| 14       | {Incorporating Driving Knowledge in Deep Learning ... | 2023  | 0.7728                 | 0.6755               | 0.2681          | 0.4952                   |
| 14       | {Sparse Prototype Network for Explainable Pedestri... | 2025  | 0.7514                 | 0.5805               | 0.2959          | 0.5009                   |
| 14       | {Prediction of the driver's focus of attention bas... | 2022  | 0.7095                 | 0.5577               | 0.2761          | 0.4711                   |
| 14       | {Self-Explaining Abilities of an Intelligent Agent... | 2022  | 0.6389                 | 0.4801               | 0.3213          | 0.4642                   |
| 14       | {SMEMO: Social Memory for Trajectory Forecasting}...  | 2024  | 0.6605                 | 0.5038               | 0.3518          | 0.4907                   |
| 14       | {An Efficient Deep Reinforcement Learning-Based Ca... | 2024  | 0.7063                 | 0.623                | 0.3318          | 0.5003                   |
| 14       | {Drive in Corridors: Enhancing the Safety of End-t... | 2025  | 0.8348                 | 1.0                  | 0.2648          | 0.5213                   |
| 15       | {Game-theoretic Counterfactual Explanation for Gra... | 2024  | 0.761                  | 0.6583               | 0.276           | 0.4943                   |
| 15       | {Reversible and irreversible bracket-based dynamic... | 2023  | 0.6068                 | 0.4764               | 0.4049          | 0.4957                   |
| 15       | {Neural Message Passing for Multi-label Classifica... | 2020  | 0.5947                 | 0.5324               | 0.4063          | 0.4911                   |
| 15       | {GRETEL 2.0: Generation and Evaluation of Graph Co... | 2024  | 0.7487                 | 0.666                | 0.2983          | 0.501                    |
| 15       | {Defining and Quantifying the Emergence of Sparse ... | 2023  | 0.6636                 | 0.554                | 0.3671          | 0.5005                   |
| 15       | {Exploring Faithful Rationale for Multi-Hop Fact V... | 2023  | 0.7603                 | 0.6609               | 0.3148          | 0.5153                   |
| 15       | {Explainable Spatio-Temporal Graph Neural Networks... | 2023  | 0.7459                 | 0.6686               | 0.3012          | 0.5013                   |
| 15       | {FIVES: Feature Interaction Via Edge Search for La... | 2021  | 0.6321                 | 0.5736               | 0.37            | 0.488                    |
| 15       | {MENTORGNN: Deriving Curriculum for Pre-Training G... | 2022  | 0.8056                 | 0.683                | 0.263           | 0.5072                   |
| 15       | {GFS-Node: Graph Fuzzy Systems for Node Prediction... | 2024  | 0.6449                 | 0.5299               | 0.3392          | 0.4767                   |
| 15       | {CI-GNN: A Granger causality-inspired graph neural... | 2024  | 0.7296                 | 0.6369               | 0.316           | 0.5021                   |
| 15       | {Uncertainty-aware network alignment}...              | 2021  | 0.6007                 | 0.4704               | 0.4271          | 0.5052                   |
| 16       | {MutaPLM: Protein Language Modeling for Mutation E... | 2024  | 0.6872                 | 0.6131               | 0.2774          | 0.4618                   |
| 16       | {Adverse drug reaction prediction with symbolic la... | 2017  | 0.6972                 | 0.6499               | 0.2976          | 0.4774                   |
| 16       | {scACT: Accurate Cross-modality Translation via Cy... | 2024  | 0.6318                 | 0.5059               | 0.3983          | 0.5033                   |
| 16       | {Ad hoc learning of peptide fragmentation from mas... | 2022  | 0.6342                 | 0.5132               | 0.443           | 0.529                    |
| 16       | {A generalized-template-based graph neural network... | 2022  | 0.6215                 | 0.4798               | 0.3421          | 0.4679                   |
| 16       | {Informative top-k class associative rule for canc... | 2020  | 0.5802                 | 0.4262               | 0.4086          | 0.4858                   |
| 16       | {MSN-DTA: A multi-scale node adaptive graph neural... | 2025  | 0.772                  | 0.6512               | 0.2763          | 0.4993                   |
| 16       | {RPI-GGCN: Prediction of RNA-Protein Interaction B... | 2025  | 0.7006                 | 0.4973               | 0.2991          | 0.4798                   |
| 16       | {A feature pair-based neural network embedded deci... | 2025  | 0.7536                 | 0.6751               | 0.3088          | 0.509                    |
| 17       | {Reinforcement Learning for Reduced-order Models o... | 2024  | 0.6844                 | 0.5493               | 0.2896          | 0.4673                   |
| 17       | {DiffVL: Scaling Up Soft Body Manipulation using V... | 2023  | 0.7153                 | 0.6157               | 0.3721          | 0.5265                   |
| 17       | {RAGG: Retrieval-Augmented Grasp Generation Model}... | 2025  | 0.6647                 | 0.4895               | 0.4109          | 0.5251                   |
| 17       | {Learning Policies by Learning Rules}...              | 2022  | 0.7787                 | 0.6862               | 0.2699          | 0.4988                   |
| 17       | {Explaining in Time: Meeting Interactive Standards... | 2021  | 0.6443                 | 0.6009               | 0.321           | 0.4665                   |
| 17       | {An Object-Driven Navigation Strategy Based on Act... | 2024  | 0.6628                 | 0.5316               | 0.3227          | 0.4758                   |
| 17       | {Double-Feedback: Enhancing Large Language Models ... | 2025  | 0.7294                 | 0.6516               | 0.3387          | 0.5145                   |
| 17       | {GAN-Based Editable Movement Primitive From High-V... | 2023  | 0.729                  | 0.6079               | 0.2998          | 0.4929                   |
| 17       | {Practical Preassigned Fixed-Time Fuzzy Control fo... | 2024  | 0.4129                 | 0.219                | 0.4401          | 0.4279                   |
| 18       | {Exact and consistent interpretation for piecewise... | 2018  | 0.7228                 | 0.6579               | 0.3421          | 0.5134                   |
| 18       | {FSP-LAPLACE: Function-Space Priors for the Laplac... | 2024  | 0.6563                 | 0.5623               | 0.4186          | 0.5256                   |
| 18       | {Finding High-Value Training Data Subset Through D... | 2021  | 0.6829                 | 0.4841               | 0.3866          | 0.5199                   |
| 18       | {GRAND-SLAMIN' Interpretable Additive Modeling wit... | 2023  | 0.6242                 | 0.4988               | 0.4409          | 0.5234                   |
| 18       | {FEATURE COLLAPSE}...                                 | 2024  | 0.6362                 | 0.4808               | 0.4267          | 0.521                    |
| 18       | {BORT: TOWARDS EXPLAINABLE NEURAL NETWORKS WITH BO... | 2023  | 0.7894                 | 1.0                  | 0.2847          | 0.5118                   |
| 18       | {Learning Debiased Representations via Conditional... | 2023  | 0.6972                 | 0.5384               | 0.3035          | 0.4806                   |
| 18       | {Interpretable deep model pruning}...                 | 2025  | 0.713                  | 0.564                | 0.3172          | 0.4953                   |
| 18       | {Network-based dimensionality reduction of high-di... | 2022  | 0.3317                 | 0.096                | 0.499           | 0.4237                   |
| 18       | {Evolving stochastic configure network: A more com... | 2023  | 0.6947                 | 0.5597               | 0.3666          | 0.5142                   |
| 18       | {HOPE: High-Order Polynomial Expansion of Black-Bo... | 2024  | 0.6945                 | 0.5418               | 0.2787          | 0.4658                   |
| 19       | {Identifying ATT&CK Tactics in Android Malware Con... | 2022  | 0.6127                 | 0.3942               | 0.3936          | 0.4922                   |
| 19       | {Evaluation of Large Language Models on Code Obfus... | 2024  | 0.4769                 | 0.2769               | 0.4018          | 0.4356                   |
| 19       | {Adversarial Algorithm Unrolling Network for Inter... | 2024  | 0.6578                 | 0.5045               | 0.3426          | 0.4844                   |
| 19       | {A soft prototype-based autonomous fuzzy inference... | 2024  | 0.7123                 | 0.5906               | 0.2931          | 0.4817                   |
| 19       | {LRP2A: Layer-wise Relevance Propagation based Adv... | 2022  | 0.6749                 | 0.5365               | 0.2807          | 0.4581                   |
| 19       | {Distributed and explainable GHSOM for anomaly det... | 2024  | 0.626                  | 0.5253               | 0.3323          | 0.4645                   |
| 19       | {Adaptive deep learning for network intrusion dete... | 2022  | 0.8199                 | 1.0                  | 0.2789          | 0.5224                   |
| 20       | {Unfooling Perturbation-Based Post Hoc Explainers}... | 2023  | 0.703                  | 0.6547               | 0.3414          | 0.5041                   |
| 20       | {Less is More: Attention Supervision with Counterf... | 2020  | 0.5856                 | 0.4311               | 0.3751          | 0.4698                   |
| 20       | {Towards Robust Contrastive Explanations for Human... | 2023  | 0.6624                 | 0.6351               | 0.3649          | 0.4988                   |
| 20       | {CLEAR: Generative Counterfactual Explanations on ... | 2022  | 0.7446                 | 0.5781               | 0.2614          | 0.4788                   |
| 20       | {Generating Perturbation-based Explanations with R... | 2022  | 0.6841                 | 0.5656               | 0.3653          | 0.5087                   |
| 20       | {Interpretable Counterfactual Explanations Guided ... | 2021  | 0.7816                 | 0.5901               | 0.277           | 0.504                    |
| 20       | {Stop ordering machine learning algorithms by thei... | 2023  | 0.7581                 | 0.6132               | 0.3014          | 0.5069                   |
| 20       | {Towards consistency of rule-based explainer and b... | 2025  | 0.6883                 | 0.636                | 0.3557          | 0.5054                   |
| 20       | {GradCFA: A Hybrid Gradient-Based Counterfactual a... | 2025  | 0.7997                 | 0.6735               | 0.2962          | 0.5228                   |
| 20       | {Navigating explanatory multiverse through counter... | 2025  | 0.6085                 | 0.4951               | 0.4335          | 0.5123                   |
| 21       | {UMFuse: Unified Multi View Fusion for Human Editi... | 2023  | 0.7199                 | 0.6543               | 0.3327          | 0.5069                   |
| 21       | {NAISR: A 3D NEURAL ADDITIVE MODEL FOR INTERPRETAB... | 2024  | 0.719                  | 0.576                | 0.2953          | 0.486                    |
| 21       | {MoST: Multi-modality Scene Tokenization for Motio... | 2024  | 0.7508                 | 0.5593               | 0.3292          | 0.5189                   |
| 21       | {Rethinking one-shot face reenactment: A spatial-t... | 2023  | 0.6592                 | 0.5622               | 0.3344          | 0.4806                   |
| 21       | {Spatial reasoning for few-shot object detection}...  | 2021  | 0.6968                 | 0.5035               | 0.3233          | 0.4913                   |
| 21       | {Stereo Hand-Object Reconstruction for Human-to-Ro... | 2025  | 0.7065                 | 0.6757               | 0.2661          | 0.4643                   |
| 21       | {Interpretable Rotation-Equivariant Quaternion Neu... | 2024  | 0.5352                 | 0.391                | 0.4644          | 0.4963                   |
| 21       | {Improving Self-Supervised Learning of Transparent... | 2024  | 0.772                  | 0.6662               | 0.2952          | 0.5098                   |
| 21       | {DiSCO: Differentiable Scan Context with Orientati... | 2021  | 0.5832                 | 0.4916               | 0.4074          | 0.4865                   |
| 22       | {A Two-Step Approach for the Prediction of Mood Le... | 2019  | 0.6116                 | 0.4687               | 0.4284          | 0.5108                   |
| 22       | {PD-Net: Quantitative Motor Function Evaluation fo... | 2021  | 0.5582                 | 0.3801               | 0.3388          | 0.4376                   |
| 22       | {REPRESENTATION LEARNING FOR IMPROVED INTERPRETABI... | 2021  | 0.7898                 | 0.575                | 0.3461          | 0.5458                   |
| 22       | {Toward Explainable Affective Computing: A Review}... | 2024  | 0.6582                 | 0.7251               | 0.2749          | 0.4474                   |
| 22       | {Machine learning and clinical EEG data for multip... | 2025  | 0.6912                 | 0.4972               | 0.3582          | 0.5081                   |
| 22       | {Clinical-inspired skin lesions recognition based ... | 2025  | 0.5631                 | 0.498                | 0.4272          | 0.4884                   |
| 22       | {Multimodal heterogeneous graph fusion for automat... | 2025  | 0.6519                 | 0.5248               | 0.3993          | 0.513                    |
| 22       | {Adaptive neural decision tree for EEG based emoti... | 2023  | 0.5958                 | 0.47                 | 0.3691          | 0.4711                   |
| 23       | {Single Image Defocus Deblurring via Implicit Neur... | 2023  | 0.7352                 | 0.5558               | 0.2804          | 0.485                    |
| 23       | {Pluralistic Image Completion with Gaussian Mixtur... | 2022  | 0.5918                 | 0.5386               | 0.4614          | 0.52                     |
| 23       | {DeepSN-Net: Deep Semi-Smooth Newton Driven Networ... | 2025  | 0.7297                 | 0.5848               | 0.2827          | 0.4838                   |
| 23       | {Active Sensing for Two-Sided Beam Alignment and R... | 2023  | 0.3026                 | 0.2017               | 0.6937          | 0.5177                   |
| 23       | {Learning Disentangled Priors for Hyperspectral An... | 2025  | 0.7148                 | 0.55                 | 0.3372          | 0.5071                   |
| 23       | {Extracting and inserting knowledge into stacked d... | 2021  | 0.664                  | 0.6263               | 0.3737          | 0.5043                   |
| 23       | {Self-supervised network compression based on quan... | 2025  | 0.7201                 | 0.5462               | 0.3588          | 0.5214                   |
| 24       | {LINe: Out-of-Distribution Detection by Leveraging... | 2023  | 0.6429                 | 0.4818               | 0.4331          | 0.5275                   |
| 24       | {Logic Rule Guided Attribution with Dynamic Ablati... | 2022  | 0.7561                 | 0.643                | 0.2961          | 0.5031                   |
| 24       | {GAM: Explainable Visual Similarity and Classifica... | 2021  | 0.7446                 | 0.6239               | 0.3234          | 0.5129                   |
| 24       | {Step-Wise Explanations of Constraint Satisfaction... | 2020  | 0.5528                 | 0.4956               | 0.4134          | 0.4761                   |
| 24       | {Red Teaming Deep Neural Networks with Feature Syn... | 2023  | 0.7435                 | 0.648                | 0.3178          | 0.5094                   |
| 24       | {Relation of Activity and Confidence When Training... | 2025  | 0.5489                 | 0.4613               | 0.4273          | 0.482                    |
| 24       | {A Study of Explainability Features to Scrutinize ... | 2021  | 0.6258                 | 0.5108               | 0.3813          | 0.4913                   |
| 24       | {Do Input Gradients Highlight Discriminative Featu... | 2021  | 0.7741                 | 0.6497               | 0.3109          | 0.5194                   |
| 24       | {Right for the Right Concept: Revising Neuro-Symbo... | 2021  | 0.7371                 | 0.6414               | 0.3371          | 0.5171                   |
| 24       | {A Model-Agnostic Approach for Explaining the Pred... | 2019  | 0.7534                 | 0.7283               | 0.2717          | 0.4884                   |
| 24       | {Improving performance of deep learning models wit... | 2021  | 0.8282                 | 0.7351               | 0.2608          | 0.5161                   |
| 24       | {Single-input interpretation of a deep classificat... | 2025  | 0.6831                 | 0.5704               | 0.3353          | 0.4918                   |
| 25       | {Imbalanced Time Series Classification for Flight ... | 2020  | 0.61                   | 0.4536               | 0.4619          | 0.5286                   |
| 25       | {Bridging Self-Attention and Time Series Decomposi... | 2022  | 0.7605                 | 0.6328               | 0.2998          | 0.5072                   |
| 25       | {Online Deep Hybrid Ensemble Learning for Time Ser... | 2023  | 0.7153                 | 0.5982               | 0.3394          | 0.5086                   |
| 25       | {FAMC-Net: Frequency Domain Parity Correction Atte... | 2023  | 0.7138                 | 0.5917               | 0.2998          | 0.4861                   |
| 25       | {Neuro-symbolic Models for Interpretable Time Seri... | 2022  | 0.7389                 | 0.5353               | 0.3286          | 0.5132                   |
| 25       | {A data-aware explainable deep learning approach f... | 2023  | 0.6005                 | 0.418                | 0.4155          | 0.4987                   |
| 25       | {Towards interpretability in fingerprint based ind... | 2023  | 0.5326                 | 0.4003               | 0.459           | 0.4921                   |
| 25       | {Sparse Graphical Linear Dynamical Systems}...        | 2024  | 0.6494                 | 0.5369               | 0.3414          | 0.48                     |
| 25       | {A graph structure feature-based framework for the... | 2023  | 0.53                   | 0.3752               | 0.4656          | 0.4946                   |
| 26       | {A Reasoning Approach to Financial Data Exchange w... | 2021  | 0.4742                 | 0.4298               | 0.3888          | 0.4272                   |
| 26       | {Financial Risk Management using Machine Learning ... | 2021  | 0.7966                 | 0.6535               | 0.2728          | 0.5085                   |
| 26       | {Explainable Graph-based Fraud Detection via Neura... | 2022  | 0.6323                 | 0.5039               | 0.3139          | 0.4572                   |
| 26       | {Multi-Performance Estimation for Deploying Bank B... | 2023  | 0.6349                 | 0.4037               | 0.4367          | 0.5259                   |
| 26       | {Predicting and interpreting financial distress us... | 2022  | 0.7344                 | 0.6271               | 0.3085          | 0.5001                   |
| 26       | {Credit risk assessment of small and medium-sized ... | 2022  | 0.6138                 | 0.4802               | 0.4107          | 0.5021                   |
| 27       | {Explanations for Negative Query Answers under Exi... | 2020  | 0.5919                 | 0.4105               | 0.4547          | 0.5164                   |
| 27       | {Explainable Conversational Question Answering ove... | 2023  | 0.7829                 | 0.6358               | 0.3064          | 0.5208                   |
| 27       | {Self-assembling modular networks for interpretabl... | 2019  | 0.7483                 | 0.6416               | 0.273           | 0.4869                   |
| 27       | {Causal Probing for Dual Encoders}...                 | 2024  | 0.6744                 | 0.5411               | 0.3203          | 0.4797                   |
| 27       | {Unanswerable Question Correction and Explanation ... | 2022  | 0.7623                 | 0.6351               | 0.3323          | 0.5258                   |
| 27       | {Social-IQ: A Question Answering Benchmark for Art... | 2019  | 0.6072                 | 0.523                | 0.4421          | 0.5164                   |
| 27       | {AFS Graph: Multidimensional Axiomatic Fuzzy Set K... | 2023  | 0.7359                 | 0.5836               | 0.3509          | 0.5241                   |
| 28       | {ECR-Chain: Advancing Generative Language Models t... | 2024  | 0.6659                 | 0.5274               | 0.3941          | 0.5164                   |
| 28       | {A document-grounded matching network for response... | 2019  | 0.6475                 | 0.4938               | 0.391           | 0.5064                   |
| 28       | {An LLM Feature-based Framework for Dialogue Const... | 2024  | 0.7496                 | 0.6405               | 0.341           | 0.5249                   |
| 28       | {Advanced dialog state tracking with noetic graphs... | 2025  | 0.7867                 | 0.6735               | 0.3198          | 0.5299                   |
| 29       | {Why We Go Where We Go: Profiling User Decisions o... | 2020  | 0.6896                 | 0.6647               | 0.3353          | 0.4947                   |
| 29       | {Generate Neural Template Explanations for Recomme... | 2020  | 0.6987                 | 0.5491               | 0.3028          | 0.481                    |
| 29       | {Explainable Cross-Domain Recommendations through ... | 2018  | 0.753                  | 0.6351               | 0.3442          | 0.5282                   |
| 29       | {Modeling Dynamic Missingness of Implicit Feedback... | 2018  | 0.7289                 | 0.7337               | 0.2663          | 0.4745                   |
| 29       | {Tower Bridge Net (TB-Net): Bidirectional Knowledg... | 2022  | 0.7776                 | 0.6636               | 0.2625          | 0.4943                   |
| 29       | {2nd International Workshop on Industrial Recommen... | 2021  | 0.6404                 | 0.4859               | 0.4355          | 0.5277                   |
| 29       | {Metapath-guided multi-headed attention networks f... | 2023  | 0.6515                 | 0.5163               | 0.3389          | 0.4796                   |
| 30       | {Relational Neural Markov Random Fields}...           | 2022  | 0.6557                 | 0.6275               | 0.3444          | 0.4845                   |
| 30       | {Scalable Coupling of Deep Learning with Logical R... | 2023  | 0.7643                 | 0.6911               | 0.3089          | 0.5138                   |
| 30       | {Lifting Symmetry Breaking Constraints with Induct... | 2021  | 0.6441                 | 0.5168               | 0.3563          | 0.4858                   |
| 30       | {Explanatory machine learning for sequential human... | 2023  | 0.6121                 | 0.468                | 0.3809          | 0.4849                   |
| 30       | {FFNSL: Feed-Forward Neural-Symbolic Learner}...      | 2023  | 0.7991                 | 0.6809               | 0.2757          | 0.5113                   |
| 30       | {A Primer for Neural Arithmetic Logic Modules}...     | 2022  | 0.6779                 | 0.5346               | 0.3857          | 0.5172                   |
| 30       | {Neural transition system abstraction for neural n... | 2025  | 0.6659                 | 0.526                | 0.4071          | 0.5236                   |
| 31       | {Learning to simulate dynamic environments with ga... | 2020  | 0.6875                 | 0.5678               | 0.3249          | 0.488                    |
| 31       | {Self-Discovering Interpretable Diffusion Latent D... | 2024  | 0.5824                 | 0.4445               | 0.4462          | 0.5075                   |
| 31       | {Interpreting CNN predictions using conditional Ge... | 2024  | 0.7688                 | 0.6505               | 0.3203          | 0.5221                   |
| 31       | {Attribute-based regularization of latent spaces f... | 2021  | 0.6908                 | 0.5258               | 0.3628          | 0.5104                   |
| 31       | {Spectral bounding: Strictly satisfying the 1-Lips... | 2020  | 0.7203                 | 0.6554               | 0.3446          | 0.5137                   |
| 32       | {Model Distillation for Revenue Optimization: Inte... | 2021  | 0.6946                 | 0.5439               | 0.3127          | 0.4846                   |
| 32       | {An Interactive Neural Network Approach to Keyphra... | 2021  | 0.5364                 | 0.4216               | 0.4754          | 0.5029                   |
| 32       | {Thresholded ConvNet ensembles: neural networks fo... | 2020  | 0.7524                 | 0.5821               | 0.3185          | 0.5137                   |
| 32       | {Research on public opinion effecting on stock pri... | 2024  | 0.5836                 | 0.5227               | 0.3987          | 0.4819                   |
| 33       | {TabVer: Tabular Fact Verification with Natural Lo... | 2024  | 0.7612                 | 0.665                | 0.2825          | 0.4979                   |
| 33       | {Model Checking Causality}...                         | 2024  | 0.5831                 | 0.4926               | 0.4234          | 0.4952                   |
| 33       | {Investigating Transformer guided Chaining for Int... | 2023  | 0.7782                 | 0.6794               | 0.2787          | 0.5035                   |
| 33       | {Decoupled Context Processing for Context Augmente... | 2022  | 0.6529                 | 0.579                | 0.3821          | 0.504                    |
| 33       | {Learn to Explain: Multimodal Reasoning via Though... | 2022  | 0.7954                 | 0.6712               | 0.2694          | 0.5061                   |
| 33       | {In-Context Impersonation Reveals Large Language M... | 2023  | 0.5459                 | 0.4213               | 0.4128          | 0.4727                   |
| 33       | {Improved Logical Reasoning of Language Models via... | 2023  | 0.8133                 | 0.6762               | 0.2854          | 0.5229                   |
| 33       | {Explainable and Efficient Editing for Large Langu... | 2025  | 0.6631                 | 0.5349               | 0.3516          | 0.4918                   |
| 34       | {Learning and Interpreting Potentials for Classica... | 2020  | 0.5944                 | 0.5011               | 0.3627          | 0.467                    |
| 34       | {PHYSICS-AS-INVERSE-GRAPHICS: UNSUPERVISED PHYSICA... | 2020  | 0.6832                 | 0.6116               | 0.3635          | 0.5073                   |
| 34       | {LEARNING THE DYNAMICS OF PHYSICAL SYSTEMS FROM SP... | 2022  | 0.7124                 | 0.6381               | 0.2933          | 0.4819                   |
| 34       | {Development of a cascaded multitask physics-infor... | 2025  | 0.5749                 | 0.3858               | 0.41            | 0.4842                   |
| 34       | {Towards trustworthy civil aviation hazards identi... | 2025  | 0.5921                 | 0.4047               | 0.3937          | 0.483                    |
| 34       | {Spatio-temporal attention-based hidden physics-in... | 2025  | 0.7056                 | 0.5693               | 0.3631          | 0.5172                   |
| 34       | {Deep learning assisted physics-based modeling of ... | 2023  | 0.8068                 | 0.7036               | 0.2911          | 0.5231                   |
| 35       | {Relational Concept Bottleneck Models}...             | 2024  | 0.7583                 | 0.7278               | 0.2722          | 0.491                    |
| 35       | {Knowledge Graph Completion with Counterfactual Au... | 2023  | 0.7865                 | 0.7108               | 0.2755          | 0.5054                   |
| 35       | {Knowledge graph embedding with the special orthog... | 2023  | 0.6247                 | 0.4757               | 0.2885          | 0.4398                   |
| 35       | {Ontology Completion with Graph-Based Machine Lear... | 2022  | 0.7227                 | 0.591                | 0.3194          | 0.5009                   |
| 35       | {Provenance-Aware Knowledge Representation: A Surv... | 2020  | 0.5753                 | 0.4105               | 0.428           | 0.4943                   |
| 35       | {Attention-based explainable friend link predictio... | 2022  | 0.6624                 | 0.5232               | 0.2897          | 0.4574                   |
| 36       | {Distributed provenance compression}...               | 2017  | 0.5561                 | 0.3708               | 0.4851          | 0.517                    |
| 36       | {Role of Zero-Knowledge Proof in Blockchain Securi... | 2022  | 0.657                  | 0.5134               | 0.3864          | 0.5082                   |
| 36       | {Do you need a blockchain in construction? Use cas... | 2020  | 0.5267                 | 0.4508               | 0.5404          | 0.5342                   |
| 36       | {Enhanced Dynamic Deep Q-Network for Federated Lea... | 2025  | 0.4749                 | 0.3059               | 0.4821          | 0.4788                   |
| 36       | {An intrinsic integrity-driven rating model for a ... | 2024  | 0.6425                 | 0.5166               | 0.3973          | 0.5076                   |
| 37       | {Variational graph embedding and clustering with L... | 2019  | 0.7591                 | 0.5676               | 0.2899          | 0.5011                   |
| 37       | {CrysGNN: Distilling Pre-trained Knowledge to Enha... | 2023  | 0.7017                 | 0.6321               | 0.3524          | 0.5096                   |
| 37       | {Multi-View Empowered Structural Graph Wordificati... | 2025  | 0.7393                 | 0.5491               | 0.3281          | 0.5132                   |
| 37       | {NED: An Inter-Graph Node Metric Based On Edit Dis... | 2017  | 0.6187                 | 0.4869               | 0.3419          | 0.4665                   |
| 37       | {EasyDGL: Encode, Train and Interpret for Continuo... | 2024  | 0.6438                 | 0.5057               | 0.3562          | 0.4856                   |
| 37       | {Preserving high-order ego-centric topological pat... | 2025  | 0.7464                 | 0.6879               | 0.3121          | 0.5075                   |
| 37       | {Lower order information preserved network embeddi... | 2021  | 0.7166                 | 0.6099               | 0.2645          | 0.468                    |
| 37       | {Exploring graph capsual network and graphormer fo... | 2023  | 0.7725                 | 0.6947               | 0.3053          | 0.5155                   |
| 38       | {Information Discrepancy in Strategic Learning}...    | 2022  | 0.2917                 | 0.1974               | 0.3878          | 0.3446                   |
| 38       | {Differentiable Pattern Set Mining}...                | 2021  | 0.6528                 | 0.5606               | 0.3947          | 0.5108                   |
| 38       | {Judgment Prediction via Injecting Legal Knowledge... | 2021  | 0.7089                 | 0.6068               | 0.3182          | 0.494                    |
| 38       | {Classification Rules in Relaxed Logical Form}...     | 2020  | 0.7478                 | 0.6779               | 0.2933          | 0.4978                   |
| 38       | {Uninorm-like parametric activation functions for ... | 2023  | 0.7482                 | 0.6233               | 0.3097          | 0.507                    |
| 38       | {PIPER: A logic-driven deep contrastive optimizati... | 2023  | 0.7142                 | 0.6075               | 0.3326          | 0.5043                   |
| 39       | {Legal Judgment Prediction with Multi-Stage Case R... | 2021  | 0.5307                 | 0.3857               | 0.5152          | 0.5222                   |
| 39       | {Crosslingual Topic Modeling with WikiPDA}...         | 2021  | 0.7598                 | 0.6646               | 0.3293          | 0.523                    |
| 39       | {Semi-supervised Semantic Visualization for Networ... | 2021  | 0.6892                 | 0.5845               | 0.3678          | 0.5124                   |
| 39       | {Locally weighted embedding topic modeling by mark... | 2018  | 0.7914                 | 0.6875               | 0.2671          | 0.503                    |
| 40       | {EDGE: Entity-Diffusion Gaussian Ensemble for Inte... | 2021  | 0.6616                 | 0.443                | 0.3898          | 0.5122                   |
| 40       | {FDTI: Fine-Grained Deep Traffic Inference with Ro... | 2023  | 0.7467                 | 0.6066               | 0.3075          | 0.5051                   |
| 40       | {Short-term trajectory prediction for individual m... | 2022  | 0.6507                 | 0.4819               | 0.4097          | 0.5181                   |
| 40       | {Interpretable Graph Reservoir Computing With the ... | 2024  | 0.5566                 | 0.471                | 0.4772          | 0.5129                   |
| 41       | {Physics-guided neural network for predicting asph... | 2024  | 0.781                  | 1.0                  | 0.3968          | 0.5697                   |
| 41       | {Explainable AI for engineering design: A unified ... | 2024  | 0.6741                 | 0.4743               | 0.4195          | 0.534                    |
| 41       | {A Review of Orebody Knowledge Enhancement Using M... | 2024  | 0.6373                 | 0.4415               | 0.4228          | 0.5193                   |
| 41       | {Dynamic prediction of sulfur dioxide concentratio... | 2025  | 0.338                  | 0.1539               | 0.4936          | 0.4236                   |
| 41       | {Machine-learning-assisted classification of const... | 2024  | 0.5896                 | 0.3954               | 0.4379          | 0.5061                   |
| 41       | {Ultra-low cycle fatigue life prediction of stainl... | 2024  | 0.7515                 | 0.5768               | 0.3192          | 0.5137                   |
| 41       | {Predicting the properties of metamaterials consis... | 2024  | 0.679                  | 0.5209               | 0.4127          | 0.5326                   |
| 42       | {Towards Multi-dimensional Explanation Alignment f... | 2024  | 0.6726                 | 0.5139               | 0.3675          | 0.5048                   |
| 42       | {CAM-Based Methods Can See Through Walls}...          | 2024  | 0.7886                 | 0.7023               | 0.2935          | 0.5163                   |
| 42       | {Deep-NFA: A deep a contrario framework for tiny o... | 2024  | 0.6203                 | 0.4645               | 0.43            | 0.5156                   |
| 42       | {Interpreting Image Classifiers by Generating Disc... | 2022  | 0.7359                 | 0.5893               | 0.2863          | 0.4886                   |
| 42       | {Frequency-aware feature aggregation network with ... | 2024  | 0.5808                 | 0.4851               | 0.4571          | 0.5127                   |
| 43       | {Explainable Person Re-Identification with Attribu... | 2021  | 0.7189                 | 0.5293               | 0.3579          | 0.5204                   |
| 43       | {Dynamic-Structured Semantic Propagation Network}...  | 2018  | 0.6093                 | 0.5346               | 0.439           | 0.5156                   |
| 43       | {Audiovisual Generalised Zero-shot Learning with C... | 2022  | 0.7116                 | 0.6488               | 0.3142          | 0.493                    |
| 43       | {A Machine Learning Paradigm for Studying Pictoria... | 2024  | 0.524                  | 0.3668               | 0.3648          | 0.4365                   |
| 44       | {Modelling Cellular Perturbations with the Sparse ... | 2023  | 0.548                  | 0.5379               | 0.4621          | 0.5008                   |
| 44       | {Brain Decodes Deep Nets}...                          | 2024  | 0.6821                 | 0.5364               | 0.4061          | 0.5303                   |
| 44       | {Bio-inspired computational model for direction an... | 2024  | 0.6716                 | 0.4793               | 0.2784          | 0.4553                   |
| 44       | {Transfer Learning of Fuzzy Spatio-Temporal Rules ... | 2023  | 0.6957                 | 0.4811               | 0.3607          | 0.5115                   |
| 44       | {Spiking neural P systems with long-term potentiat... | 2023  | 0.5761                 | 0.3747               | 0.3417          | 0.4472                   |
| 45       | {3D Shape Reconstruction of Semi-Transparent Worms... | 2023  | 0.6969                 | 0.4777               | 0.3672          | 0.5156                   |
| 45       | {RGB-Depth Fusion GAN for Indoor Depth Completion}... | 2022  | 0.7324                 | 0.6611               | 0.2884          | 0.4882                   |
| 45       | {Guest Editorial: Special Issue on Explainable Rep... | 2024  | 0.3013                 | 0.179                | 0.6865          | 0.5132                   |
| 46       | {KiL 2023 : 3rd International Workshop on Knowledg... | 2023  | 0.7177                 | 0.5294               | 0.38            | 0.5319                   |
| 46       | {Actual Causality Canvas: A General Framework for ... | 2020  | 0.5547                 | 0.4402               | 0.5232          | 0.5374                   |
| 46       | {A novel framework for artificial intelligence exp... | 2024  | 0.773                  | 0.6691               | 0.263           | 0.4925                   |
| 46       | {An explainable AI system for automated COVID-19 a... | 2021  | 0.597                  | 0.4139               | 0.3957          | 0.4863                   |
| 46       | {AI-Based Advanced Approaches and Dry Eye Disease ... | 2024  | 0.5475                 | 0.3867               | 0.5273          | 0.5364                   |
| 47       | {A Uniform Treatment of Aggregates and Constraints... | 2020  | 0.2724                 | 0.1885               | 0.7474          | 0.5337                   |
| 47       | {Evolution of Building Energy Management Systems f... | 2025  | 0.7838                 | 1.0                  | 0.3187          | 0.528                    |
| 47       | {Ensemble learning framework for detecting electri... | 2025  | 0.6808                 | 0.4291               | 0.4371          | 0.5468                   |
| 48       | {On Robust Trimming of Bayesian Network Classifier... | 2018  | 0.6716                 | 0.4832               | 0.2743          | 0.4531                   |
| 48       | {On Tractable Computation of Expected Predictions}... | 2019  | 0.7458                 | 0.5541               | 0.4062          | 0.559                    |
| 48       | {NSNet: A General Neural Probabilistic Framework f... | 2022  | 0.691                  | 0.5642               | 0.4148          | 0.5391                   |
| 48       | {Flexible Bayesian Nonlinear Model Configuration}...  | 2021  | 0.6283                 | 0.4917               | 0.4468          | 0.5285                   |
| 48       | {Syntactic reasoning with conditional probabilitie... | 2023  | 0.5001                 | 0.3889               | 0.4733          | 0.4854                   |
| 48       | {Advances in Bayesian networks for industrial proc... | 2025  | 0.5623                 | 0.417                | 0.411           | 0.4791                   |
| 49       | {A Non-asymptotic Approach to Best-Arm Identificat... | 2022  | 0.3706                 | 0.267                | 0.6227          | 0.5093                   |
| 49       | {Graph-Guided Architecture Search for Real-Time Se... | 2020  | 0.517                  | 0.3902               | 0.4987          | 0.5069                   |
| 49       | {DisCo-DSO: Coupling Discrete and Continuous Optim... | 2025  | 0.756                  | 0.5737               | 0.3635          | 0.5401                   |
| 49       | {Symbolic regression as a feature engineering meth... | 2024  | 0.7369                 | 0.6817               | 0.3142          | 0.5044                   |
| 50       | {A Screening Rule for l1-Regularized Ising Model E... | 2017  | 0.3957                 | 0.3054               | 0.5846          | 0.4996                   |
| 50       | {IMAGES AS WEIGHT MATRICES: SEQUENTIAL IMAGE GENER... | 2023  | 0.5664                 | 0.4794               | 0.4994          | 0.5296                   |
| 50       | {Braid: Weaving Symbolic and Neural Knowledge into... | 2022  | 0.7694                 | 0.6401               | 0.3377          | 0.5319                   |
| 50       | {Learning Logic Rules for Document-level Relation ... | 2021  | 0.7568                 | 0.6544               | 0.3009          | 0.5061                   |
| 50       | {Neurosymbolic AI for Reasoning Over Knowledge Gra... | 2025  | 0.8237                 | 0.6986               | 0.2609          | 0.5141                   |
| 51       | {Reinforcement learning based meta-path discovery ... | 2020  | 0.7913                 | 0.7361               | 0.2639          | 0.5013                   |
| 51       | {Multi-level Recommendation Reasoning over Knowled... | 2022  | 0.7592                 | 0.6964               | 0.3036          | 0.5087                   |
| 51       | {EXACTA: Explainable Column Annotation}...            | 2021  | 0.6383                 | 0.5398               | 0.4241          | 0.5205                   |
| 52       | {Deep Generative Models for Spatial Networks}...      | 2021  | 0.6752                 | 0.5979               | 0.301           | 0.4694                   |
| 52       | {Learning Disentangled Representations and Group S... | 2020  | 0.6735                 | 0.561                | 0.3548          | 0.4982                   |
| 52       | {Sequence modeling with hierarchical Deep Generati... | 2017  | 0.7485                 | 0.6864               | 0.2969          | 0.5001                   |
| 52       | {A semi-supervised cross-modal memory bank for cro... | 2024  | 0.5205                 | 0.4201               | 0.4356          | 0.4738                   |
| 53       | {Provably robust boosted decision stumps and trees... | 2019  | 0.5618                 | 0.468                | 0.314           | 0.4255                   |
| 53       | {Addressing Bias and Fairness in Search Systems}...   | 2021  | 0.5492                 | 0.3603               | 0.4681          | 0.5046                   |
| 53       | {Recycling Privileged Learning and Distribution Ma... | 2017  | 0.7771                 | 0.6833               | 0.3167          | 0.5239                   |
| 54       | {Symbols as a Lingua Franca for Bridging Human-AI ... | 2022  | 0.6397                 | 0.5454               | 0.3734          | 0.4932                   |
| 54       | {A Semantic Loss Function for Deep Learning with S... | 2018  | 0.757                  | 0.6047               | 0.3             | 0.5056                   |
| 54       | {Neurosymbolic AI Approach to Attribution in Large... | 2024  | 0.6518                 | 0.5371               | 0.4166          | 0.5225                   |
| 55       | {Beyond Rank-1: Discovering Rich Community Structu... | 2020  | 0.4982                 | 0.3511               | 0.4849          | 0.4909                   |
| 55       | {A self-explanatory contrastive logical knowledge ... | 2023  | 0.7019                 | 0.6178               | 0.2836          | 0.4718                   |
| 55       | {The Case of Aspect in Sentiment Analysis: Seeking... | 2022  | 0.7416                 | 0.6173               | 0.2937          | 0.4953                   |
| 56       | {On dynamic network models and application to caus... | 2019  | 0.5838                 | 0.3943               | 0.4453          | 0.5076                   |
| 56       | {On Reliability Scores for Knowledge Graphs}...       | 2022  | 0.4728                 | 0.3963               | 0.4433          | 0.4566                   |
| 56       | {TCrossE: Cross-space interaction of bicomplex and... | 2025  | 0.7373                 | 0.6838               | 0.3162          | 0.5057                   |
| 57       | {Self-Clustering Hierarchical Multi-Agent Reinforc... | 2025  | 0.3989                 | 0.2995               | 0.5776          | 0.4972                   |
| 57       | {Multi-domain modeling of atrial fibrillation dete... | 2020  | 0.7721                 | 0.6538               | 0.2615          | 0.4912                   |
| 57       | {Identifying pediatric heart murmurs and distingui... | 2024  | 0.6791                 | 0.568                | 0.2964          | 0.4686                   |

---

## Statistical Insights

### Topic Distribution Analysis

- **Largest Topic:** 208 papers
- **Smallest Topic:** 22 papers
- **Topic Size Range:** 186 papers
- **Standard Deviation:** 43.9

### Selection Strategy Performance

The selection strategy successfully identified representative papers across all topics, 
with selection ratios varying appropriately based on cluster size and quality metrics.

### Research Focus Alignment

- **XAI Research Alignment:** 0.146
- **Symbolic AI Alignment:** 0.108
- **Sub-symbolic AI Alignment:** 0.147

The selected papers show sub-symbolic AI-focused 
research focus alignment, indicating the composition of the systematic review.

### Quality Assessment

The selected representatives demonstrate strong balance between:
- **Centrality:** High similarity to topic centroids
- **Diversity:** Adequate coverage of topic variations  
- **Representativeness:** Optimal combination of both factors


### Statistical Validation Results

#### Mann-Whitney U Tests (Non-parametric comparison)
- **Tests Performed:** 6 variables analyzed
- **Significant Differences:** 2/6 tests show significant differences (p < 0.05)
- **Multiple Testing Correction:** Applied Benjamini-Hochberg procedure

**Key Findings:**
- **Similarity To Centroid:** Not significant (p = 0.7490)
- **Diversity Score:** **SIGNIFICANT** (p = 0.0000)
- **Representativeness Score:** **SIGNIFICANT** (p = 0.0000)
- **Xai Alignment:** Not significant (p = 0.6936)
- **Symbolic Alignment:** Not significant (p = 0.9662)
- **Subsymbolic Alignment:** Not significant (p = 0.4397)

#### Effect Size Analysis (Cohen's d)
- **Large Effects (d > 0.8):** 0 variables
- **Medium Effects (0.5 < d < 0.8):** 2 variables

**Effect Sizes:**
- **Similarity To Centroid:** d = 0.033 (negligible) [95% CI: -0.059, 0.124]
- **Diversity Score:** d = 0.573 (medium) [95% CI: 0.480, 0.665]
- **Representativeness Score:** d = 0.636 (medium) [95% CI: 0.543, 0.729]
- **Xai Alignment:** d = -0.014 (negligible) [95% CI: -0.105, 0.078]
- **Symbolic Alignment:** d = 0.012 (negligible) [95% CI: -0.080, 0.103]
- **Subsymbolic Alignment:** d = -0.040 (negligible) [95% CI: -0.132, 0.051]

#### Temporal Bias Assessment
- **Kolmogorov-Smirnov Test:** p = 0.2058
- **Temporal Bias Detected:** No
- **Interpretation:** Selection is temporally representative across publication years

### Research Methodology Validation

The statistical analysis provides rigorous validation of the paper selection methodology:

1. **Selection Effectiveness:** Significant differences between selected and non-selected papers confirm that the algorithm successfully identifies papers with distinct characteristics.

2. **Methodological Soundness:** Effect size analysis quantifies the practical significance of selection criteria.

3. **Bias Assessment:** Temporal analysis ensures selection methodology does not favor specific time periods.

4. **Reproducibility:** All statistical tests include confidence intervals and multiple testing corrections for robust inference.

---

## Research Recommendations

### Priority Reading List

Based on representativeness scores, the following papers are recommended for priority reading:

1. **{Physics-guided neural network for predicting asphalt mixture rutting with balanced accuracy, stability and rationality}** - Deng, Y and Wang, H and Shi, X (Score: 0.570)
2. **{On Tractable Computation of Expected Predictions}** - Khosravi, P and Choi, Y and Liang, Y T and Vergari, A and {Van den Broeck}, G (Score: 0.559)
3. **{Ensemble learning framework for detecting electricity theft in smart grids using weighted average method}** - Zhang, K and Wang, J and Zhu, Y and Si, Y and Yin, S and Zhang, H (Score: 0.547)
4. **{PROMO for Interpretable Personalized Social Emotion Mining}** - Zhang, J and Lee, D W (Score: 0.546)
5. **{REPRESENTATION LEARNING FOR IMPROVED INTERPRETABILITY AND CLASSIFICATION ACCURACY OF CLINICAL FACTORS FROM EEG}** - Honke, G and Higgins, I and Thigpen, N and Miskovic, V and Link, K and Duan, S and Gupta, P and Klawohn, J and Hajcak, G (Score: 0.546)

### Review Strategy

1. **Start with High-Representativeness Papers:** Begin with the papers listed above as they provide 
   optimal coverage of their respective topics.

2. **Topic-Based Deep Dive:** Use the interactive visualizations to explore papers within specific 
   topics of interest.

3. **Methodological Diversity:** The selected papers ensure coverage of diverse methodological 
   approaches within each research area.

### Research Gaps

Areas for potential future research can be identified by examining:
- Topics with low representativeness scores
- Clusters with minimal coverage
- Emerging themes at topic boundaries

---

## Technical Notes

### Methodology

- **Topic Modeling:** BERTopic with HDBSCAN clustering
- **Hierarchy:** HDBSCAN Condensed Tree and Single Linkage analysis
- **Centrality:** Cosine similarity to Cluster Centroid (mean) and Medoid (exemplar)
- **Diversity:** Pairwise similarity-based uniqueness score
- **Selection:** Multi-objective optimization (Centrality + Diversity)

### Data Quality

- All metrics are normalized to [0,1] range
- Outlier papers were processed through multi-strategy reduction
- R1 papers assigned to topics based on maximum centroid similarity

### Reproducibility

- Fixed random seeds ensure consistent results
- Pretrained model reuse (`all-MiniLM-L6-v2`) for augmentation embeddings
- Complete processing pipeline with configuration persistence

---

*Report generated by the Research Analysis Framework - Stage 3: Visualization (Augmented/R1)*