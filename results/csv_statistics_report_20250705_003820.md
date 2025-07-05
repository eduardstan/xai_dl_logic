# Systematic Literature Review - CSV Statistics Report

**Generated on:** 2025-07-05 00:38:20
**Source File:** comprehensive_analysis_20250705_003727.csv
**Parameters:** max_nonzero_alignments = 0

## 📊 Executive Summary

### Dataset Overview
- **Total Papers Analyzed:** 3,735
- **Representative Papers Selected:** 533
- **Non-Representative Papers:** 3,202
- **Selection Ratio:** 14.27%
- **Topics Covered:** 56

### Topic Distribution
- **Average Papers per Topic:** 66.7
- **Average Representatives per Topic:** 9.5
- **Topics with Zero Representatives:** 0
- **Topic Size Range:** 16 - 204 papers

## 🔍 Alignment Analysis

### Alignment Score Distributions

#### XAI Alignment
- **Zero Values:** 208 papers (5.6%)
- **Non-Zero Values:** 3,527 papers (94.4%)
- **Range:** 0.000000 - 0.625000
- **Mean:** 0.147055 ± 0.077375
- **Median:** 0.125000

#### SYMBOLIC Alignment
- **Zero Values:** 604 papers (16.2%)
- **Non-Zero Values:** 3,131 papers (83.8%)
- **Range:** 0.000000 - 0.666667
- **Mean:** 0.107497 ± 0.087375
- **Median:** 0.083333

#### SUBSYMBOLIC Alignment
- **Zero Values:** 871 papers (23.3%)
- **Non-Zero Values:** 2,864 papers (76.7%)
- **Range:** 0.000000 - 0.555556
- **Mean:** 0.150974 ± 0.106636
- **Median:** 0.111111

### Alignment Patterns
- **Papers with ALL three alignments = 0:** 25 (0.7%)
- **Papers with at least one alignment = 0:** 1401 (37.5%)

### Distribution by Number of Non-Zero Alignments
- **0 non-zero alignments:** 25 papers (0.7%)
- **1 non-zero alignments:** 232 papers (6.2%)
- **2 non-zero alignments:** 1144 papers (30.6%)
- **3 non-zero alignments:** 2334 papers (62.5%)

## 📈 Similarity Analysis

### is_similar_to_representative Column
- **Total Rows:** 3,735
- **'False' Values:** 3,201
- **'True' Values:** 1
- **NaN Values:** 533 (14.3%)

### similarity_to_representative Scores
- **Non-Null Values:** 3,202 (85.7%)
- **Null Values:** 533 (14.3%)
- **Range:** 0.183135 - 1.000000
- **Mean:** 0.564023 ± 0.082396

### Similarity Threshold Analysis (Non-Representatives)
- **Papers with similarity ≥ 0.3:** 3191 (99.7%)
- **Papers with similarity ≥ 0.4:** 3088 (96.4%)
- **Papers with similarity ≥ 0.5:** 2551 (79.7%)
- **Papers with similarity ≥ 0.6:** 1096 (34.2%)
- **Papers with similarity ≥ 0.7:** 133 (4.2%)
- **Papers with similarity ≥ 0.8:** 1 (0.0%)
- **Papers with similarity ≥ 0.9:** 1 (0.0%)

## 🎯 Data Quality Assessment & Interpretation

### ✅ INTENDED PATTERNS (Normal Behavior)

1. **NaN values in 'is_similar_to_representative' for representatives**
   - The 533 NaN values correspond exactly to the 533 representatives
   - Representatives don't need similarity scores to themselves - this is by design

2. **Varying zero percentages in alignment columns**
   - XAI: 5.6% zeros
   - Symbolic: 16.2% zeros  
   - Sub-symbolic: 23.3% zeros
   - This reflects natural distribution of research focus keywords in the corpus

3. **High percentage of 'False' in similarity classification**
   - Most papers are legitimately dissimilar to their representatives
   - Only highly similar papers (likely near-duplicates) get marked as 'True'

### ⚠️ POTENTIAL AREAS FOR INVESTIGATION

1. **Very few papers marked as 'similar to representative'**
   - Only 1 out of 3,735 papers
   - Consider if similarity threshold (typically 0.8) is too restrictive

2. **Papers with zero alignment across all categories**
   - 25 papers (0.7%) have no alignment
   - These might be off-topic or require keyword expansion

### 📋 RECOMMENDATIONS

1. **Review similarity threshold** for marking papers as "similar to representative"
2. **Examine papers with zero alignments** to verify they're genuinely relevant
3. **Consider expanding keyword lists** in configuration to capture more domain terms
4. **Overall data quality appears good** - most patterns reflect intended system behavior

## 📝 Papers with ≤ 0 Non-Zero Alignment Scores

**Found 25 papers matching criteria:**


### Paper 1
- **Title:** {Medical Transformer: Universal Encoder for 3-D Brain MRI Analysis}
- **Authors:** Jun, E and Jeong, S and Heo, D -W. and Suk, H -I.
- **Year:** 2024
- **Journal:** IEEE Transactions on Neural Networks and Learning Systems
- **Topic ID:** 1
- **Non-Zero Alignments:** 0

**Abstract:** Transfer learning has attracted considerable attention in medical image analysis because of the limited number of annotated 3-D medical datasets available for training data-driven deep learning models in the real world. We propose Medical Transformer, a novel transfer learning framework that effectively models 3-D volumetric images as a sequence of 2-D image slices. To improve the high-level representation in 3-D-form empowering spatial relations, we use a multiview approach that leverages information from three planes of the 3-D volume, while providing parameter-efficient training. For building a source model generally applicable to various tasks, we pretrain the model using self-supervised learning (SSL) for masked encoding vector prediction as a proxy task, using a large-scale normal, healthy brain magnetic resonance imaging (MRI) dataset. Our pretrained model is evaluated on three downstream tasks: 1) brain disease diagnosis; 2) brain age prediction; and 3) brain tumor segmentation, which are widely studied in brain MRI research. Experimental results demonstrate that our Medical Transformer outperforms the state-of-the-art (SOTA) transfer learning methods, efficiently reducing the number of parameters by up to approximately 92% for classification and regression tasks and 97% for segmentation task, and it also achieves good performance in scenarios where only partial training samples are used.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 2
- **Title:** {Visual Reference Resolution using Attention Memory for Visual Dialog}
- **Authors:** Seo, P H and Lehrmann, A and Han, B and Sigal, L
- **Year:** 2017
- **Journal:** nan
- **Topic ID:** 2
- **Non-Zero Alignments:** 0

**Abstract:** Visual dialog is a task of answering a series of inter-dependent questions given an input image, and often requires to resolve visual references among the questions. This problem is different from visual question answering (VQA), which relies on spatial attention (a.k.a. visual grounding) estimated from an image and question pair. We propose a novel attention mechanism that exploits visual attentions in the past to resolve the current reference in the visual dialog scenario. The proposed model is equipped with an associative attention memory storing a sequence of previous (attention, key) pairs. From this memory, the model retrieves the previous attention, taking into account recency, which is most relevant for the current question, in order to resolve potentially ambiguous references. The model then merges the retrieved attention with a tentative one to obtain the final attention for the current question; specifically, we use dynamic parameter prediction to combine the two attentions conditioned on the question. Through extensive experiments on a new synthetic visual dialog dataset, we show that our model significantly outperforms the state-of-the-art (by approximate to 16 % points) in situations, where visual reference resolution plays an important role. Moreover, the proposed model achieves superior performance ( approximate to 2 % points improvement) in the Visual Dialog dataset [1], despite having significantly fewer parameters than the baselines.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 3
- **Title:** {Relevance-based Infilling for Natural Language Counterfactuals}
- **Authors:** Betti, L and Abrate, C and Bonchi, F and Kaltenbrunner, A and ACM
- **Year:** 2023
- **Journal:** nan
- **Topic ID:** 3
- **Non-Zero Alignments:** 0

**Abstract:** Counterfactual explanations are a natural way for humans to gain understanding and trust in the outcomes of complex machine learning algorithms. In the context of natural language processing, generating counterfactuals is particularly challenging as it requires the generated text to be fluent, grammatically correct, and meaningful. In this study, we improve the current state of the art for the generation of such counterfactual explanations for text classifiers. Our approach, named RELITC (Relevance-based Infilling for Textual Counterfactuals), builds on the idea of masking a fraction of text tokens based on their importance in a given prediction task and employs a novel strategy, based on the entropy of their associated probability distributions, to determine the infilling order of these tokens. Our method uses less time than competing methods to generate counterfactuals that require less changes, are closer to the original text and preserve its content better, while being competitive in terms of fluency. We demonstrate the effectiveness of the method on four different datasets and show the quality of its outcomes in a comparison with human generated counterfactuals.(1)

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 4
- **Title:** {Towards Reliable and Practicable Algorithmic Recourse}
- **Authors:** Lakkaraju, H and ACM
- **Year:** 2021
- **Journal:** nan
- **Topic ID:** 3
- **Non-Zero Alignments:** 0

**Abstract:** As predictive models are increasingly being deployed in high-stakes decision making (e.g., loan approvals), there has been growing interest in developing post hoc techniques which provide recourse to individuals who have been adversely impacted by predicted outcomes. For example, when an individual is denied loan by a predictive model deployed by a bank, they should be informed about reasons for this decision and what can be done to reverse it. While several approaches have been proposed to tackle the problem of generating recourses, these techniques rely heavily on various restrictive assumptions. For instance, these techniques generate recourses under the assumption that the underlying predictive models do not change. In practice, however, models are often updated for a variety of reasons including data distribution shifts. There is little to no research that systematically investigates and addresses these limitations. In this talk, I will discuss some of our recent work that sheds light on and addresses the aforementioned challenges, thereby paving the way for making algorithmic recourse practicable and reliable. First, I will present theoretical and empirical results which demonstrate that the recourses generated by state-of-the-art approaches are often invalidated due to model updates. Next, I will introduce a novel algorithmic framework based on adversarial training to generate recourses that remain valid even if the underlying models are updated. I will conclude the talk by presenting theoretical and empirical evidence for the efficacy of our solutions, and also discussing other open problems in the burgeoning field of algorithmic recourse.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 5
- **Title:** {GRETEL: Graph Counterfactual Explanation Evaluation Framework}
- **Authors:** Prado-Romero, M A and Stilo, G and ACM
- **Year:** 2022
- **Journal:** nan
- **Topic ID:** 3
- **Non-Zero Alignments:** 0

**Abstract:** Machine Learning (ML) systems are a building part of the modern tools which impact our daily life in several application domains. Due to their black-box nature, those systems are hardly adopted in application domains (e.g. health, finance) where understanding the decision process is of paramount importance. Explanation methods were developed to explain how the ML model has taken a specific decision for a given case/instance. Graph Counterfactual Explanations (GCE) is one of the explanation techniques adopted in the Graph Learning domain. The existing works on Graph Counterfactual Explanations diverge mostly in the problem definition, application domain, test data, and evaluation metrics, and most existing works do not compare exhaustively against other counterfactual explanation techniques present in the literature. We present GRETEL, a unified framework to develop and test GCE methods in several settings. GRETEL is a highly extensible evaluation framework which promotes Open Science and the reproducibility of the evaluation by providing a set of well-defined mechanisms to integrate and manage easily: both real and synthetic datasets, ML models, state-of-the-art explanation techniques, and evaluation measures. Lastly, we also show the experiments conducted to integrate and test several existing scenarios (datasets, measures, explainers).

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 6
- **Title:** {S-LIME: Stabilized-LIME for Model Explanation}
- **Authors:** Zhou, Z Z and Hooker, G and Wang, F and MACHINERY, ASSOC COMP
- **Year:** 2021
- **Journal:** nan
- **Topic ID:** 3
- **Non-Zero Alignments:** 0

**Abstract:** An increasing number of machine learning models have been deployed in domains with high stakes such as finance and healthcare. Despite their superior performances, many models are black boxes in nature which are hard to explain. There are growing efforts for researchers to develop methods to interpret these black-box models. Post hoc explanations based on perturbations, such as LIME [39], are widely used approaches to interpret a machine learning model after it has been built. This class of methods has been shown to exhibit large instability, posing serious challenges to the effectiveness of the method itself and harming user trust. In this paper, we propose S-LIME, which utilizes a hypothesis testing framework based on central limit theorem for determining the number of perturbation points needed to guarantee stability of the resulting explanation. Experiments on both simulated and real world data sets are provided to demonstrate the effectiveness of our method.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 7
- **Title:** {ADT²R: Adaptive Decision Transformer for Dynamic Treatment Regimes in Sepsis}
- **Authors:** Jeon, E and Choi, J -H. and Suk, H -I.
- **Year:** 2025
- **Journal:** IEEE Transactions on Neural Networks and Learning Systems
- **Topic ID:** 6
- **Non-Zero Alignments:** 0

**Abstract:** Dynamic treatment regimes (DTRs), which comprise a series of decisions taken to select adequate treatments, have attracted considerable attention in the clinical domain, especially from sepsis researchers. Existing sepsis DTR learning studies are mainly based on offline reinforcement learning (RL) approaches working on electronic healthcare records data. However, a trained policy may choose a treatment different from a human clinician's prescription. Furthermore, most of them do not consider: 1) heterogeneity in sepsis; 2) short-term transitions; and 3) the relationship between a patient's health state and the prescription. We propose a novel framework, an adaptive decision transformer for DTR (ADT2R), which recommends an optimal treatment action for each time step depending on the heterogeneity of the sepsis and a patient's evolving health states. Specifically, we devise a trajectory-optimization-based module to be trained with supervision for treatments and adaptively aggregate the multihead self-attentions by deliberating on various inherent time-varying patterns among sepsis patients. Furthermore, we estimate the patient's health state by adopting an actor-critic (AC) algorithm and inform the treatment recommendation by learning about its short-term changes. We validated the effectiveness of the proposed framework on the Medical Information Mart for Intensive Care III (MIMIC-III) dataset, an extensive intensive care database, by demonstrating performance comparable to the state-of-the-art methods.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 8
- **Title:** {On Path Integration of Grid Cells: Group Representation and Isotropic Scaling}
- **Authors:** Gao, R Q and Xie, J W and Wei, X X and Zhu, S C and Wu, Y N
- **Year:** 2021
- **Journal:** nan
- **Topic ID:** 10
- **Non-Zero Alignments:** 0

**Abstract:** Understanding how grid cells perform path integration calculations remains a fundamental problem. In this paper, we conduct theoretical analysis of a general representation model of path integration by grid cells, where the 2D self-position is encoded as a higher dimensional vector, and the 2D self-motion is represented by a general transformation of the vector. We identify two conditions on the transformation. One is a group representation condition that is necessary for path integration. The other is an isotropic scaling condition that ensures locally conformal embedding, so that the error in the vector representation translates conformally to the error in the 2D self-position. Then we investigate the simplest transformation, i.e., the linear transformation, uncover its explicit algebraic and geometric structure as matrix Lie group of rotation, and explore the connection between the isotropic scaling condition and a special class of hexagon grid patterns. Finally, with our optimization-based approach, we manage to learn hexagon grid patterns that share similar properties of the grid cells in the rodent brain. The learned model is capable of accurate long distance path integration. Code is available at https://github.com/ruiqigao/grid- cell-path.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 9
- **Title:** {Confound Removal and Normalization in Practice: A Neuroimaging Based Sex Prediction Case Study}
- **Authors:** More, S and Eickhoff, S B and Caspers, J and Patil, K R
- **Year:** 2021
- **Journal:** nan
- **Topic ID:** 13
- **Non-Zero Alignments:** 0

**Abstract:** Machine learning (ML) methods are increasingly being used to predict pathologies and biological traits using neuroimaging data. Here controlling for confounds is essential to get unbiased estimates of generalization performance and to identify the features driving predictions. However, a systematic evaluation of the advantages and disadvantages of available alternatives is lacking. This makes it difficult to compare results across studies and to build deployment quality models. Here, we evaluated two commonly used confound removal schemes-whole data confound regression (WDCR) and cross-validated confound regression (CVCR)-to understand their effectiveness and biases induced in generalization performance estimation. Additionally, we study the interaction of the confound removal schemes with Z-score normalization, a common practice in ML modelling. We applied eight combinations of confound removal schemes and normalization (pipelines) to decode sex from resting-state functional MRI (rfMRI) data while controlling for two confounds, brain size and age. We show that both schemes effectively remove linear univariate and multivariate confounding effects resulting in reduced model performance with CVCR providing better generalization estimates, i.e., closer to out-of-sample performance than WDCR. We found no effect of normalizing before or after confound removal. In the presence of dataset and confound shift, four tested confound removal procedures yielded mixed results, raising new questions. We conclude that CVCR is a better method to control for confounding effects in neuroimaging studies. We believe that our in-depth analyses shed light on choices associated with confound removal and hope that it generates more interest in this problem instrumental to numerous applications.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 10
- **Title:** {FacTeR-Check: Semi-automated fact-checking through semantic similarity and natural language inference}
- **Authors:** Martin, A and Huertas-Tato, J and Huertas-Garcia, A and Villar-Rodriguez, G and Camacho, D
- **Year:** 2022
- **Journal:** KNOWLEDGE-BASED SYSTEMS
- **Topic ID:** 14
- **Non-Zero Alignments:** 0

**Abstract:** Our society produces and shares overwhelming amounts of information through Online Social Net-works (OSNs). Within this environment, misinformation and disinformation have proliferated, becom-ing a public safety concern in most countries. Allowing the public and professionals to efficiently find reliable evidence about the factual veracity of a claim is a crucial step to mitigate this harmful spread. To this end, we propose FacTeR-Check, a multilingual architecture for semi-automated fact-checking and hoaxes propagation analysis that can be used to implement applications designed for both the general public and for fact-checking organisations. FacTeR-Check implements three different modules relying on the XLM-RoBERTa Transformer architecture to evaluate semantic similarity, to calculate natural language inference and to build search queries through automatic keywords extraction and Named-Entity Recognition. The three modules have been validated using state-of-the-art benchmark datasets, exhibiting good performance in all of them. Besides, FacTeR-Check is employed to collect and label a dataset, called NLI19-SP, composed of more than 40,000 tweets supporting or denying 60 hoaxes related to COVID-19, released publicly. Finally, an analysis of the data collected in this dataset is provided, which allows to obtain a deep insight of how disinformation operated during the COVID-19 pandemic in Spanish-speaking countries. (c) 2022 The Author(s). Published by Elsevier B.V.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 11
- **Title:** {Model-agnostic and diverse explanations for streaming rumour graphs}
- **Authors:** Nguyen, T T and Phan, T C and Nguyen, M H and Weidlich, M and Yin, H and Jo, J and Nguyen, Q V H
- **Year:** 2022
- **Journal:** KNOWLEDGE-BASED SYSTEMS
- **Topic ID:** 14
- **Non-Zero Alignments:** 0

**Abstract:** The propagation of rumours on social media poses an important threat to societies, so that various techniques for rumour detection have been proposed recently. Yet, existing work focuses on what entities constitute a rumour, but provides little support to understand why the entities have been classified as such. This prevents an effective evaluation of the detected rumours as well as the design of countermeasures. In this work, we argue that explanations for detected rumours may be given in terms of examples of related rumours detected in the past. A diverse set of similar rumours helps users to generalize, i.e., to understand the properties that govern the detection of rumours. Since the spread of rumours in social media is commonly modelled using feature-annotated graphs, we propose a queryby-example approach that, given a rumour graph, extracts the k most similar and diverse subgraphs from past rumours. The challenge is that all of the computations require fast assessment of similarities between graphs. To achieve an efficient and adaptive realization of the approach in a streaming setting, we present a novel graph representation learning technique and report on implementation considerations. Our evaluation experiments show that our approach outperforms baseline techniques in delivering meaningful explanations for various rumour propagation behaviours. (C) 2022 Elsevier B.V. All rights reserved.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 12
- **Title:** {Fault Detection in Wastewater Treatment Plants: Application of Autoencoders Models with Streaming Data}
- **Authors:** Salles, R and Mendes, J and Ribeiro, R P and Gama, J
- **Year:** 2023
- **Journal:** nan
- **Topic ID:** 15
- **Non-Zero Alignments:** 0

**Abstract:** Water is a fundamental human resource and its scarcity is reflected in social, economic and environmental problems. Water used in human activities must be treated before reusing or returning to nature. This treatment takes place in wastewater treatment plants (WWTPs), which need to perform their functions with high quality, low cost, and reduced environmental impact. This paper aims to identify failures in real-time, using streaming data to provide the necessary preventive actions to minimize damage to WWTPs, heavy fines and, ultimately, environmental hazards. Convolutional and Long short-term memory (LSTM) autoencoders (AEs) were used to identify failures in the functioning of the dissolved oxygen sensor used in WWTPs. Five faults were considered (drift, bias, precision degradation, spike and stuck) in three different scenarios with variations in the appearance order, intensity and duration of the faults. The best performance, considering different model configurations, was achieved by Convolutional-AE.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 13
- **Title:** {A Geometric Approach to Predicting Bounds of Downstream Model Performance}
- **Authors:** Goode, B J and Datta, D and MACHINERY, ASSOC COMP
- **Year:** 2020
- **Journal:** nan
- **Topic ID:** 16
- **Non-Zero Alignments:** 0

**Abstract:** This paper presents the motivation and methodology for including model application criteria into baseline analysis. We will focus on detailing the interplay between the common measures of mean square error (MSE) and accuracy as it relates to perceived model performance. MSE is a common aggregate measure for the performance of predictive regression models. The advantages are numerous. MSE is agnostic to the choice of model given that the set of possible outcome values are defined on the appropriate metric space. In practice, decisions on how to subsequently use a trained model are based on predictive performance, relative to a baseline where input features are not used - colloquially a "random model". However, the relative performance gains of a model in terms of MSE to the baseline does not guarantee commensurate gains when deployed in downstream applications, systems, or processes. This paper demonstrates one derivation of a distribution to qualify MSE performance for multi-class decision making systems desiring a certain level of accuracy. The model error is qualified through comparison to relevant baselines tied to the application suited to evaluating individual outcome performance criteria.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 14
- **Title:** {Quantifying the Confidence of Anomaly Detectors in Their Example-Wise Predictions}
- **Authors:** Perini, L and Vercruyssen, V and Davis, J
- **Year:** 2021
- **Journal:** nan
- **Topic ID:** 19
- **Non-Zero Alignments:** 0

**Abstract:** Anomaly detection focuses on identifying examples in the data that somehow deviate from what is expected or typical. Algorithms for this task usually assign a score to each example that represents how anomalous the example is. Then, a threshold on the scores turns them into concrete predictions. However, each algorithm uses a different approach to assign the scores, which makes them difficult to interpret and can quickly erode a user's trust in the predictions. This paper introduces an approach for assessing the reliability of any anomaly detector's example-wise predictions. To do so, we propose a Bayesian approach for converting anomaly scores to probability estimates. This enables the anomaly detector to assign a confidence score to each prediction which captures its uncertainty in that prediction. We theoretically analyze the convergence behaviour of our confidence estimate. Empirically, we demonstrate the effectiveness of the framework in quantifying a detector's confidence in its predictions on a large benchmark of datasets.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 15
- **Title:** {Learning V1 Simple Cells with Vector Representation of Local Content and Matrix Representation of Local Motion}
- **Authors:** Gao, R Q and Xie, J W and Huang, S Y and Ren, Y F and Zhu, S C and Wu, Y N and Intelligence, Assoc Advancement Artificial
- **Year:** 2022
- **Journal:** nan
- **Topic ID:** 20
- **Non-Zero Alignments:** 0

**Abstract:** This paper proposes a representational model for image pairs such as consecutive video frames that are related by local pixel displacements, in the hope that the model may shed light on motion perception in primary visual cortex (V1). The model couples the following two components: (1) the vector representations of local contents of images and (2) the matrix representations of local pixel displacements caused by the relative motions between the agent and the objects in the 3D scene. When the image frame undergoes changes due to local pixel displacements, the vectors are multiplied by the matrices that represent the local displacements. Thus the vector representation is equivariant as it varies according to the local displacements. Our experiments show that our model can learn Gabor-like filter pairs of quadrature phases. The profiles of the learned filters match those of simple cells in Macaque V1. Moreover, we demonstrate that the model can learn to infer local motions in either a supervised or unsupervised manner. With such a simple model, we achieve competitive results on optical flow estimation.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 16
- **Title:** {Focused or Stuck Together: Multimodal Patterns Reveal Triads' Performance in Collaborative Problem Solving}
- **Authors:** Vrzakova, H and Amon, M J and Stewart, A and Duran, N D and D'Mello, S K and Machinery, Assoc Comp
- **Year:** 2020
- **Journal:** nan
- **Topic ID:** 21
- **Non-Zero Alignments:** 0

**Abstract:** Collaborative problem solving (CPS) in virtual environments is an increasingly important context of 21st century learning. However, our understanding of this complex and dynamic phenomenon is still limited. Here, we examine unimodal primitives (activity on the screen, speech, and body movements), and their multimodal combinations during remote CPS. We analyze two datasets where 116 triads collaboratively engaged in a challenging visual programming task using video conferencing software. We investigate how UI-interactions, behavioral primitives, and multimodal patterns were associated with teams' subjective and objective performance outcomes. We found that idling with limited speech (i.e., silence or backchannel feedback only) and without movement was negatively correlated with task performance and with participants' subjective perceptions of the collaboration. However, being silent and focused during solution execution was positively correlated with task performance. Results illustrate that in some cases, multimodal patterns improved the predictions and improved explanatory power over the unimodal primitives. We discuss how the findings can inform the design of real-time interventions for remote CPS.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 17
- **Title:** {When Comparing to Ground Truth is Wrong: On Evaluating GNN Explanation Methods}
- **Authors:** Faber, L and Moghaddam, A K and Wattenhofer, R and MACHINERY, ASSOC COMP
- **Year:** 2021
- **Journal:** nan
- **Topic ID:** 27
- **Non-Zero Alignments:** 0

**Abstract:** We study the evaluation of graph explanation methods. The state of the art to evaluate explanation methods is to first train a GNN, then generate explanations, and finally compare those explanations with the ground truth. We show five pitfalls that sabotage this pipeline because the GNN does not use the ground-truth edges. Thus, the explanation method cannot detect the ground truth. We propose three novel benchmarks: (i) pattern detection, (ii) community detection, and (iii) handling negative evidence and gradient saturation. In a re-evaluation of state-of-the-art explanation methods, we show paths for improving existing methods and highlight further paths for GNN explanation research.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 18
- **Title:** {A robust algorithm for explaining unreliable machine learning survival models using the Kolmogorov-Smirnov bounds}
- **Authors:** Kovalev, M S and Utkin, L V
- **Year:** 2020
- **Journal:** NEURAL NETWORKS
- **Topic ID:** 28
- **Non-Zero Alignments:** 0

**Abstract:** A new robust algorithm based on the explanation method SurvLIME called SurvLIME-KS is proposed for explaining machine learning survival models. The algorithm is developed to ensure robustness to cases of a small amount of training data or outliers of survival data. The first idea behind SurvLIME-KS is to apply the Cox proportional hazards model to approximate the black-box survival model at the local area around a test example due to the linear relationship of covariates in the model. The second idea is to incorporate the well-known Kolmogorov-Smirnov bounds for constructing sets of predicted cumulative hazard functions. As a result, the robust maximin strategy is used, which aims to minimize the average distance between cumulative hazard functions of the explained black-box model and of the approximating Cox model, and to maximize the distance over all cumulative hazard functions in the interval produced by the Kolmogorov-Smirnov bounds. The maximin optimization problem is reduced to the quadratic program. Various numerical experiments with synthetic and real datasets demonstrate the SurvLIME-KS efficiency. (c) 2020 Elsevier Ltd. All rights reserved.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 19
- **Title:** {Shapley Values and Meta-Explanations for Probabilistic Graphical Model Inference}
- **Authors:** Liu, Y F and Chen, C and Liu, Y Z and Zhang, X and Xie, S H and MACHINERY, ASSOC COMP
- **Year:** 2020
- **Journal:** nan
- **Topic ID:** 32
- **Non-Zero Alignments:** 0

**Abstract:** Probabilistic graphical models, such as Markov random fields (MRF), exploit dependencies among random variables to model a rich family of joint probability distributions. Inference algorithms, such as belief propagation (BP), can effectively compute the marginal posteriors for decision making. Nonetheless, inferences involve sophisticated probability calculations and are difficult for humans to interpret. Among all existing explanation methods for MRFs, no method is designed for fair attributions of an inference outcome to elements on the MRF where the inference takes place. Shapley values provide rigorous attributions but so far have not been studied on MRFs. We thus define Shapley values for MRFs to capture both probabilistic and topological contributions of the variables on MRFs. We theoretically characterize the new definition regarding independence, equal contribution, additivity, and submodularity. As brute-force computation of the Shapley values is challenging, we propose GraphShapley, an approximation algorithm that exploits the decomposability of Shapley values, the structure of MRFs, and the iterative nature of BP inference to speed up the computation. In practice, we propose meta-explanations to explain the Shapley values and make them more accessible and trustworthy to human users. On four synthetic and nine real-world MRFs, we demonstrate that GraphShapley generates sensible and practical explanations.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 20
- **Title:** {Towards Description of Block Model on Graph}
- **Authors:** Bai, Z L and Ravi, S S and Davidson, I
- **Year:** 2021
- **Journal:** nan
- **Topic ID:** 34
- **Non-Zero Alignments:** 0

**Abstract:** Existing block modeling methods can detect communities as blocks. However it remains a challenge to easily explain to a human why nodes belong to the same block. Such a description is very useful for answering why people in the same community tend to interact cohesively. In this paper we explore a novel problem: Given a block model already found, describe the blocks using an auxiliary set of information. We formulate a combinatorial optimization problem which finds a unique disjunction of the auxiliary information shared by the nodes either in the same block or between a pair of different blocks. The former terms intra-block description, the latter inter-block description. Given an undirected graph and its k-block model, our method generates k + k(k-1)/2 different descriptions. If the tags are descriptors of events occurring at the vertices, our descriptions can be interpreted as common events occurring within blocks and between blocks. We show that this problem is intractable even for simple cases, e.g., when the underlying graph is a tree with just two blocks. However, simple and efficient ILP formulations and algorithms exist for its relaxation and yield insights different from a state-of-the-art related work in unsupervised description. We empirically show the power of our work on multiple real-world large datasets.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 21
- **Title:** {Grad-SAM: Explaining Transformers via Gradient Self-Attention Maps}
- **Authors:** Barkan, O and Hauon, E and Caciularu, A and Katz, O and Malkiel, I and Armstrong, O and Koenigstein, N and ACM
- **Year:** 2021
- **Journal:** nan
- **Topic ID:** 35
- **Non-Zero Alignments:** 0

**Abstract:** Transformer-based language models significantly advanced the state-of-the-art in many linguistic tasks. As this revolution continues, the ability to explain model predictions has become a major area of interest for the NLP community. In this work, we present Gradient Self-Attention Maps (Grad-SAM) - a novel gradient-based method that analyzes self-attention units and identifies the input elements that explain the model's prediction the best. Extensive evaluations on various benchmarks show that Grad-SAM obtains significant improvements over state-of-the-art alternatives.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 22
- **Title:** {Improving Software Defects Detection: An In-Depth Analysis of Machine Learning Methods and Static Analysis Tools for Greater Accuracy}
- **Authors:** Bobade, V and Puri, C
- **Year:** 2025
- **Journal:** nan
- **Topic ID:** 45
- **Non-Zero Alignments:** 0

**Abstract:** Enhancing software quality and reducing testing expenses requires better software fault detection. To guarantee the dependability and usability of software, Software Defect Prediction (SDP) uses machine learning approaches to anticipate and address any problems early in the development cycle. Numerous machine learning techniques, such as Support Vector Machines (SVM), Random Forest (RF), Logistic Regression, Na{\"{i}}ve Bayes, Multilayer Perceptron (MLP), Decision Stump, J48, Lazy IBK, and ZeroR, are thoroughly examined in this work. With an emphasis on feature selection as a crucial step in increasing prediction accuracy and lower computing costs, these approaches are assessed for their capacity to detect software flaws precisely. This research study demonstrates how these models may improve fault identification and prevention procedures using historical records from sources such as the PROMISE repository. The study highlights the best practices for machine learning-based software quality assurance, lowering expenses while guaranteeing superior results.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 23
- **Title:** {UniRank: Unimodal Bandit Algorithms for Online Ranking}
- **Authors:** Gauthier, C S and Gaudel, R and Fromont, E and Chaudhuri, K and Jegelka, S and Song, L and Szepesvari, C and Niu, G and Sabato, S
- **Year:** 2022
- **Journal:** nan
- **Topic ID:** 49
- **Non-Zero Alignments:** 0

**Abstract:** We tackle, in the multiple-play bandit setting, the online ranking problem of assigning L items to K predefined positions on a web page in order to maximize the number of user clicks. We propose a generic algorithm, UniRank, that tackles state-of-the-art click models. The regret bound of this algorithm is a direct consequence of the unimodality-like property of the bandit setting with respect to a graph where nodes are ordered sets of indistinguishable items. The main contribution of UniRank is its O (L/Delta log T) regret for T consecutive assignments, where Delta relates to the reward-gap between two items. This regret bound is based on the usually implicit condition that two items may not have the same attractiveness. Experiments against state-of-the-art learning algorithms specialized or not for different click models, show that our method has better regret performance than other generic algorithms on real life and synthetic datasets.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 24
- **Title:** {Causality-guided Graph Learning for Session-based Recommendation}
- **Authors:** Yu, D E and Li, Q and Yin, H Z and Xu, G D and ACM
- **Year:** 2023
- **Journal:** nan
- **Topic ID:** 49
- **Non-Zero Alignments:** 0

**Abstract:** Session-based recommendation systems (SBRs) aim to capture user preferences over time by taking into account the sequential order of interactions within sessions. One promising approach within this domain is session graph-based recommendation, which leverages graph-based models to represent and analyze user sessions. However, current graph-based methods for SBRs mainly rely on attention or pooling mechanisms that are prone to exploiting shortcut paths and thus lead to suboptimal recommendations. To address this issue, we propose Causality-guided Graph Learning for Session-based Recommendation (CGSR) that is capable of blocking shortcut paths on the session graph and exploring robust causal connections capturing users' true preferences. Specifically, by employing back-door adjustment of causality, we can generate a distilled causal session graph capturing causal relations among items. CGSR then performs high-order aggregation on the distilled graph, incorporating information from various edge types, to estimate the session preference of the user. This enables us to provide more accurate recommendations grounded in causality while offering fine-grained interaction explanations by highlighting influential items in the graph. Extensive experiments on three datasets show the superior performance of CGSR compared to state-of-the-art SBRs.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

### Paper 25
- **Title:** {Error Variance, Fairness, and the Curse on Minorities}
- **Authors:** Beauxis-Aussalet, E
- **Year:** 2021
- **Journal:** nan
- **Topic ID:** 54
- **Non-Zero Alignments:** 0

**Abstract:** Machine learning systems can make more errors for certain populations and not others, and thus create discriminations. To assess such fairness issue, errors are typically compared across populations. We argue that we also need to account for the variability of errors in practice, as the errors measured in test data may not be exactly the same in real-life data (called target data). We first introduce statistical methods for estimating random error variance in machine learning problems. The methods estimate how often errors would exceed certain magnitudes, and how often the errors of a population would exceed that of another (e.g., by more than a certain range). The methods are based on well-established sampling theory, and the recently introduced Sample-to-Sample estimation. The latter shows that small target samples yield high error variance, even if the test sample is very large. We demonstrate that, in practice, minorities are bound to bear higher variance, thus amplified error and bias. This can occur even if the test and training sets are accurate, representative, and extremely large. We call this statistical phenomenon the curse on minorities, and we show examples of its impact with basic classification and regression problems. Finally, we outline potential approaches to protect minorities from such curse, and to develop variance-aware fairness assessments.

**Alignment Scores:**
- **XAI Alignment:** 0.000000
- **Symbolic Alignment:** 0.000000
- **Sub-symbolic Alignment:** 0.000000

## 📊 Summary Statistics

### Selected Papers Metrics (if available)

- **Average Centrality:** 0.591810 ± 0.099347
- **Average Diversity:** 0.390418 ± 0.082881
- **Average Representativeness:** 0.440766 ± 0.047720
- **Average XAI Alignment:** 0.135553
- **Average Symbolic Alignment:** 0.096154
- **Average Sub-symbolic Alignment:** 0.138837

---
*Report generated by CSV Statistics Analyzer*
*Source: results/comprehensive_analysis_20250705_003727.csv*
*Generated at: 2025-07-05 00:38:20*
