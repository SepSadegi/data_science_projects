## Medical Question Answering with BERT


In this project, we explore how BERT (Bidirectional Encoder Representations from Transformers) is used for medical question answering, focusing on its application, word representations, and answer extraction techniques.

### Medical Question Answering

- **Objective**: To answer questions about medical diagnoses or treatments by extracting relevant information from text passages. This involves identifying the most accurate segment of text that answers a given question.
- **Example**: For the question "What is the drug Forxiga used for?", the system extracts the relevant segment stating "reduces blood glucose levels."

### BERT Model Overview

- **Architecture**: BERT consists of several layers of transformer blocks that process input text. The input includes a question and a passage, separated by special tokens.
- **Word Representation**:
  - **Contextualized vs. Non-Contextualized**:
    - **Non-Contextualized Representations**: Techniques like Word2Vec and GloVe use a single representation for each word, regardless of its context.
    - **Contextualized Representations**: BERT, and its medical variant BioBERT, provide different representations based on surrounding context, allowing for nuanced understanding of words with multiple meanings.

## Handling Word Representations

- **Training**: BERT learns word representations through a process called masked language modeling, where words in a passage are masked, and the model learns to predict them. This approach helps BERT understand context-dependent meanings of words.
- **Visualization**: Words with similar meanings are represented by vectors that are closer in the vector space, while unrelated words are farther apart.

## Answer Extraction

- **Process**:
  1. **Input**: The model receives a question and a passage. It outputs an answer segment from the passage.
  2. **Start and End Scores**: For each word in the passage, BERT generates start and end scores indicating the likelihood of the word being the start or end of the answer.
  3. **Score Computation**: Scores are used to form a grid where each cell represents a possible answer segment. The highest scores indicate the most likely start and end of the answer.
  4. **Prediction**: The model outputs the segment with the highest combined score of start and end probabilities.

## Practical Considerations

- **Label Extraction**: BERT can also be used to automatically generate labels by extracting mentions of diseases from medical reports, aiding in model training for tasks like chest x-ray classification.
- **Training Datasets**: BERT is initially trained on general datasets like SQuAD and then fine-tuned on medical datasets such as BioASQ to adapt it to medical contexts.

## Summary

BERT's advanced capabilities in understanding and representing word meanings in context make it an effective tool for medical question answering and label extraction. Its ability to generate contextualized word representations and accurately extract answers from passages enhances its utility in medical and other domains.
