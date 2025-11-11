# Implementing-Mamba-Network-from-Scratch
This repository is created to implement linear sequences modeling architectures for both study and research. In this project, I will use **JAX** and **Pytorch** to build models, from basic RNNs up to Mamba model, to solve **Speech Enhancement** and **Text Classification tasks** and then compare their performance. I will also try to build the **Transformer** architecture; however, this type of model is not yet within the scope of this project. Therefore, if I cannot finish it on time, I will use the predefined architecture in Pytorch. 
## Introduction.
Sequences modeling remains a fundamental part of Artificial Intelligence, as it allows AI to capture the structure of sequence data. While there are various applications of this field, the most well-known task today is Language Modeling. 

Recently, a powerful Architecutre called **Transformer** has been dominating this field. Its **attention mechanism** allows the model to capture complex relationships between tokens within the text. However, the attention mechanism faces a critical problem: **its computional cost grows quadradically** with sequence length. This limits the Transfomer from capture longer sequences. 

As a result, much research is now focusing on **linear models** which have showed comparable performance to the Transformer while maintaining a low computional cost. 

This repository is created focusing on implementing these models for **educational purposes**. Throughout the implementation, I will try to explain the concepts behind these models as simply as possible. The repository is implemented in the simplest way possible; more efficient implementations will be placed in a seperate repository.
## II. Tranditonal RNNs
## III. Linear Recurrence Unit (LRU)
## IV. Structured State Space sequence model (S4)
## V. Mamba 