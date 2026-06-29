# Transformer Architecture Understanding

This repository contains PyTorch implementations of Transformer architectures built from scratch to understand the core components introduced in the paper *"Attention Is All You Need"*.

## Projects

### 1. Machine Translation (`machine_translation.py`)

An Encoder-Decoder Transformer trained for English-to-French machine translation using the OPUS Books dataset.

**Key Concepts**

* Tokenization and vocabulary creation
* Positional Encoding
* Multi-Head Self Attention
* Encoder-Decoder Attention
* Feed Forward Networks
* Masking
* Sequence-to-Sequence Learning

---

### 2.Sentiment analysis using Vanilla Transformer (`sentiment_analysis_vanilla_transformer.py`)

A Transformer built from scratch and applied to sentiment classification on the IMDB dataset.

**Key Concepts**

* Transformer Encoder and Decoder
* Self Attention Mechanism
* Residual Connections
* Layer Normalization
* Positional Encoding
* Sequence Classification

---

## Features

* Transformer implemented from scratch using PyTorch
* Custom Multi-Head Attention implementation
* Custom Positional Encoding
* Encoder and Decoder blocks
* Attention masking
* Xavier weight initialization
* End-to-end training pipeline

---

## Tech Stack

* Python
* PyTorch
* Hugging Face Datasets
* TorchText

---

## Learning Outcomes

This project helped in understanding:

* Scaled Dot-Product Attention
* Multi-Head Attention
* Encoder-Decoder Architecture
* Positional Encoding
* Transformer Training Pipeline
* Machine Translation
* Sequence Classification

---

## Author

Built for learning and understanding Transformer architectures from first principles.
