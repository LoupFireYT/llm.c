# The Little Book of Artificial Intelligence

This is _The Little Book of llm.c_ - a friendly companion that explains in plain
English what
[Andrej Karpathy’s `llm.c` repository](https://github.com/karpathy/llm.c) does.

The original repo itself is the real star: it is a tiny but powerful reference
implementation of GPT-2, written in C (with optional CUDA). The Little Book
simply walks alongside it, unpacking the ideas step by step so that beginners
can follow along without feeling overwhelmed. Think of this as a map that helps
you explore the landscape of `llm.c`.

```bash
Small file, giant dream,
llm.c whispers tokens,
worlds unfold in text.
```

## Formats

- [Download PDF](releases/book.pdf) - print-ready
- [Download EPUB](releases/book.epub) - e-reader friendly
- [View LaTeX](releases/book.tex) - `.tex` source
- [Read on GitHub Pages](https://little-book-of.github.io/llm.c/) - online
  website

## Build it yourself (Quarto)

We use [Quarto](https://quarto.org/docs/get-started/) to generate all outputs.

Preview locally:

```bash
quarto preview
```

Render outputs:

```bash
# All configured formats
quarto render

# Individual formats
quarto render --to html     # site into docs/
quarto render --to pdf      # docs/book.pdf
quarto render --to epub     # docs/book.epub
quarto render --to latex    # docs/book-latex/book.tex
```

## Table of Contents (1–100)

### Chapter 1 - Orientation

1. What llm.c Is - explains the scope and goals of the project
2. Repository Tour - shows how the files and folders are organized
3. Makefile Targets & Flags - describes the build options for CPU and CUDA
4. Quickstart: CPU Reference Path - walks through running train\_gpt2.c
5. Quickstart: 1-GPU Legacy Path - introduces train\_gpt2\_fp32.cu
6. Quickstart: Modern CUDA Path - explains train\_gpt2.cu
7. Starter Artifacts & Data Prep - describes the starter pack and datasets
8. Debugging Tips & IDE Stepping - covers gdb, lldb, and IDE usage
9. Project Constraints & Readability - explains why the repo stays simple
10. Community and Learning Path - points to discussions and resources

### Chapter 2 - Data, Tokenization, and Loaders

11. GPT-2 Tokenizer Artifacts - explains the gpt2\_tokenizer.bin file
12. Binary Dataset Format - describes how token data is stored
13. Dataset Scripts - shows how Tiny Shakespeare and OpenWebText are prepared
14. DataLoader Design - explains batching and epochs
15. EvalLoader - describes validation data handling
16. Sequence Length Budgeting - explains memory tradeoffs
17. Reproducibility and Seeding - shows how runs stay consistent
18. Error Surfaces from Bad Data - discusses common pitfalls
19. Tokenization Edge Cases - explains UNK, EOS, and BOS tokens
20. Data Hygiene and Logging - covers dataset quality tracking

### Chapter 3 - Model Definition & Weights

21. GPT-2 Config - outlines vocab, layers, heads, and channels
22. Parameter Tensors and Layout - explains weight storage in memory
23. Embedding Tables - shows token and positional embeddings
24. Attention Stack - introduces QKV projections
25. MLP Block - describes linear layers and activations
26. LayerNorm - explains normalization in theory and practice
27. Residual Streams - shows how skip connections work
28. Loss Head - describes tied embeddings and logits
29. Checkpoint Loading - explains importing PyTorch weights
30. Parameter Counting - teaches sanity checks for model size

### Chapter 4 - CPU Inference

31. Forward Pass Walkthrough - outlines how inference works
32. Token and Positional Embeddings - step-by-step embedding lookup
33. Attention on CPU - explains matmuls and masking
34. MLP on CPU - describes GEMMs and activations
35. LayerNorm on CPU - explains normalization in detail
36. Residual Adds - shows signal flow with skip connections
37. Cross-Entropy Loss on CPU - explains loss calculation
38. Putting It All Together - walks through gpt2\_forward
39. OpenMP Pragmas - shows parallel loops for speed
40. CPU Memory Footprint - discusses efficiency

### Chapter 5 - Training Loop (CPU Path)

41. Skeleton of Training Loop - introduces the main structure
42. AdamW Implementation - shows the optimizer in C
43. Learning Rate Schedulers - explains cosine and warmup
44. Gradient Accumulation - describes micro-batching
45. Logging and Progress - covers progress reporting
46. Validation Runs - adds evaluation during training
47. Checkpointing State - explains saving weights and optimizer
48. Reproducibility Issues - discusses divergences
49. Command-Line Flags - explains arguments
50. Example Training Logs - provides sample output

### Chapter 6 - Testing, Profiling, & Parity

51. Debug State Structs - explains diagnostic data
52. test\_gpt2.c - compares CPU with PyTorch
53. test\_gpt2cu.cu - compares CUDA with PyTorch
54. Matching Tolerances - discusses acceptable ranges
55. Profiling with profile\_gpt2.cu - shows performance measurement
56. Measuring FLOPs - explains how to count FLOPs
57. Reproducing Loss Curves - validates model correctness
58. Common CUDA Pitfalls - lists common build issues
59. FlashAttention Testing - covers USE\_CUDNN option
60. From Unit Test to Training - connects tests to real runs

### Chapter 7 - CUDA Training Internals

61. CUDA Architecture Overview - explains kernels and streams
62. Matrix Multiplication via cuBLAS - shows GPU GEMMs
63. Attention Kernels - explains FlashAttention
64. Mixed Precision - describes FP16 and BF16 training
65. Loss Scaling - explains stability tricks
66. Activation Checkpointing - trades memory for compute
67. GPU Memory Planning - explains allocation strategies
68. Kernel Launch Configs - covers occupancy planning
69. CUDA Error Handling - explains debugging approaches
70. dev/cuda Directory - describes custom kernels

### Chapter 8 - Multi-GPU & Multi-Node

71. Data Parallelism - explains splitting work across GPUs
72. MPI Process Model - shows GPU affinity setup
73. NCCL All-Reduce - explains gradient synchronization
74. Multi-GPU Trainers - describes execution setup
75. Multi-Node Bootstrapping - covers MPI launch
76. SLURM and PMIx - describes scheduler caveats
77. Debugging Multi-GPU Hangs - lists common problems
78. Scaling Stories - explains scaling GPT-2 sizes
79. NCCL Tuning - covers efficiency tricks
80. Common Multi-GPU Errors - documents fixes

### Chapter 9 - Extending the Codebase

81. Custom Kernels in dev/cuda - explains how to extend kernels
82. New Dataset Pipelines - shows adding data loaders
83. Adding a New Optimizer - explains extending optimizers
84. Adding a Scheduler - describes new LR schedules
85. Alternative Attention - explores new mechanisms
86. Profiling New Kernels - explains performance testing
87. PyTorch Oracle - describes PyTorch as a reference
88. Beyond GPT-2 - explains adapting to other models
89. Porting Playbook - shows moving to Go, Rust, Metal
90. Keeping it Clean - explains repo minimalism

### Chapter 10 - Reproductions & Roadmap

91. Reproducing GPT-2 124M - single-node run
92. Reproducing GPT-2 355M - tricks for larger models
93. Reproducing GPT-2 774M - scaling up further
94. Reproducing GPT-2 1.6B - 8×H100 run in 24h
95. CPU-only Fine-Tuning - explains Tiny Shakespeare training
96. Cost and Time Estimates - shows expected resources
97. Hyperparameter Sweeps - introduces sweep.sh
98. Validating Loss Curves - explains curve checks
99. Future Work - lists next steps for the repo
100. Community and Discussions - points to collaboration

## Contributing

We welcome contributions: fixing typos, clarifying sections, adding diagrams, or
improving examples.

## Citation

Nguyen, Duc-Tam (2025). _The Little Book of llm.c_

```
@book{Nguyen2025TLBoAlgo,
  author = {Duc-Tam Nguyen},
  title  = {The Little Book of llm.c},
  year   = {2025},
  url    = {https://github.com/little-book-of/llm.c}
}
```

## License

Licensed under **CC BY-NC-SA 4.0**. See [LICENSE](LICENSE) for details.
