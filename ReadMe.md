# Intel Bench
This repo facilitates the benchmarking of SciKit-Learn algorithms and 
appropiate hardware acceleration with:
- RAPIDS (nVidia)
- Intelex CPU acceleration based on SSE and AVX
- Intelex GPU acceleration based on the ARC770 GPU
The benchmark tests varying dataset-sizes and various algorithms. The 
  execution is managed and timed with `timeit`.

# Dataset
The dataset that has been used is [HARTH](https://archive-beta.ics.uci.edu/dataset/779/harth)

# Review
The complete review is posted in [Medium]()