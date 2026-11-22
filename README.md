# Poppy Reproduction

This repository contains the modified Poppy source code and experimental results for reproducing the Poppy paper.

## Contents

- `poppy/`: Poppy source code, including local modifications and experiment scripts.
- `out_full_base/`: MoGe-2 baseline results.
- `out_full_imgonly/`: Image-shift-only results.
- `out_full_poppy/`: Full Poppy results.
- `out_test_base/`, `out_test_poppy/`: Initial test results.
- `out_steps_*/`: Optimization-step experiments.
- `figs/`, `figs_full/`: Additional visualization figures.

## Environment

- Python 3.12
- PyTorch 2.7.0a0+7c8ec84dab.nv25.03
- NumPy 1.26.4
- Transformers 5.3.0
- Diffusers 0.37.1
- OpenCV 4.10.0

## Notes

Dataset files, model weights, Hugging Face caches, and other large external resources are not included in this repository.
