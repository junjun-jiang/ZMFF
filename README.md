# ZMFF

This is an implementation of the following paper.

```
@article{Hu2023ZMFF,
title = {ZMFF: Zero-shot multi-focus image fusion},
journal = {Information Fusion},
volume = {92},
pages = {127-138},
year = {2023},
}
```

## Preparation

### Dependencies

- python >= 3.6
- pytorch >= 1.5.0
- NVIDIA GPU
- CUDA

### Data Preparation

- [Lytro](http://mansournejati.ece.iut.ac.ir/content/lytro-multi-focus-dataset)
- [Real-MFF](https://github.com/Zancelot/Real-MFF)
- [MFI-WHU](https://github.com/HaoZhang1018/MFI-WHU)

## Testing

For testing, please run:

```shell
python main.py
```

The output results will be stored in `./result`.


