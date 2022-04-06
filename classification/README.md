# Adaptive Fourier Neural Operators for Image Classification

This repository contains PyTorch implementation for AFNONet with ImageNet classification.

The Adaptive Fourier Neural Operator is a token mixer that learns to mix in the Fourier domain. AFNO is based on a principled foundation of operator learning which allows us to frame token mixing as a continuous global convolution without any dependence on the input resolution. This principle was previously used to design FNO, which solves global convolution efficiently in the Fourier domain and has shown promise in learning challenging PDEs. To handle challenges in visual representation learning such as discontinuities in images and high resolution inputs, we propose principled architectural modifications to FNO which results in memory and computational efficiency. This includes imposing a block-diagonal structure on the channel mixing weights, adaptively sharing weights across tokens, and sparsifying the frequency modes via soft-thresholding and shrinkage. The resulting model is highly parallel with a quasi-linear complexity and has linear memory in the sequence size.

![intro](figs/mixer.jpeg)

Our code is based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit), and [GFNet](https://github.com/raoyongming/GFNet).

[[arXiv]](https://arxiv.org/pdf/2111.13587.pdf)

## Usage

### Requirements

- torch>=1.8.0
- torchvision
- timm

_Note_: To use the `rfft2` and `irfft2` functions in PyTorch, you need to install PyTorch>=1.8.0. Complex numbers are supported after PyTorch 1.6.0, but the `fft` API is slightly different from the current version.

**Data preparation**: download and extract ImageNet images from http://image-net.org/. The directory structure should be

```
│ILSVRC2012/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Training

#### ImageNet

To train AFNONet models on ImageNet from scratch, run:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_afnonet.py --batch-size 128 --data-path /path/to/ILSVRC2012/ --hidden-size 256 --num-layers 12 --fno-blocks 4
```

## License

MIT License

## Citation

If you find our work useful in your research, please consider citing:

```
@article{guibas2021adaptive,
  title={Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers},
  author={Guibas, John and Mardani, Morteza and Li, Zongyi and Tao, Andrew and Anandkumar, Anima and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2111.13587},
  year={2021}
}
```
