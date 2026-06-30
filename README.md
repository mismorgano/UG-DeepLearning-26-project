# Pix2Pix denoising, a comparative

In this repository we'll make a comparative between two models both based on the paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004)
for denoising tasks.
The difference between the models resides in the generative model $G$, for one of them we'll use
the [U-Net](https://arxiv.org/pdf/1505.04597) architecture and for the another one will use the 
[Restormer](https://arxiv.org/pdf/2111.09881).



## References

```bibtex
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}

@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}
```

## Credits & Acknowledgment

The code is inspired in [pytorch-pix2pix][pix2pix] mainly on how to structure the `code` and the `options`
for training and testing.

[pix2pix]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
[unet]: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
[restormer]: https://github.com/swz30/Restormer
