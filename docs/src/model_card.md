# Model Card: DALL·E dVAE

Following [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993) and [Lessons from
Archives (Jo & Gebru)](https://arxiv.org/pdf/1912.10389.pdf), we're providing some information about about the discrete
VAE (dVAE) that was used to train DALL·E.

## Model Details

The dVAE was developed by researchers at OpenAI to reduce the memory footprint of the transformer trained on the
text-to-image generation task. The details involved in training the dVAE are described in [the paper][dalle_paper]. This
model card describes the first version of the model, released in February 2021. The model consists of a convolutional
encoder and decoder whose architectures are described [here](dall_e/encoder.py) and [here](dall_e/decoder.py), respectively.
For questions or comments about the models or the code release, please file a Github issue.

## Model Use

### Intended Use

The model is intended for others to use for training their own generative models.

### Out-of-Scope Use Cases

This model is inappropriate for high-fidelity image processing applications. We also do not recommend its use as a
general-purpose image compressor.

## Training Data

The model was trained on publicly available text-image pairs collected from the internet. This data consists partly of
[Conceptual Captions][cc] and a filtered subset of [YFCC100M][yfcc100m]. We used a subset of the filters described in
[Sharma et al.][cc_paper] to construct this dataset; further details are described in [our paper][dalle_paper]. We will
not be releasing the dataset.

## Performance and Limitations

The heavy compression from the encoding process results in a noticeable loss of detail in the reconstructed images. This
renders it inappropriate for applications that require fine-grained details of the image to be preserved.

[dalle_paper]: https://arxiv.org/abs/2102.12092
[cc]: https://ai.google.com/research/ConceptualCaptions
[cc_paper]: https://www.aclweb.org/anthology/P18-1238/
[yfcc100m]: http://projects.dfki.uni-kl.de/yfcc100m/
