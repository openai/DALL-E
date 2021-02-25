{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import io\n",
    "import os, sys\n",
    "import requests\n",
    "import PIL\n",
    "\n",
    "import torch\n",
    "import torchvision.transforms as T\n",
    "import torchvision.transforms.functional as TF\n",
    "\n",
    "from dall_e          import map_pixels, unmap_pixels, load_model\n",
    "from IPython.display import display, display_markdown\n",
    "\n",
    "target_image_size = 256\n",
    "\n",
    "def download_image(url):\n",
    "    resp = requests.get(url)\n",
    "    resp.raise_for_status()\n",
    "    return PIL.Image.open(io.BytesIO(resp.content))\n",
    "\n",
    "def preprocess(img):\n",
    "    s = min(img.size)\n",
    "    \n",
    "    if s < target_image_size:\n",
    "        raise ValueError(f'min dim for image {s} < {target_image_size}')\n",
    "        \n",
    "    r = target_image_size / s\n",
    "    s = (round(r * img.size[1]), round(r * img.size[0]))\n",
    "    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)\n",
    "    img = TF.center_crop(img, output_size=2 * [target_image_size])\n",
    "    img = torch.unsqueeze(T.ToTensor()(img), 0)\n",
    "    return map_pixels(img)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# This can be changed to a GPU, e.g. 'cuda:0'.\n",
    "dev = torch.device('cpu')\n",
    "\n",
    "# For faster load times, download these files locally and use the local paths instead.\n",
    "enc = load_model(\"https://cdn.openai.com/dall-e/encoder.pkl\", dev)\n",
    "dec = load_model(\"https://cdn.openai.com/dall-e/decoder.pkl\", dev)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = preprocess(download_image('https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg'))\n",
    "display_markdown('Original image:')\n",
    "display(T.ToPILImage(mode='RGB')(x[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch.nn.functional as F\n",
    "\n",
    "z_logits = enc(x)\n",
    "z = torch.argmax(z_logits, axis=1)\n",
    "z = F.one_hot(z, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()\n",
    "\n",
    "x_stats = dec(z).float()\n",
    "x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))\n",
    "x_rec = T.ToPILImage(mode='RGB')(x_rec[0])\n",
    "\n",
    "display_markdown('Reconstructed image:')\n",
    "display(x_rec)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
