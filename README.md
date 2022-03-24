# wavelet-texture-synthesis

Code for the paper: "Generalized Rectifier Wavelet Covariance Model For texture Synthesis" (Brochard, Zhang, Mallat, ICLR 2022) <https://openreview.net/pdf?id=ziRLU3Y2PN_>.

**Requirements**:
- Pytorch (version >=1.8.0)
- Kymatio (`pip install kymatio`) to create the Morlet wavelet filters.
- Numpy, scipy, scikit-image, Pillow, tqdm, matplotlib

Create the wavelet filters by running `python build-filters.py`

**To generate a synthesis**:
- Grayscale images: `python synthesis/gray.py`
- Color images: `python synthesis/color.py`

Specify the input image with the argument --image (e.g. `python synthesis/color.py --image honeycomb`). The argument --save stores the synthesis in the 'results' folder, in .npy format. Other arguments, to specify the model, can be found in 'synthesis/gray.py' and 'synthesis/color.py'.

**To generate a synthesis using PS model**:
- Grayscale images: 
- Color images: 
p.s. You need download the original Matlab code (http://www.cns.nyu.edu/~lcv/texture/)
