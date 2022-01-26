# wavelet-texture-synthesis

Code for the paper: "Generalized Rectifier Wavelet Covariance Model For texture Synthesis" (Brochard, Zhang, Mallat, ICLR 2022) https://openreview.net/pdf?id=ziRLU3Y2PN_.

**Requirements**:
- Pytorch (version >=1.8.0)
- Kymatio (`pip install kymatio`) to create the Morlet wavelet filters.
- Numpy, scipy, scikit-image, PIL, tqdm, matplotlib

Create the wavelet filters by running `python build-filters.py`

**To run a synthesis**:
- Grayscale images: `python synthesis/gray.py --plot`
- Color images: `python synthesis/color.py --plot'
