# Date: 2020-05-01 12:00
# Datasources

Training:
    - Bus: https://freesound.org/people/hnminh/packs/36821/
    - Tram: https://freesound.org/people/inkaj/packs/36887/

Validation

# Data Preprocessing
### Normalization
All samples are normalized with librosa.normalize

# Feature Extraction

The following features are extracted using librosa implemented functions:
### Mel Frequency Cepstral Coefficients (MFCC)
window_size: 512, hop size: 256 (50% overlap)

### Spectral Centroid (SC)
window_size: 512, hop size: 256 (50% overlap)

### Zero Crossing Rate (ZCR)
window_size: 512, hop size: 256 (50% overlap)

# Models

### K-Nearest Neighbor as implemented by scikit-learn.

# Results

# References

[1]