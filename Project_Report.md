# Project Report: AI-Based Fading Channel Type Classification

## Abstract
This project classifies wireless fading channels as Rayleigh or Rician using machine learning. Complex channel samples are simulated, corrupted by additive white Gaussian noise (AWGN) at different signal-to-noise ratios (SNRs), converted into statistical and signal-based features, and classified using SVM, Random Forest and MLP. Accuracy, macro-F1, confusion matrices and SNR-wise results are used for evaluation.

## 1. Introduction
Wireless signals travel through multiple paths because of reflection, diffraction and scattering. The combination of these paths causes fading. Rayleigh fading models a channel without a dominant line-of-sight (LOS) component. Rician fading models a channel with a dominant LOS component plus scattered components. This project uses AI to distinguish these two channel types.

## 2. Objectives
- Generate Rayleigh and Rician channel data.
- Add AWGN at multiple SNR levels.
- Extract statistical and signal-based features.
- Train SVM, Random Forest and MLP classifiers.
- Evaluate performance using accuracy, macro-F1 and confusion matrices.

## 3. Theory
**Rayleigh fading:** The complex channel is modelled as many scattered components with no dominant LOS path. Its envelope follows a Rayleigh distribution under the standard assumptions.

**Rician fading:** A dominant LOS component is combined with diffuse scattered components. The K-factor is the ratio of LOS power to diffuse power. The simulation uses K = 6 dB.

**AWGN and SNR:** SNR(dB) = 10 log10(signal power/noise power). Lower SNR means noise is stronger relative to the signal. Tested levels are [-5, 0, 5, 10, 15, 20, 25] dB.

## 4. Dataset
The dataset contains 3,500 balanced synthetic channel snapshots, each with 256 complex IQ samples. Samples are generated at seven SNR levels from -5 dB to 25 dB. Average channel power is normalized before adding complex AWGN. The raw dataset is in `data/channel_iq_dataset.npz`; features and labels are in `data/channel_features.csv`.

## 5. Feature Extraction
The 21 features include envelope mean, standard deviation, variance, skewness, kurtosis, median, percentiles, RMS, crest factor and coefficient of variation; power mean, standard deviation and kurtosis; phase standard deviation and phase-difference variation; real/imaginary spread, I/Q covariance and autocorrelation magnitudes at lags 1 and 2.

## 6. Classifiers
- **SVM:** RBF kernel with standardized features.
- **Random Forest:** ensemble of 250 decision trees.
- **MLP:** feed-forward neural network with hidden layers of 64 and 32 neurons.
A stratified 75%/25% train/test split is used. Numeric class labels are used internally for compatibility with machine-learning libraries.

## 7. Evaluation Metrics
Accuracy is the fraction of correct predictions. Macro-F1 averages the F1-score of both classes equally. A confusion matrix counts correct and incorrect predictions. SNR-wise evaluation calculates accuracy and macro-F1 separately at each noise level.

## 8. Results
### Overall model comparison
        model  overall_accuracy  overall_macro_f1
          SVM          1.000000          1.000000
Random Forest          0.997714          0.997714
          MLP          0.997714          0.997714

Best overall accuracy in this run: **SVM**, accuracy = **1.000**, macro-F1 = **1.000**.

### Performance by SNR
        model  snr_db  accuracy  macro_f1
          SVM      -5  1.000000  1.000000
          SVM       0  1.000000  1.000000
          SVM       5  1.000000  1.000000
          SVM      10  1.000000  1.000000
          SVM      15  1.000000  1.000000
          SVM      20  1.000000  1.000000
          SVM      25  1.000000  1.000000
Random Forest      -5  0.981982  0.981910
Random Forest       0  1.000000  1.000000
Random Forest       5  1.000000  1.000000
Random Forest      10  1.000000  1.000000
Random Forest      15  1.000000  1.000000
Random Forest      20  1.000000  1.000000
Random Forest      25  1.000000  1.000000
          MLP      -5  0.981982  0.981945
          MLP       0  1.000000  1.000000
          MLP       5  1.000000  1.000000
          MLP      10  1.000000  1.000000
          MLP      15  1.000000  1.000000
          MLP      20  1.000000  1.000000
          MLP      25  1.000000  1.000000

See the included accuracy and macro-F1 plots, plus individual confusion matrices, for visual evaluation.

## 9. Conclusion
The project implements the complete workflow of channel generation, feature extraction, machine-learning classification and performance evaluation. The comparison shows how SNR affects the ability to distinguish Rayleigh and Rician fading.

## 10. Limitations and Future Work
This is a simulation-based educational project, not a measurement campaign. It uses two channel types, a fixed Rician K-factor and a simplified independent-sample model. Future work can use measured IQ data, multiple K-factors, correlated time-varying channels, frequency-selective channels and additional fading distributions such as Nakagami-m.

## 11. Reproducibility
Install dependencies with `pip install -r requirements.txt`, then run `python run_project.py`. The source code regenerates the dataset, extracts features and trains/evaluates the classifiers.
