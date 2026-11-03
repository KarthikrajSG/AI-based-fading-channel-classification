AI-Based Fading Channel ClassificationAn end-to-end Machine Learning and Deep Learning pipeline for automatic classification of wireless fading channel environments (e.g., AWGN, Rayleigh, Rician, Nakagami-$m$, Shadowed Fading) using raw IQ samples, Channel Impulse Responses (CIR), and time-frequency representations (spectrograms).Table of ContentsOverviewKey FeaturesChannel Models SupportedRepository StructureInstallationDataset Generation & FeaturesUsage1. Generate Dataset2. Model Training3. EvaluationSupported Model ArchitecturesResults & PerformanceCitationLicenseOverviewAccurate identification of wireless propagation environments is crucial for adaptive modulation and coding (AMC), dynamic resource allocation, and cognitive radio systems. Traditional statistical hypothesis testing methods often fail under low Signal-to-Noise Ratio (SNR) conditions or complex multipath environments.This repository implements data-driven AI models (1D-CNN, 2D-CNN, LSTM, ResNet, and classical ML baselines) to classify fading channels directly from received signals across a wide range of SNR levels (e.g., $-10\text{ dB}$ to $+20\text{ dB}$).Key FeaturesSynthetic Wireless Dataset Generator: Built-in signal processing pipeline to generate synthetic IQ data for various channel fading distributions.Multiple Feature Representations: Supports raw time-series IQ samples, high-order statistical cumulants, CIR estimates, and time-frequency STFT spectrograms.Deep Learning Suite: Implementations of 1D Convolutional Neural Networks (1D-CNN), ResNet-18, Bi-LSTM, and Transformer-based models.Robust SNR Benchmarking: Automated scripts to evaluate model classification accuracy across distinct SNR levels.Modular Pipeline: Clean separation of signal generation, pre-processing, training, evaluation, and visualization.Channel Models SupportedThe pipeline generates and classifies the following fading channels:Channel TypeProbability Density Function (PDF) / DescriptionKey ParametersAWGNAdditive White Gaussian Noise baselineNoise variance $\sigma^2$RayleighNo Line-of-Sight (NLOS) multipath fadingScale parameter $\sigma$RicianDominant Line-of-Sight (LOS) path + multipathRice factor $K$ (dB)Nakagami-$m$Generalized fading for severe to mild multipathFading parameter $m$, Spread $\Omega$Log-Normal ShadowingLarge-scale fading/shadowing effectStandard deviation $\sigma_{dB}$Repository Structure├── data/
│   ├── raw/                # Generated raw IQ signal datasets
│   └── processed/          # Extracted features (Spectrograms, Cumulants)
├── models/
│   ├── cnn1d.py            # 1D-CNN for raw time-series IQ data
│   ├── resnet_2d.py        # 2D-ResNet for Spectrogram classification
│   ├── lstm_model.py       # Recurrent models for temporal features
│   └── ml_baselines.py     # SVM, Random Forest, XGBoost baselines
├── utils/
│   ├── channel_generator.py# Synthetic channel signal generation
│   ├── feature_extraction.py# STFT, Cumulants, CIR estimation
│   └── metrics.py          # SNR-dependent accuracy, Confusion Matrix
├── weights/                # Saved model checkpoints
├── config.yaml             # Global configuration file for hyperparams
├── generate_data.py        # Dataset generation script
├── train.py                # Model training script
├── evaluate.py             # Evaluation & plotting script
├── requirements.txt        # Python dependency list
└── README.md               # Repository documentation
InstallationClone the repository:git clone https://github.com/your-username/ai-fading-channel-classification.git
cd ai-fading-channel-classification
Create and activate a virtual environment:python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
Install dependencies:pip install -r requirements.txt
Dataset Generation & FeaturesIf you do not have a pre-recorded dataset, you can generate synthetic IQ signals using our configurable channel generator:Signal Modulation: BPSK, QPSK, 16-QAMSample Length: 1024 samples per frameSNR Range: $-10\text{ dB}$ to $+20\text{ dB}$ in steps of $2\text{ dB}$Extracted Features:Raw IQ Samples: Tensor shape $(2, N)$ representing In-Phase and Quadrature components.STFT Spectrograms: Time-frequency image representation $(H, W, C)$ for 2D-CNNs.Higher-Order Cumulants: Statistical moments ($C_{20}, C_{21}, C_{40}, C_{42}$) for traditional ML classifiers.Usage1. Generate DatasetGenerate $10,000$ signal frames per fading channel across specified SNR levels:python generate_data.py --samples 10000 --snr_min -10 --snr_max 20 --output_dir data/raw/
2. Model TrainingTrain a 1D-CNN model on the raw IQ dataset:python train.py --config config.yaml --model cnn1d --epochs 50 --batch_size 64
To train a 2D-ResNet on Spectrogram features:python train.py --config config.yaml --model resnet18 --feature spectrogram --epochs 50
3. EvaluationEvaluate trained models and generate Accuracy vs. SNR plots along with Confusion Matrices:python evaluate.py --model_path weights/cnn1d_best.pth --test_data data/raw/test.pkl
Supported Model Architectures1D-CNN: Lightweight deep architecture designed for real-time edge processing on raw IQ frames.ResNet-18 / ResNet-50: Deep residual networks processing Short-Time Fourier Transform (STFT) spectrograms.Bi-LSTM: Captures long-range temporal dependencies in time-varying fading channels.SVM / Random Forest: Baseline machine learning classifiers using statistical cumulants.Results & PerformanceBelow is an example classification performance breakdown across varying SNR levels:SNR LevelAWGN AccuracyRayleigh AccuracyRician (K=3dB)Nakagami-m (m=0.5)Overall Average$-10\text{ dB}$68.5%55.2%52.0%50.1%56.45%$0\text{ dB}$92.1%84.6%81.3%80.0%84.50%$+10\text{ dB}$99.8%96.4%95.1%94.8%96.52%$+20\text{ dB}$100.0%99.2%98.7%98.5%99.10%(Note: Replace table values with your experimental results.)CitationIf you find this repository useful in your research, please consider citing:@misc{fading_channel_classification_2026,
  author = {Your Name},
  title = {AI-Based Fading Channel Classification in Wireless Communication Systems},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/your-username/ai-fading-channel-classification}}
}
