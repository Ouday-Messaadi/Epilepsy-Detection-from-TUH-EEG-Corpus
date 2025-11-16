# Epilepsy Prediction using Deep Learning

A comprehensive signal processing and deep learning project for predicting epileptic seizures using EEG data from the TUH EEG Seizure Corpus dataset.

## 👥 Authors

- **Ouday Messaadi** - [ouday.messaadi@etudiant-enit.utm.tn](mailto:ouday.messaadi@etudiant-enit.utm.tn)
- **Oussema Cherni** - [oussema.cherni@etudiant-enit.utm.tn](mailto:oussema.cherni@etudiant-enit.utm.tn)

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualization](#visualization)
- [References](#references)
- [License](#license)

## 🎯 Overview

This project implements a Convolutional Neural Network (CNN) for epilepsy seizure prediction using EEG signal processing. The system analyzes brain wave patterns to distinguish between seizure and non-seizure states, potentially enabling early warning systems for epileptic patients.

### Key Objectives

- Statistical analysis of the TUH EEG Seizure Corpus dataset
- Comprehensive data visualization of seizure vs non-seizure EEG patterns
- Advanced signal preprocessing pipeline
- CNN-based classification model
- Performance evaluation and hyperparameter optimization

## 📊 Dataset

**TUH EEG Seizure Corpus Dataset**

The Temple University Hospital (TUH) EEG Seizure Corpus is one of the largest publicly available datasets for seizure detection, containing:

- Multi-channel EEG recordings (26-29 channels)
- Annotated seizure and non-seizure segments
- Diverse patient recordings with various seizure types
- Real-world clinical EEG data

### Dataset Statistics

The project includes comprehensive statistical analysis covering:
- Distribution of seizure vs non-seizure recordings
- Channel-wise signal characteristics
- Temporal patterns and frequencies
- Patient demographics and recording durations

## 🔄 Project Pipeline

### 1. Statistical Analysis

Comprehensive exploratory data analysis of the TUH dataset including:
- Data distribution analysis
- Seizure type categorization
- Recording duration statistics
- Channel availability analysis

### 2. Data Visualization

Utilizing **MNE-Python** for advanced EEG visualization:
- Time-series plots of EEG signals
- Comparison of seizure vs non-seizure patterns
- Topographic brain maps showing spatial activation
- Spectral analysis and frequency domain visualization

### 3. Data Preprocessing Pipeline

#### Step 1: Selection
- Channel selection and standardization
- Artifact removal
- Quality control filtering

#### Step 2: Filtering
- Bandpass filtering for relevant frequency ranges
- Noise reduction
- Signal normalization

#### Step 3: Transformation
- Spectrogram generation using Short-Time Fourier Transform (STFT)
- Time-frequency representation
- Grayscale conversion for CNN input

#### Step 4: Construction
- Window-based segmentation (4-second windows)
- Sequence generation from spectrograms
- Dataset partitioning (train/eval/test splits)

### 4. CNN Model Architecture

Deep Convolutional Neural Network designed for spectrogram-based seizure detection:

**Input**: Spectrogram sequences (time-frequency representations)

**Architecture Features**:
- Multiple convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Dropout layers for regularization
- Fully connected layers for classification
- Binary output (seizure/non-seizure)

**Hyperparameter Optimization**:
- Grid search over 8 different configurations
- Optimized learning rate, batch size, and architecture depth
- Cross-validation for robust model selection

### 5. Results & Evaluation

Comprehensive evaluation metrics including:
- Training and validation accuracy curves
- Test set performance
- Confusion matrix analysis
- Precision, recall, and F1-score
- ROC curves and AUC metrics

## ✨ Features

### Signal Processing
- **Spectrogram Generation**: Per-channel time-frequency analysis
- **Window-based Processing**: 4-second sliding windows for temporal context
- **Multi-channel Integration**: Utilizing 26-29 EEG channels

### Visualization Enhancements
- **PCA-based RGB Representation**: Dimensionality reduction of channels for intuitive visualization
- **Topographic Maps**: Spatial distribution of brain activity
- **Comparative Plots**: Side-by-side seizure vs non-seizure analysis

### Model Features
- **End-to-end Learning**: From raw spectrograms to predictions
- **Transfer Learning Ready**: Architecture adaptable for fine-tuning
- **Real-time Prediction**: Optimized for efficient inference

## 🛠️ Installation

### Prerequisites

```bash
Python >= 3.8
CUDA-compatible GPU (recommended)
```

### Dependencies

```bash
pip install -r requirements.txt
```

**Key Libraries**:
```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow / pytorch
mne
scipy
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/epilepsy-prediction.git
cd epilepsy-prediction

# Install dependencies
pip install -r requirements.txt

# Download TUH dataset
# Follow instructions at: https://www.isip.piconepress.com/projects/tuh_eeg/
```

## 💻 Usage

### 1. Data Preparation

```python
# Run statistical analysis
python scripts/statistical_analysis.py

# Generate visualizations
python scripts/visualize_eeg.py --patient_id PATIENT_ID
```

### 2. Preprocessing

```python
# Execute preprocessing pipeline
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed
```

### 3. Model Training

```python
# Train CNN model
python scripts/train.py --config configs/best_config.yaml

# With hyperparameter search
python scripts/train.py --grid_search --num_configs 8
```

### 4. Evaluation

```python
# Evaluate on test set
python scripts/evaluate.py --model_path models/best_model.h5 --test_data data/test
```

### 5. Prediction

```python
# Make predictions on new EEG data
python scripts/predict.py --model_path models/best_model.h5 --input_file data/new_recording.edf
```

## 📈 Results

### Model Performance

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | XX.X% | XX.X% | XX.X% |
| Precision | XX.X% | XX.X% | XX.X% |
| Recall | XX.X% | XX.X% | XX.X% |
| F1-Score | XX.X% | XX.X% | XX.X% |

*(Replace with actual results)*

### Key Findings

- Successfully distinguished seizure patterns from normal EEG activity
- Spectrogram-based representation proved effective for CNN learning
- Optimal window size of 4 seconds balanced temporal context and computational efficiency
- Grid search identified optimal hyperparameters improving baseline performance

## 🎨 Visualization

The project includes rich visualizations for understanding EEG patterns:

### EEG Signal Comparisons
- **Non-seizure recordings**: Stable, rhythmic patterns
- **Seizure recordings**: High-amplitude, irregular spikes and waves

### Topographic Maps
- Spatial distribution of electrical activity across the scalp
- Clear differences in activation patterns during seizures

### Spectrograms
- Time-frequency analysis showing:
  - Frequency content evolution over time
  - Energy concentration during seizure events
  - Channel-specific patterns

### PCA-based RGB Sequences
For visualization purposes, the 26-29 channels are reduced to 3 components using PCA, creating RGB-like images that capture multi-channel dynamics in an intuitive format.

## 📚 References

### Dataset
- **TUH EEG Seizure Corpus**: [https://www.isip.piconepress.com/projects/tuh_eeg/](https://www.isip.piconepress.com/projects/tuh_eeg/)

### Libraries and Tools
- **MNE-Python**: EEG data processing and visualization
- **TensorFlow/PyTorch**: Deep learning framework
- **SciPy**: Signal processing utilities
- **Scikit-learn**: Machine learning utilities

### Related Work
*(Add relevant papers and research articles)*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Temple University Hospital for providing the EEG Seizure Corpus
- Course instructors and advisors
- Open-source community for excellent tools and libraries

## 📧 Contact

For questions, suggestions, or collaborations:

- **Ouday Messaadi**: ouday.messaadi@etudiant-enit.utm.tn
- **Oussema Cherni**: oussema.cherni@etudiant-enit.utm.tn

---

**Note**: This project was developed as part of the Signal Processing for Data Science course. It is intended for educational and research purposes.