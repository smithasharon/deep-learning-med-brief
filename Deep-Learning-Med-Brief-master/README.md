# Med-Brief: Deep Learning-Based Summarization of Clinical Data

This project provides a framework for training and evaluating text summarization models on clinical research abstracts. It includes data preprocessing, model inference (using baseline, GPT-2, T5, and BART), and comprehensive evaluation using ROUGE metrics. The workflow is implemented in Jupyter notebooks and leverages Hugging Face Transformers, scikit-learn, and visualization libraries.

---

## Features

- **Data Preprocessing**: Cleans and splits clinical abstracts and summaries from `ClinicalData.csv`.
- **Multiple Summarization Models**: Supports Baseline (first 3 sentences), GPT-2, T5, and BART for text summarization.
- **Evaluation Metrics**: Calculates ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum for model comparison.
- **Visualization**: Provides bar, line, box, stacked bar, and radar charts for ROUGE score analysis.
- **Batch Processing**: Efficient batch inference for large datasets.
- **Extensible**: Modular code for adding new models or metrics.

---

## Project Structure

```
.
├── ClinicalData.csv
├── Custom_Model_Training.ipynb
├── Pretrained_Model_Evaluation.ipynb
└── 2.3/
```

- **ClinicalData.csv**: Main dataset with clinical abstracts and summaries.
- **Custom_Model_Training.ipynb**: Notebook for training custom models (CNN, RNN, LSTM, biLSTM, etc.).
- **Pretrained_Model_Evaluation.ipynb**: Notebook for evaluating pretrained models and visualizing results.
- **2.3/**: (Purpose unspecified, possibly for versioning or additional resources.)

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- Jupyter Notebook or VS Code with Jupyter extension

### Install Required Libraries

Run the following commands in your terminal or notebook cells:

```sh
pip install pandas scikit-learn transformers sacrebleu rouge-score datasets matplotlib seaborn nltk tqdm torch
```

---

## Usage

### 1. Prepare the Data

Ensure `ClinicalData.csv` is present in the project root.

### 2. Run the Notebooks

- **Custom Model Training**:  
  Open `Custom_Model_Training.ipynb` to train and evaluate custom neural models.
- **Pretrained Model Evaluation**:  
  Open `Pretrained_Model_Evaluation.ipynb` to run summarization with pretrained models and visualize ROUGE scores.

### 3. Model Inference & Evaluation

- The notebooks will:
  - Load and preprocess the data.
  - Generate summaries using different models.
  - Compute and visualize ROUGE metrics.

---

## Testing

To validate the code and outputs, run all cells in the notebooks.  
Unit tests for custom functions can be added as needed.

---

## Build & Production

No explicit build step is required.  
For reproducibility, consider exporting notebooks as HTML or Python scripts.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to your branch (`git push origin feature-branch`).
5. Open a Pull Request.

For major changes, please open an issue first to discuss your proposal.

---

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [ROUGE Score](https://github.com/google-research/text-metrics)

---

## Example

To start the evaluation notebook, run:

```sh
jupyter notebook Pretrained_Model_Evaluation.ipynb
```

Or open in VS Code and run all cells.

---
