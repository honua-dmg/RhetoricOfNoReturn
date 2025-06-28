# Rhetoric of No Return: Predicting Conflict Escalation from News Analysis

This project analyzes news articles leading up to major geopolitical events to determine if the rhetoric and sentiment in the media can predict whether an event will escalate into a conflict. It uses natural language processing (NLP) and machine learning to model and predict outcomes based on textual data.

## Project Overview

The core of this project is to analyze the language used in news coverage in the weeks before a significant event. By quantifying metrics like sentiment, rhetoric, and certainty, we can build a model that predicts whether a given situation will result in a conflict or de-escalation.

## Features

- **Data Collection**: Gathers news articles from various sources leading up to specified events.
- **Data Processing**: Cleans and preprocesses the text data for analysis.
- **Feature Engineering**: Extracts key features from the text, including:
  - Sentiment Analysis
  - Rhetoric and Certainty Scores
  - Article Relevance
- **Predictive Modeling**: Uses machine learning models to predict event outcomes based on the engineered features.
- **Visualization**: Generates plots and charts to visualize the data and model results, including feature importance and confusion matrices.

## Getting Started

### Prerequisites

- Python 3.x
- Pip for package management

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/RhetoricOfNoReturn.git
   cd RhetoricOfNoReturn
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The project is divided into several scripts that should be run in the following order:

1. **`DataCollection.py`**: Collects raw data for the events specified in `events.csv`.
2. **`ProcessData.py`**: Cleans and processes the raw data.
3. **`Aggregrate.py`**: Aggregates the processed data into weekly features for each event.
4. **`work.ipynb`**: A Jupyter Notebook for analysis, model training, and visualization.

## Project Structure

- **`DataCollection.py`**: Script to collect raw news articles.
- **`ProcessData.py`**: Script for cleaning and preprocessing text data.
- **`Aggregrate.py`**: Script to aggregate features on a weekly basis.
- **`work.ipynb`**: Jupyter Notebook for analysis, modeling, and visualization.
- **`events.csv`**: The main file containing the list of events to analyze.
- **`final_model_data.csv`**: The final dataset used for model training.
- **`/raw_data`**: Directory containing the raw, unprocessed data.
- **`/processed_data`**: Directory for cleaned and preprocessed data.
- **`/features`**: Directory containing the engineered features for each event.
- **`/initial_visuals`**: Directory for storing visualizations and plots.

## Results

The project generates several visualizations to help understand the model's performance, including:

- **Confusion Matrix**: To evaluate the accuracy of the classification model.
- **Classification Report**: Provides precision, recall, and F1-score for the model.
- **Feature Importance**: Shows which features are most influential in predicting outcomes.

These visualizations can be found in the root directory of the project.
