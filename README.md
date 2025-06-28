# Rhetoric of No Return: Predicting Conflict Escalation from News Analysis

This project analyzes news articles leading up to major geopolitical events to determine if the rhetoric and sentiment in the media can predict whether an event will escalate into a conflict. It uses natural language processing (NLP) and machine learning to model and predict outcomes based on textual data.

## Project Overview
REPORT: https://docs.google.com/document/d/1qsqmNexrNksXaQsTToHuV5RZs6_jWBOoZHS0S30vvH4/edit?usp=sharing
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

4. **Set up your API Key:**
   This project uses The Guardian's API to collect news articles.
   - Get a free developer API key from [The Guardian Open Platform](https://open-platform.theguardian.com/access/).
   - Create a file named `.env` in the root directory of the project.
   - Add your API key to the `.env` file like this:
     ```
     GUARDIAN_API_KEY='your-api-key-here'
     ```

## Usage

To replicate the findings, run the scripts from your terminal in the following order:

1.  **`DataCollection.py`**: Collects raw news articles.
    ```bash
    python DataCollection.py
    ```
2.  **`Make_features.py`**: Processes raw data and creates features.
    ```bash
    python Make_features.py
    ```
3.  **`Aggregrate.py`**: Aggregates features into weekly data.
    ```bash
    python Aggregrate.py
    ```
4.  **`Model.py`**: Trains the model and generates results.
    ```bash
    python Model.py
    ```

The `work.ipynb` notebook is also available for a more interactive exploration of the data and models.

## Project Structure

- **`DataCollection.py`**: Script to collect raw news articles.
- **`Make_features.py`**: Script for feature engineering from raw text.
- **`Aggregrate.py`**: Script to aggregate features on a weekly basis.
- **`Model.py`**: Script for training the model and evaluating performance.
- **`work.ipynb`**: Jupyter Notebook for analysis, modeling, and visualization.
- **`events.csv`**: The main file containing the list of events to analyze.
- **`final_model_data.csv`**: The final dataset used for model training.


## Results

The project generates several visualizations to help understand the model's performance, including:

- **Confusion Matrix**: To evaluate the accuracy of the classification model.
- **Classification Report**: Provides precision, recall, and F1-score for the model.
- **Feature Importance**: Shows which features are most influential in predicting outcomes.

These visualizations can be found in the visuals directory of the project.
