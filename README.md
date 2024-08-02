# Intelligent Recipe Recommendation System

**Author:** Varsha Rajawat  
**Course:** CSD-4523 Python Programming  
**Final-term Exam Project**  
**Date:** 18 July 2024

## Overview

The Intelligent Recipe Recommendation System is designed to recommend recipes based on the ingredients provided by the user. It uses TF-IDF vectorization and cosine similarity to match recipes with the user's input.

## Features

- Load and preprocess recipe data from a CSV file
- Normalize and vectorize ingredients using TF-IDF
- Recommend recipes based on user input
- Display recipe details including name, description, ingredients, and cooking steps

## Prerequisites

To run this project, you'll need to have the following Python packages installed:

- `pandas`
- `scikit-learn`
- `streamlit`

These can be installed using the provided `requirements.txt` file.

## Installation

1. Clone the repository

2. Navigate to the project directory:

3. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your recipe CSV file (e.g., `RAW_recipes.csv`) in the project directory.

2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3. Open the provided URL in your browser to use the application.

## File Structure

- `app.py`: Main application file containing the Streamlit code.
- `RAW_recipes.csv`: CSV file containing the recipes data (ensure this file is formatted correctly).
- `requirements.txt`: File listing the project dependencies.

## Acknowledgments

- Thanks to the creators of the libraries and tools used in this project.
