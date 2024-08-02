"""
Intelligent Recipe Recommendation System

Author: Varsha Rajawat
Course: CSD-4523 Python Programming
Final-term Exam Project
Date: 18 July 2024
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
import ast

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    
    Parameters:
    - file_path (str): The path to the CSV file containing the recipes data.

    Returns:
    - recipes_df (DataFrame): The preprocessed recipes DataFrame.
    """
    # Load the dataset
    recipes_df = pd.read_csv(file_path)
    
    # Remove null or duplicate entries
    recipes_df.dropna(inplace=True)
    recipes_df.drop_duplicates(inplace=True)
    
    def normalize_ingredients(ingredients):
        """
        Normalize the ingredients by converting to lowercase and removing non-alphanumeric characters.
        
        Parameters:
        - ingredients (str): A string representation of a list of ingredients.

        Returns:
        - list: A list of normalized ingredients.
        """
        return [re.sub(r'[^\w\s]', '', ingredient.lower()) for ingredient in eval(ingredients)]
    
    # Apply normalization to the 'ingredients' column
    recipes_df['ingredients'] = recipes_df['ingredients'].apply(normalize_ingredients)
    
    # Join the list of ingredients into a single string
    recipes_df['combined_ingredients'] = recipes_df['ingredients'].apply(lambda x: ' '.join(x))
    
    return recipes_df

def vectorize_data(recipes_df):
    """
    Vectorize the combined ingredients using TF-IDF.

    Parameters:
    - recipes_df (DataFrame): The preprocessed recipes DataFrame.

    Returns:
    - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    - tfidf_matrix (sparse matrix): The TF-IDF matrix for the combined ingredients.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(recipes_df['combined_ingredients'])
    return vectorizer, tfidf_matrix

def get_recommendations(user_input, vectorizer, tfidf_matrix):
    """
    Get cosine similarity scores between user input and the recipes.

    Parameters:
    - user_input (str): The user's input ingredients.
    - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    - tfidf_matrix (sparse matrix): The TF-IDF matrix for the combined ingredients.

    Returns:
    - cosine_similarities (array): The cosine similarity scores.
    """
    user_input = re.sub(r'[^\w\s]', '', user_input.lower())
    user_tfidf = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    return cosine_similarities

def recommend_recipes(user_input, recipes_df, vectorizer, tfidf_matrix, top_n=5):
    """
    Recommend top N recipes based on the user's input.

    Parameters:
    - user_input (str): The user's input ingredients.
    - recipes_df (DataFrame): The preprocessed recipes DataFrame.
    - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    - tfidf_matrix (sparse matrix): The TF-IDF matrix for the combined ingredients.
    - top_n (int): The number of top recipes to recommend (default is 5).

    Returns:
    - similar_items (list): A list of tuples containing recommended recipes and their similarity scores.
    """
    cosine_similarities = get_recommendations(user_input, vectorizer, tfidf_matrix)
    similar_indices = cosine_similarities[0].argsort()[:-top_n-1:-1]
    similar_items = [(recipes_df.iloc[i], cosine_similarities[0][i]) for i in similar_indices]
    return similar_items

def normalize_steps(steps):
    """
    Normalize the steps by converting the string representation of a list to an actual list.

    Parameters:
    - steps (str): A string representation of a list of steps.

    Returns:
    - list or None: A list of steps or None if the input is not in the expected format.
    """
    try:
        if isinstance(steps, str):
            steps = ast.literal_eval(steps)
        if isinstance(steps, list):
            return steps
        else:
            return None
    except (ValueError, SyntaxError):
        return None

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title('Intelligent Recipe Recommendation System :knife_fork_plate:')
    st.write('Enter ingredients you have (comma separated) to get recipe recommendations:')
    
    user_input = st.text_input('Ingredients:')
    
    if st.button('Recommend Recipes'):
        
        # Add loading screen
        loading_screen = st.image('./cooking.gif', caption="Picking out deliciousness...")
    
        # Load and preprocess data
        recipes_df = load_and_preprocess_data('./RAW_recipes.csv')
        
        # Vectorize data
        vectorizer, tfidf_matrix = vectorize_data(recipes_df)
        
        # Get recommendations
        recommendations = recommend_recipes(user_input, recipes_df, vectorizer, tfidf_matrix)
        
        # Remove loading screen
        loading_screen.empty()
        
        # Display recommendations
        st.write('Recommended Recipes:')
        for index, recipe in enumerate(recommendations, start=1):
            recipe_info = recipe[0]
            st.write(recipe_info)
            st.subheader(f"Recipe {index}: {recipe_info['name']}")
            st.write(f"**Description:** {recipe_info['description']}")
            st.write(f"**Ingredients:** {', '.join(recipe_info['ingredients'])}")
            st.write(f"**Cooking Time:** {recipe_info['minutes']} minutes")
            steps = recipe_info.get('steps')
            normalized_steps = normalize_steps(steps)

            st.write("**Steps:**")
            if normalized_steps is None:
                st.write("Steps data is not in the expected format.")
            else:
                for i, step in enumerate(normalized_steps, start=1):
                    st.write(f"{i}. {step}")

if __name__ == '__main__':
    main()
