# 1.Title and Author

## Project Title
Personalized Recipe Recommender System

## Prepared for
UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang

## Author Name
Mounika Reddy Kummetha

## Profiles and Presentations
- **GitHub Profile:** https://github.com/mounikakummetha

## 2. Background

### What is the Project about?

The core objective of this project is to develop a **Personalized Recipe Recommender System** using data-driven techniques and machine learning models. This system is designed to suggest recipes to users based on their unique interactions, preferences, and dietary needs. Key aspects of user data being analyzed include:

- **User Preferences and Dietary Restrictions (`PP_users.csv`):** Understanding individual user preferences, dietary restrictions, and cooking techniques to tailor recipe recommendations.
- **User Interactions (`RAW_interactions.csv`, `interactions_train.csv`, `interactions_test.csv`, `interactions_validation.csv`):** Analyzing user engagement with different recipes, including ratings, reviews, and frequency of interactions, to understand user satisfaction and preferences.
- **Recipe Characteristics (`RAW_recipes.csv`, `PP_recipes.csv`):** Utilizing detailed recipe information, such as ingredients, nutritional values, and preparation steps, to match recipes with user profiles and preferences.
- **Ingredient Standardization (`ingr_map.pkl`):** Ensuring consistency in ingredient naming and usage across recipes, aiding in accurate recipe recommendation.

### Why does it matter?

- **Enhanced User Experience:** Leveraging actual user interaction data minimizes reliance on guesswork, providing a more accurate and personalized user experience.
- **Dietary Management and Diversity:** The system facilitates dietary management by recommending recipes that align with users' dietary restrictions and preferences, promoting healthier and more diverse dietary choices.
- **Data-Driven Culinary Insights:** Offers valuable insights into culinary trends and user behavior, allowing for a deeper understanding of user preferences and cooking habits.

### What are your research questions?

1. How can we effectively utilize user data, including preferences, interactions, and dietary profiles, to recommend recipes that precisely cater to individual tastes and dietary needs?
2. Which machine learning models and algorithms are best suited for personalizing recipe recommendations based on user data?
3. Can the system identify and recommend underexplored or new recipes that align with users' tastes and dietary preferences?
4. Which factors (ingredients, nutrition, preparation time) most significantly influence user preferences and recipe ratings?


# 3. About Dataset

The dataset collection for this project is focused on recipe and user interaction data, sourced from "Food.com Recipes and User Interactions" on Kaggle. It features comprehensive information about recipes, user profiles, and interactions, aiming to facilitate the development of a personalized recipe recommender system.

- **Data sources:** [Food.com Recipes and User Interactions on Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- **Data size:** ( The combined size of all datasets, "104.59 MB")
- **Data shape:** The collective datasets contain over (total number of rows across all datasets) rows and vary in the number of columns:
  - `RAW_recipes.csv` - 231637 rows, 12 columns
  - `RAW_interactions.csv` - 1132367 rows, 5 columns
  - `PP_recipes.csv` - 178265rows, 8 columns
  - `PP_users.csv` - 25076 rows, 6 columns
  - `interactions_train.csv`, `interactions_test.csv`, `interactions_validation.csv` - 23176 rows, 5 columns each
  - `ingr_map.pkl` - (11659 rows), 3 columns
- **Time period:** The datasets collectively cover a time span of (from 01/02/2010 to 01/02/2020), representing a comprehensive period of culinary trends and user interactions.
- **What does each row represent?**
  - `RAW_recipes.csv` and `PP_recipes.csv`: Each row represents a unique recipe with detailed information.
  - `RAW_interactions.csv`, `interactions_train.csv`, `interactions_test.csv`, `interactions_validation.csv`: Each row signifies a user interaction with a recipe, such as rating or reviewing.
  - `PP_users.csv`: Each row depicts a user profile, including their preferences and dietary information.
  - `ingr_map.pkl`: Each row corresponds to a specific ingredient and its standardized form.

### Data Dictionary for Recipe Recommender System Dataset

#### I. RAW_recipes.csv
- **Purpose:** Provides comprehensive information about each recipe, including ingredients, cooking steps, and nutritional details.
- **Columns:**
  - **I. recipe_id**
    - **Dtype:** Categorical (Integer)
    - **Definition:** Unique identifier for each recipe.
    - **Potential Values:** Range from 38 to 537716
  - **II. name**
    - **Dtype:** Categorical (String)
    - **Definition:** The title of the recipe.
    - **Potential Values:** 231637 unique values
  - **III. minutes**
    - **Dtype:** Numerical (Integer)
    - **Definition:** Time required to prepare and cook the recipe.
    - **Potential Values:** Range from 0 to 2147483647
  - **IV. contributor_id**
    - **Dtype:** Categorical (Integer)
    - **Definition:** ID of the user who contributed the recipe.
    - **Potential Values:** Range from 1530 to 2002312794
  - **V. submitted**
    - **Dtype:** Categorical (Date)
    - **Definition:** The date when the recipe was submitted.
    - **Potential Values:** Dates ranging from 1999 to 2018
  - **VI. tags**
    - **Dtype:** Categorical (String)
    - **Definition:** Categorizing tags for the recipe (e.g., "vegan").
    - **Potential Values:** 13124 unique values
  - **VII. nutrition**
    - **Dtype:** Categorical (String)
    - **Definition:** Nutritional information of the recipe.
    - **Potential Values:** String representation of nutritional facts
  - **VIII. n_steps**
    - **Dtype:** Numerical (Integer)
    - **Definition:** Number of steps in the recipe.
    - **Potential Values:** Range from 0 to 145
  - **IX. steps**
    - **Dtype:** Categorical (String)
    - **Definition:** Detailed cooking instructions.
    - **Potential Values:** String containing cooking steps
  - **X. description**
    - **Dtype:** Categorical (String)
    - **Definition:** A brief description of the recipe.
    - **Potential Values:** Textual descriptions, varying in length
  - **XI. ingredients**
    - **Dtype:** Categorical (String)
    - **Definition:** List of ingredients used in the recipe.
    - **Potential Values:** Array of ingredients names
  - **XII. n_ingredients**
    - **Dtype:** Numerical (Integer)
    - **Definition:** Number of ingredients in the recipe.
    - **Potential Values:** Range from 1 to 43

#### II. RAW_interactions.csv
- **Purpose:** Captures user interactions with recipes, providing insights into user preferences and behavior.
- **Columns:**
  - **I. user_id**
    - **Dtype:** Categorical (Integer)
    - **Definition:** Unique identifier for each user.
    - **Potential Values:** Range specific to dataset
  - **II. recipe_id**
    - **Dtype:** Categorical (Integer)
    - **Definition:** Identifier for the interacted recipe.
    - **Potential Values:** Corresponds to recipe_id in RAW_recipes.csv
  - **III. date**
    - **Dtype:** Categorical (Date)
    - **Definition:** The date of the interaction.
    - **Potential Values:** Date range specific to dataset
  - **IV. rating**
    - **Dtype:** Numerical (Integer)
    - **Definition:** User-given rating to the recipe.
    - **Potential Values:** Typically 1 to 5
  - **V. review**
    - **Dtype:** Categorical (String)
    - **Definition:** User's review or comment on the recipe.
    - **Potential Values:** Textual content, varies in length

#### III. PP_recipes.csv
- **Purpose:** Contains preprocessed recipe data, optimized for analysis and model training.
- **Columns:** (Adjusted from RAW_recipes.csv for preprocessing)
  - **I. recipe_id**
    - **Dtype:** Categorical (Integer)
    - **Definition:** Unique identifier for each preprocessed recipe.
    - **Potential Values:** Range specific to dataset
  - **II. name**
    - **Dtype:** Categorical (String)
    - **Definition:** The title of the recipe after preprocessing.
    - **Potential Values:** Number of unique values may vary after preprocessing
  - **III. minutes**
    - **Dtype:** Numerical (Integer)
    - **Definition:** Time required to prepare and cook the recipe, adjusted if necessary during preprocessing.
    - **Potential Values:** Adjusted range based on preprocessing

#### IV. PP_users.csv
- **Purpose:** Includes user profile data, essential for understanding user preferences and dietary restrictions.
- **Columns:**
  - **I. user_id**
    - **Dtype:** Categorical (Integer)
    - **Definition:** Unique identifier for each user.
    - **Potential Values:** Range specific to dataset
  - **II. techniques**
    - **Dtype:** Categorical (String)
    - **Definition:** Cooking techniques used or preferred by the user.
    - **Potential Values:** Specific techniques listed
  - **III. items**
    - **Dtype:** Categorical (String)
    - **Definition:** Ingredients or items used or preferred by the user.
    - **Potential Values:** Array of item names
  - **IV. n_items**
    - **Dtype:** Numerical (Integer)
    - **Definition:** Number of items associated with the user.
    - **Potential Values:** Range specific to dataset
  - **V. ratings_count**
    - **Dtype:** Numerical (Integer)
    - **Definition:** Total count of ratings given by the user.
    - **Potential Values:** Range specific to dataset
  - **VI. reviews_count**
    - **Dtype:** Numerical (Integer)
    - **Definition:** Total count of reviews written by the user.
    - **Potential Values:** Range specific to dataset

#### V. interactions_train.csv, interactions_test.csv, interactions_validation.csv
- **Purpose:** These datasets are segmented for model training, testing, and validation, containing user interactions for different phases.
- **Columns:** (Similar to RAW_interactions.csv)
  - **I. user_id**
    - **Dtype:** Categorical (Integer)
    - **Definition:** Unique identifier for each user, consistent across training, testing, and validation sets.
    - **Potential Values:** Range specific to dataset
  - **II. recipe_id**
    - **Dtype:** Categorical (Integer)
    - **Definition:** Identifier for the interacted recipe, corresponding to preprocessed recipes in PP_recipes.csv.
    - **Potential Values:** Range specific to dataset
  - **III. date**
    - **Dtype:** Categorical (Date)
    - **Definition:** The date of the interaction, relevant for the specific phase (training, testing, or validation).
    - **Potential Values:** Date range specific to each dataset
  - **IV. rating**
    - **Dtype:** Numerical (Integer)
    - **Definition:** User-given rating to the recipe, used as a target variable in model training and evaluation.
    - **Potential Values:** Typically 1 to 5
  - **V. review**
    - **Dtype:** Categorical (String)
    - **Definition:** User's review or comment on the recipe, potentially used for sentiment analysis or additional features.
    - **Potential Values:** Textual content, varies in length

#### VI. ingr_map.pkl
- **Purpose:** Aids in standardizing ingredients across the datasets, ensuring data uniformity.
- **Columns:**
  - **I. ingredient_id**
    - **Dtype:** Categorical (Integer)
    - **Definition:** Unique identifier for each ingredient.
    - **Potential Values:** Range specific to dataset
  - **II. ingredient_name**
    - **Dtype:** Categorical (String)
    - **Definition:** Name of the ingredient.
    - **Potential Values:** Specific ingredient names
  - **III. mapped_ingredient**
    - **Dtype:** Categorical (String)
    - **Definition:** Standardized form of the ingredient name.
    - **Potential Values:** Standardized ingredient names

# Target/Label in ML Model
The primary target for the machine learning models in this project is the user ratings from the interactions datasets. These ratings provide a quantitative measure of user preferences and satisfaction, making them a crucial factor in developing an effective recipe recommendation system.

# Features/Predictors for ML Models
Key features and predictors that will be used in the machine learning models for this project include:

- **Ingredients:** The ingredients listed in each recipe, which are essential to match recipes with user preferences and dietary restrictions.
- **Recipe Categories:** Categories or tags associated with each recipe (such as cuisine type, meal type), which help in categorizing and recommending recipes based on user interests.
- **User Dietary Restrictions:** Information from user profiles indicating any dietary restrictions or preferences, ensuring recommended recipes are suitable for individual users.
- **Historical Interaction Data:** Past user interactions including ratings, reviews, and other forms of engagement with recipes. This data helps in understanding user preferences and improving the accuracy of the recommendation system.




