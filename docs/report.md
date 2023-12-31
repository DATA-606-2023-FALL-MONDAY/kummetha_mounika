## 1. Title and Author
- **Personalized Recipe Recommender System**
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- Author Name : [Mounika Reddy Kummetha]()
- [Github](https://github.com/DATA-606-2023-FALL-MONDAY/kummetha_mounika)
- [Linkedin](https://www.linkedin.com/in/mounika-kummetha-546212211?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)
- [Presentation](https://docs.google.com/presentation/d/11z2MgFdl-M1OcHvvTpfyTbQWz0W1dVgsHqkfJqaES3Q/edit#slide=id.p)
- [Presentation Video Link](https://youtu.be/Ci1H7cPZtIw) 


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


# 4. Exploratory Data Analysis (EDA) - Recipe Dataset

This document presents a detailed Exploratory Data Analysis (EDA) on a dataset related to recipes. The EDA aims to uncover insights that can aid in enhancing a recipe recommendation system.

## Data Cleaning and Preprocessing

### Handling Missing Values
- Analysis of missing data in the dataset and steps taken to address these.

### Data Type Conversion
- Conversion of specific columns to more appropriate data types for effective analysis.

## Descriptive Statistics

### Summary Statistics
- Basic statistical summaries (mean, median, mode, etc.) of key columns in the dataset.

### Distribution Analysis
- Examination of the distribution of crucial variables like recipe ratings and preparation time.

![Distribution Analysis](https://github.com/mounikakummetha/images/blob/main/distribution_analysis.png)

## Data Visualization

### Boxplots
- Visualization of distributions and identification of outliers, especially in user ratings.

![Box plot](https://github.com/mounikakummetha/images/blob/main/box.plot.user.ratings.png)

### Time Series Analysis
- Analysis of user interactions over time to identify trends in recipe popularity.

![Time Series Analysis](https://github.com/mounikakummetha/images/blob/main/Time_series.png)

### Bar Charts
- Displaying top common ingredient replacements and the top 20 ingredients to gain insights into popular choices.

![Bar Chart - Ingredient Analysis](https://github.com/mounikakummetha/images/blob/main/Bar_chart_top_ingredients.png)

## Correlation Analysis

### Heatmaps
- Correlation heatmap to understand the relationships between different features.

![Correlation Heatmap](https://github.com/mounikakummetha/images/blob/main/heat_map.png)

## Insights for Recipe Recommendation System

### User Ratings
- Analysis suggests leveraging high user ratings to recommend popular recipes.

### Seasonal Trends
- Identifying popular recipes during specific times of the year for targeted recommendations.

### Ingredient Analysis
- Utilizing common ingredient replacements and popular ingredients for recipe suggestions.

### User Engagement Features
- Determining features that predict user engagement, such as interactions and average user rating.

## Conclusion

This EDA provides valuable insights into user behavior and preferences in relation to recipes. These findings will significantly contribute to the development of a personalized and efficient recipe recommendation system.

# 5. Model Training Process

### Step 1: Data Preparation
The initial step involved preparing the dataset for the modeling process. This included:

- **Loading the Data**: The dataset was loaded into a pandas DataFrame.
- **Data Cleaning**: Irrelevant columns were removed, and missing values were handled.
- **Data Preprocessing**: The data was formatted and preprocessed to ensure compatibility with the modeling techniques.

```python
# Example code for data preparation
df = pd.read_csv('data.csv')
df.dropna(inplace=True)
df.drop(['irrelevant_column'], axis=1, inplace=True)
```

### Step 2: Feature Selection and Splitting the Dataset
After cleaning, the next step was to select relevant features and split the data into training and testing sets.

- **Feature Selection**: Features relevant to predicting recipe ratings were identified and isolated.
- **Splitting Data**: The dataset was divided into a training set and a testing set.

from sklearn.model_selection import train_test_split

X = df.drop('rating', axis=1)
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

### Step 3: Model Training
The RandomForestClassifier model was selected for its proficiency with tabular data. The training process involved:

Model Instantiation: Creating an instance of the RandomForestClassifier.
Model Fitting: Fitting the model on the training dataset.

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

### Step 4: Model Evaluation
Finally, the model's performance was evaluated using the testing set.

Making Predictions: Using the model to predict ratings on the testing set.
Evaluating Accuracy: Assessing the model's accuracy to understand its effectiveness.

from sklearn.metrics import accuracy_score

y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

## Analysis and Improvements
The RandomForestClassifier achieved an accuracy of 75.71%, indicating a reasonably good predictive ability. Future improvements may include advanced parameter tuning, feature engineering, and exploring other machine learning algorithms to enhance prediction accuracy.

# 5. Conclusion

The development of our Recipe Recommendation System, underpinned by meticulous exploratory data analysis and model training, has yielded profound insights into user behaviors and recipe characteristics, vital for crafting personalized culinary experiences. Analysis of user interaction data revealed a predominance of casual users and a preference for highly-rated recipes, suggesting a strategic focus on engaging these users with tailored, high-quality recommendations. The time series analysis highlighted seasonal trends, offering opportunities for timely recipe suggestions. On the technical side, complex patterns in the data, identified through scatter plots, correlation matrices, and heatmaps, emphasized the necessity for advanced machine learning models and nuanced feature engineering. This is particularly evident in the diverse relationships among key features like preparation time and number of ingredients. The project's findings, from user engagement trends to ingredient preferences, are instrumental in refining the recommendation system, ensuring it not only aligns with current user preferences but also remains adaptable to changing culinary trends and tastes.

# 6 . References

### References

1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *Computer*, 42(8), 30-37. [https://doi.org/10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263)

2. Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer International Publishing. [https://doi.org/10.1007/978-3-319-29659-3](https://doi.org/10.1007/978-3-319-29659-3)

3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer Series in Statistics. Springer New York. [https://doi.org/10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)

4. Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.

5. Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis* (2nd ed.). Springer International Publishing. [https://doi.org/10.1007/978-3-319-24277-4](https://doi.org/10.1007/978-3-319-24277-4)

6. Chollet, F. (2018). *Deep Learning with Python*. Manning Publications.

7. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [http://www.deeplearningbook.org](http://www.deeplearningbook.org)
