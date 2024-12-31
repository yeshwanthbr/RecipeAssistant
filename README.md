# RecipeAssist
A personalized chatbot that recommends recipes based on your dietary preferences, ingredients, and cuisine choices integrated with LLM API

# Features
- Personalized Recipe Recommendations: Suggests recipes tailored to user preferences such as dietary needs, cuisine types, and nutritional goals.

- Dietary Preference Parsing: Understands specific dietary requirements like Vegan, Vegetarian, Keto, Gluten-Free, or No Preference.

- Nutritional Analysis: Filters recipes based on macronutrient levels (Carb, Protein, Fat) categorized as High, Medium, or Low.

- Time-Conscious Options: Recommends recipes based on prep time preferences for users seeking quick meals.

- Interactive Clarifications: Asks follow-up questions to capture missing details and refine recommendations.

- Content Moderation: Includes input moderation to ensure safe and relevant interactions.

- Recipe Filtering: Validates and recommends recipes from a database, prioritizing highly rated options.

- Flexible Query Handling: Handles vague or open-ended user inputs with clarifying questions and step-by-step refinements.

- Preloaded Catalog: Summarizes available recipes in a catalog format with key details like nutritional values and prep times.

# Tech Stack
- OpenAI API 
- Python
- Pandas, Flask (for a simple web app to be able to communicate with the chatbot)

# Overview and Design and Evaluations
See documentation [here](RecipeAssist_Design.pdf)

# To run the app
### Add your OpenAI API key to the file OpenAI_API_Key.txt
### Command to run Example with Function Calling

```
FLASK_APP=app.py flask run 
```

### Command to run Example without Function Calling

```
FLASK_APP=app_without_fc.py flask run

```

### Start the server and access the chat app from http://127.0.0.1:5000