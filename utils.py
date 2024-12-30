import openai
import json
import os
import re
import ast
import pandas as pd  # Importing the pandas library for data manipulation

# Path to the recipes.json file
RECIPES_FILE_PATH = os.path.join(os.path.dirname(__file__), 'recipes.json')
MODEL = "gpt-4o-mini"


# Initialize conversation with OpenAI model
def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"
    example_user_req = "Recommend a recipe with high protein content, low carb, and a maximum prep time of 30 minutes."

    system_message = f"""

    You are an intelligent cuisine and recipe expert and your goal is to find the best recipe for a user.
    You need to ask relevant questions and understand the user mood and requirements by analysing the user's responses.
    You final objective is to find the values for the different keys (dietaryPreference, cuisine, prepTimeMinutes, Carb, Protein, Fat, spiceLevel) in the json format below.
    The values for these keys determine the users profile
    Values for keys 'Carb', 'Protein', 'Fat', 'spiceLevel' should be High, Medium, Low based on the importance of the corresponding keys, as stated by user.
    The values for the key 'dietaryPreference' should be one or more of the following: 'Vegan', 'Vegetarian', 'Gluten-Free', 'Keto', 'NoPreference'
    The values for key 'cuisine' should be one of the following: 'Italian', 'Mexican', 'Indian', 'Chinese', 'Thai', 'Mediterranean', 'Japanese', 'American', 'French', 'Greek', 'Spanish', 'Middle-Eastern', 'Korean', 'Vietnamese', 'British', 'Caribbean', 'German', 'African', 'Latin-American', 'Scandinavian', 'Eastern-European', 'Australian', 'Canadian', 'NoPreference'
    The value for 'prepTimeMinutes' should be a numerical value extracted from the user's response.
    The values currently in the string provided are only representative values.

    {delimiter}Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised.
    - The values for keys 'Carb', 'Protein', 'Fat', 'spiceLevel', should strictly be either 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
    - The value for 'dietaryPreference' and 'cuisine' should be from the list I have shared and should be set to 'NoPreference' if the user does not have a preference or is open on any value.
    - The value for 'prepTimeMinutes' should be a numerical value extracted from the user's response.
    - Do not randomly assign values to any of the keys. The values need to be inferred from the user's response.
    {delimiter}

    To fill the values in the string, you need to have the following chain of thoughts:
    {delimiter} Thought 1: Ask a question to understand the user's mood and requirements. \n
    If their primary use for the recipe is unclear. Ask another question to comprehend their needs.
    You are trying to fill the values of all the keys.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys.
    Answer "Yes" or "No" to indicate if you understand the requirements and have updated the values for the relevant keys. \n
    If yes, proceed to the next step. Otherwise, rephrase the question to capture their profile. \n{delimiter}

    {delimiter}Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step.
    Remember the instructions around the values for the different keys. Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    Answer "Yes" or "No" to indicate if you understood all the values for the keys and are confident about the same.
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.{delimiter}

    {delimiter}Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary.
    If you are not confident about any of the values, ask clarifying questions. {delimiter}

    Follow the above chain of thoughts and only output the final updated python dictionary. \n

    {delimiter}
    User: "Hi, I am looking for healthy meal options."
    Assistant: "That's great! Are you following any specific dietary preferences, such as vegan, gluten-free, or keto? Knowing your preferences will help me recommend meals that best suit your needs."
    User: "I'm vegan and also prefer gluten-free meals."
    Assistant: "Thanks for sharing! Based on your dietary preferences, I have a few suggestions for you. For instance, the Vegan Tacos made with corn tortillas, black beans, and avocado might be perfect. It's light and healthy with a medium level of carbs and protein. Another option could be the Vegetable Soup, a comforting dish with a variety of vegetables, all while being both vegan and gluten-free. Would you like me to suggest more recipes along these lines?"
    User: "Yes, I love soups! Can you recommend something hearty?"
    Assistant: "Sure! You might enjoy the Sweet Potato & Black Bean Chili. It's hearty, vegan, and gluten-free, with a good balance of carbs, protein, and fat. It's perfect for a warming meal. If you prefer something lighter, the Quinoa Salad with Roasted Vegetables is a refreshing choice that fits your preferences. Would you like more ideas like these?"
    User: "I’m also looking for meals that are low on spice."
    Assistant: "Got it! Based on your preference for a milder spice level, I'd recommend the Vegetable Soup, which has a low spice level. The Chicken Caesar Salad is also an option, though it’s not vegan, it’s very mild with no spice. I also suggest the Grilled Salmon with Asparagus, which is mild and packed with protein. Does this sound like what you're looking for?"
    User: "Yes, that sounds good. What about something quick?"
    Assistant: "If you need something quick, the Vegan Tacos would be perfect. It's ready in just 20 minutes and offers a good balance of nutrition. Similarly, the Chicken Tacos takes only 20 minutes to prepare and is high in protein. Let me know if any of these stand out to you or if you'd like more details!"
    User: "The Vegan Tacos seem like a great choice!"
    Assistant: "I'm glad you like the sound of the Vegan Tacos! It's a simple yet satisfying meal with a nice mix of flavors. If you decide to try it out, let me know how it turns out! Would you like more recipe recommendations, or is there anything else I can assist you with?"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements.
    """
    conversation = [{"role": "system", "content": system_message}]
    # conversation = system_message
    return conversation

# Function to get chat model completions (communication with OpenAI)
def get_chat_model_completions(messages):
    
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    return response.choices[0].message.content

# Moderation check for user input (content moderation)
def moderation_check(user_input):
    response = openai.moderations.create(input=user_input)
    return response.results[0].flagged

# Function to confirm user intent for recipe preferences
def intent_confirmation_layer(response_assistant):
    delimiter = "####"
    prompt = f"""
    You are a senior cuisines and recipe expert who has an eye for detail.
    You are provided an string input. You need to see that in the string the values for following 
    dietaryPreference, cuisine, prepTimeMinutes, Carb, Protein, Fat, spiceLevel
    have been captured successfully. Return Yes Or No

    Values for keys 'Carb', 'Protein', 'Fat', 'spiceLevel' should be High, Medium, Low, NoPreference based on the importance of the corresponding keys, as stated by user.
    The values for the key 'dietaryPreference' should be one or more of the following: 'Vegan', 'Vegetarian', 'Gluten-Free', 'Keto', 'NoPreference'
    The values for key 'cuisine' should be one of the following: 'Italian', 'Mexican', 'Indian', 'Chinese', 'Thai', 'Mediterranean', 'Japanese', 'American', 'French', 'Greek', 'Spanish', 'Middle-Eastern', 'Korean', 'Vietnamese', 'British', 'Caribbean', 'German', 'African', 'Latin-American', 'Scandinavian', 'Eastern-European', 'Australian', 'Canadian', 'NoPreference'
    The value for 'prepTimeMinutes' should be a numerical value extracted from the user's response.

    Remember return No if any one of the values is not captured

    """
    messages=[{"role": "system", "content":prompt },{"role": "user", "content":f"""Here is the input: {response_assistant}""" }]
    confirmation = openai.chat.completions.create(
                                    model=MODEL,
                                    messages = messages)

    return confirmation.choices[0].message.content

def dictionary_present(response):
    delimiter = "####"

    user_req =   {
        "dietaryPreference": ["Vegan", "Gluten-Free"],
        "cuisine": "Mediterranean",
        "prepTimeMinutes": 30,
        "Carb": "High",
        "Protein": "Medium",
        "Fat": "Medium",
        "spiceLevel": "Low"
  }

    prompt = f"""You are a python expert. You are provided an input.
            You have to check if there is a python dictionary present in the string.
            It will have the following format {user_req}.
            You final objective is to find the values for the different keys ('GPU intensity','Display quality','Portability','Multitasking','Processing speed','Budget') in the final string and be confident of the values.
            The output should match the format as {user_req}.

            {delimiter}
            Make sure that the value of prepTimeMinutes is also present in the user input. ###
            The output should contain the exact keys and values as present in the input.
            Ensure the keys and values are in the given format:
            {{
            'GPU intensity': 'low/medium/high ',
            'Display quality':'low/medium/high',
            'Portability':'low/medium/high',
            'Multitasking':'low/medium/high',
            'Processing speed':'low/medium/high',
            'Budget':'numerical value',
            
            'dietaryPreference': list of dietary preferences,
            'cuisine': one of the cuisines,
            'prepTimeMinutes': 'numerical value',
            'Carb': 'Low/Medium/High',
            'Protein': 'Low/Medium/High',
            'Fat': 'Low/Medium/High',
            'spiceLevel': 'Low/Medium/High'
            }}
            dietary preferences possible values: 'Vegan', 'Vegetarian', 'Gluten-Free', 'Keto', 'NoPreference' where NoPreference means the user is open to any value.
            cuisine posible values: 'Italian', 'Mexican', 'Indian', 'Chinese', 'Thai', 'Mediterranean', 'Japanese', 'American', 'French', 'Greek', 'Spanish', 'Middle-Eastern', 'Korean', 'Vietnamese', 'British', 'Caribbean', 'German', 'African', 'Latin-American', 'Scandinavian', 'Eastern-European', 'Australian', 'Canadian', 'NoPreference' where NOPreference means the user is open to any value.
            Here are some sample input output pairs for better understanding:
            {delimiter}
            input 1: - Dieterary Preference: keto - cuisine: Indian - Carb: Low - Protein: High - Fat: medium - spiceLevel: Medium - prepTimeMinutes: 30
            output 1: {{"dietaryPreference": ["keto"], "cuisine": "Indian", "Carb": "Low", "Protein": "High", "Fat": "Medium", "spiceLevel": "Medium", "prepTimeMinutes": 30}}

            input 2: - Dieterary Preference: NoPreference - cuisine: Mexican - Carb: Low - Protein: High - Fat: medium - spiceLevel: Medium - prepTimeMinutes: 30
            output 2: {{"dietaryPreference": ["NoPreference"], "cuisine": "Mexican", "Carb": "Low", "Protein": "High", "Fat": "Medium", "spiceLevel": "Medium", "prepTimeMinutes": 30}}

            input 3: Here is your requirements 'Dieterary Preference': 'Vegan','cuisine': 'Italina','Carb': 'Medium','Protein': 'High','Fat': 'Low','spiceLevel': 'Medium','prepTimeMinutes': 20
            output 3: {{"dietaryPreference": ["Vegan"], "cuisine": "Italian", "Carb": "Medium", "Protein": "High", "Fat": "Low", "spiceLevel": "Medium", "prepTimeMinutes": 20}}
            {delimiter}

            Do not include any other information in the output. Just output the pure string representation of the json object. Do not add any additional information in the output.
            """
    messages = [{"role": "system", "content":prompt },
                {"role": "user", "content":f"""Here is the user input: {response}""" }]

    confirmation = get_chat_model_completions(messages)

    return confirmation

def extract_dictionary_from_string(string):
    regex_pattern = r"\{[^{}]+\}"
    dictionary_matches = re.findall(regex_pattern, string)
    # Extract the first dictionary match and convert it to lowercase
    if dictionary_matches:
        dictionary_string = dictionary_matches[0]
        # Convert the dictionary string to a dictionary object using ast.literal_eval()
        dictionary = ast.literal_eval(dictionary_string)
    return dictionary

# Filter recipes based on user preferences
def get_filtered_recipes(user_preferences):
    print('callingggggg', user_preferences)
    filtered_recipes = pd.read_csv('recipes.csv')
    print(user_preferences)
    # Filter based on cuisine and prep time
    filtered_recipes = filtered_recipes[(filtered_recipes['prepTimeMinutes'] <= user_preferences['prepTimeMinutes']) & (filtered_recipes['cuisine'] == user_preferences['cuisine'])]
    filtered_recipes['score'] = 0
    # Filter recipes based on dietary preferences
    filtered_recipes['score'] += filtered_recipes['dietaryPreference'].apply(lambda x: 1 if 'NoPreference' in user_preferences['dietaryPreference'] or all(diet in x for diet in user_preferences['dietaryPreference']) else 0)
    # Filter based on Carb, Protein, Fat, spiceLevel
    filtered_recipes['score'] += filtered_recipes['Carb'].apply(lambda x: 1 if 'NoPreference' in user_preferences['Carb'] or user_preferences['Carb'].lower() != x.lower() else 0)
    filtered_recipes['score'] += filtered_recipes['Protein'].apply(lambda x: 1 if 'NoPreference' in user_preferences['Protein'] or user_preferences['Protein'].lower() != x.lower() else 0)
    filtered_recipes['score'] += filtered_recipes['Fat'].apply(lambda x: 1 if 'NoPreference' in user_preferences['Fat'] or user_preferences['Fat'].lower() != x.lower() else 0)
    filtered_recipes['score'] += filtered_recipes['spiceLevel'].apply(lambda x: 1 if 'NoPreference' in user_preferences['spiceLevel'] or user_preferences['spiceLevel'].lower() != x.lower() else 0)
    # return recipes with top 3 scores
    filtered_recipes = filtered_recipes.sort_values('score', ascending=False).head(3)
    print('-------------------')
    print(filtered_recipes)
    return filtered_recipes.to_json(orient='records')

def recommendation_validation(recommendation):
    data = json.loads(recommendation)
    print('Before validation', data)
    validated = []
    for i in range(len(data)):
        if data[i]['score'] > 2:
            validated.append(data[i])

    print('After validation', validated)
    return validated

def initialize_conv_reco(products):
    system_message = f"""
    You are an intelligent recipe expertand you are tasked with the objective to \
    solve the user queries about any recipe from the catalogue: {products}.\
    You should keep the user profile in mind while answering the questions.\

    Start with a brief summary of each recipe in the following format:
    1. <Recipe Name> : <Description>, <Nutirional Value>, <prepTime>
    2. <Recipe Name> : <Description>, <Nutirional Value>, <prepTime>
    Ensure you have provided the correct information for each recipe and the overall response message.
    """
    conversation = [{"role": "system", "content": system_message }]
    return conversation
