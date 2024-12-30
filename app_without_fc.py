from flask import Flask, redirect, url_for, render_template, request
from utils_without_fc import (
    initialize_conversation,
    get_chat_model_completions,
    moderation_check,
    intent_confirmation_layer,
    dictionary_present,
    get_filtered_recipes,
    extract_dictionary_from_string,
    recommendation_validation,
    initialize_conv_reco
)
import openai
import json

# Load OpenAI API key
openai.api_key = open("OpenAI_API_Key.txt", 'r').read().strip()

app = Flask(__name__)

# Initialize conversation variables
conversation_bot = []
conversation = initialize_conversation()

@app.route("/")
def default_func():
    global conversation_bot
    return render_template("conversation.html", name_xyz=conversation_bot)

@app.route("/end_conversation", methods=['POST', 'GET'])
def end_conv():
    global conversation_bot, conversation
    conversation_bot = []
    conversation = initialize_conversation()
    introduction = get_chat_model_completions(conversation)
    conversation_bot.append({'bot': introduction})
    return redirect(url_for('default_func'))

@app.route("/conversation", methods=['POST'])
def invite():
    global conversation_bot, conversation

    user_input = request.form["user_input_message"]

    # Moderation check for inappropriate content
    if moderation_check(user_input):
        return redirect(url_for('end_conv'))

    # Add user input to conversation history
    conversation.append({"role": "user", "content": user_input})
    conversation_bot.append({'user': user_input})

    # Get the assistant's response based on the conversation so far
    response_assistant = get_chat_model_completions(conversation)

    # Moderation check on the assistant's response
    if moderation_check(response_assistant):
        return redirect(url_for('end_conv'))

    # Intent confirmation layer to check if the assistant's response covers user preferences
    confirmation = intent_confirmation_layer(response_assistant)
    
    # If the assistant's response doesn't fully capture user preferences, ask the user for more details
    if "No" in confirmation:
        conversation.append({"role": "assistant", "content": response_assistant})
        conversation_bot.append({'bot': response_assistant})
        print(response_assistant)
    else:
        # Extract user preferences from the response
        user_preferences = dictionary_present(response_assistant)
        
        # Parse the user preferences string (it should be in JSON format)
        user_preferences = extract_dictionary_from_string(user_preferences)
        # Fetch filtered recipes based on user preferences
        filtered_recipes = get_filtered_recipes(user_preferences)
        print('is it there', filtered_recipes)
        if not filtered_recipes:
            conversation_bot.append({'bot': "Sorry, no recipes match your preferences. Please try again with different criteria."})
        else:
            validated_recommendations = recommendation_validation(filtered_recipes)
            conversation_reco = initialize_conv_reco(validated_recommendations)
            conversation_reco.append({"role": "user", "content": "This is my user profile" + str(user_preferences)})
            recommendation = get_chat_model_completions(conversation_reco)
            if (moderation_check(recommendation)):
                return redirect(url_for('end_conv'))
            conversation_bot.append({'bot': f"{recommendation}"})

    return redirect(url_for('default_func'))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
