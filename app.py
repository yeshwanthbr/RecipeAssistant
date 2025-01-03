from flask import Flask, redirect, url_for, render_template, request
from utils import (
    initialize_conversation,
    get_chat_model_completions,
    moderation_check,
    intent_confirmation_layer,
    dictionary_present,
    get_filtered_recipes,
    extract_dictionary_from_string,
    recommendation_validation,
    initialize_conv_reco,
    get_recipe_functions,
    get_top_3_recipes
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
    conversation_bot.append({'bot': introduction.content})
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
    response_assistant = get_chat_model_completions(conversation, get_recipe_functions)
    print(response_assistant.content)
    if response_assistant.function_call:
        function_name = response_assistant.function_call.name
        function_args = json.loads(response_assistant.function_call.arguments)
        recommended_recipes = []
        if function_name == "get_top_3_recipes":
            recommended_recipes = get_top_3_recipes(function_args)
        validated_recommendations = recommendation_validation(recommended_recipes)
        if len(validated_recommendations) == 0:
            conversation_bot.append({'bot': "Sorry, no recipes match your preferences. Please try again with different criteria."})
        else:
            conversation_reco = initialize_conv_reco(validated_recommendations)
            recommendation = get_chat_model_completions(conversation_reco)
            if (moderation_check(recommendation.content)):
                return redirect(url_for('end_conv'))
            conversation_bot.append({'bot': f"{recommendation.content}"})
    else:
        if moderation_check(response_assistant.content):
            return redirect(url_for('end_conv'))

        conversation_bot.append({'bot': response_assistant.content})

    return redirect(url_for('default_func'))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
