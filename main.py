import os
import warnings
from typing import Dict

from openfabric_pysdk.utility import SchemaUtil

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

import random
import json
import torch
from model import NeuralModel
from nltk_utils import bag_of_words, tokenize


############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []
    for text in request.text:
        # TODO Add code here
        response = 'Hello!'
        output.append(response)

    return SchemaUtil.create(SimpleText(), dict(text=output))

def generate_chat_response(user_input: str) -> str:
    # Load  trained model and other necessary data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('intents.json', 'r') as f:
        intents = json.load(f)
    FILE = "data.pth"
    data = torch.load(FILE)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralModel(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Process user input
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    # Generate chat response
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "This is not too scientific from you"


if __name__ == "__main__":
    print("Welcome to My Science ChatBot!")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Process user input and generate chat response
        chat_response = generate_chat_response(user_input)
        print(f"{'OpenFabric'}: {chat_response}")
