"""Run this model in Python

> pip install azure-ai-inference
"""
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

# To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings. 
# Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(os.environ["API_KEY"]),
    api_version="2024-12-01-preview",
)

response = client.complete(
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        UserMessage("Can you explain the basics of machine learning?"),
    ],
    model="openai/o4-mini"
)

print(response.choices[0].message.content)
