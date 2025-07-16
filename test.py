"""Run this model in Python

> pip install azure-ai-inference
"""
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

# To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings. 
# Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential("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"),
)

response = client.complete(
    messages=[
        SystemMessage("""You are a helpful assistant."""),
        UserMessage("Can you explain the basics of machine learning?"),
    ],
    model="meta/Meta-Llama-3.1-8B-Instruct",
    temperature=1.0,
    max_tokens=1000,
    top_p=1.0
)

print(response.choices[0].message.content)
