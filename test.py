import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
piakpiak = "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"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(piakpiak),
)

response = client.complete(
    messages=[
        SystemMessage("You are a helpful assistant."),
        UserMessage("What is the capital of France?"),
    ],
    temperature=1.0,
    top_p=1.0,
    model=model
)

print(response.choices[0].message.content)

