import streamlit as st
import boto3
import json

# Configuration
REGION = "us-east-1"
MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229"
KNOWLEDGE_BASE_ID = ""  # replace with your actual ID

# AWS Bedrock Agent runtime client
client = boto3.client("bedrock-agent-runtime", region_name=REGION)

def query_with_knowledge_base(prompt):
    response = client.retrieve_and_generate(
        input={
            "text": prompt
        },
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": MODEL_ARN
            }
        }
    )
    return response["output"]["text"]

# Streamlit UI
st.title("📚 Bedrock RAG Chatbot")
st.markdown("Powered by a Bedrock Knowledge Base")

user_input = st.text_input("Ask a question:")

if st.button("Submit") and user_input:
    with st.spinner("Querying knowledge base..."):
        try:
            answer = query_with_knowledge_base(user_input)
            st.markdown(f"**Answer:** {answer}")
        except Exception as e:
            st.error(f"Failed to query Bedrock: {e}")
