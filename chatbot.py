import streamlit as st
import boto3
import json

# --- Configuration ---
REGION_NAME = "us-east-1"
MODEL_ID = "anthropic.claude-v2"  # or other supported model IDs

# --- AWS Bedrock client ---
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=REGION_NAME)

# --- Call Bedrock model ---
def query_bedrock(prompt, model_id=MODEL_ID):
    client = get_bedrock_client()
    body = {
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        "max_tokens_to_sample": 300,
        "temperature": 0.7,
        "stop_sequences": ["\n\nHuman:"]
    }
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    response_body = json.loads(response["body"].read())
    return response_body.get("completion", "No response.")

# --- Streamlit UI ---
st.set_page_config(page_title="AWS Bedrock Chatbot", layout="centered")
st.title("🤖 AWS Bedrock Chatbot")
st.markdown("Chat with an Anthropic Claude model using AWS Bedrock.")

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    with st.spinner("Thinking..."):
        try:
            result = query_bedrock(user_input)
            st.markdown(f"**Assistant:** {result}")
        except Exception as e:
            st.error(f"Error: {e}")
