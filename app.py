import re
from langchain import PromptTemplate
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import json
import os
from langchain.chat_models import ChatOpenAI
import time


import os

st.set_page_config(layout="wide")


class WrapperFrame_v1:
    def __init__(self, client):
        self.client = client


class Sentiment_Agent(WrapperFrame_v1):
    def __init__(self, client):
        super().__init__(client)

        self.sentiment_agent = """

You are an AI language model trained to analyze and classify tweets {tweet} about El Salvador.
Your job is to categorize each tweet into one of three categories based on its sentiment: 1 (positive), 0 (neutral), or -1 (negative) using a
local Perspective from El salvador political scene.

Guidelines:
. 1 (Positive): The tweet expresses a positive sentiment, such as happiness, praise, or admiration.
. 0 (Neutral): The tweet is neutral,. It may present facts, ask questions, or be generally balanced in tone or a statement or just mention a fact or a name.
.-1 (Negative): The tweet expresses a negative sentiment, such as criticism, disappointment, or disapproval.

Examples for Reference:

1 (POS):
. Tweet: "Dios lo bendiga por ser un gran ser humano."
   - Reasoning: The tweet is praising and expressing positive feelings towards someone.

. Tweet: "Me encanta la humanidad de nuestro astronauta, un hombre con gran corazón."
   - Reasoning: The tweet shows love and admiration for the astronaut.

. Tweet: "Dos grandes hombres haciendo historia. Gracias por todo."
   - Reasoning: The tweet appreciates the actions of two men, showing gratitude and positivity.

0 (NEU):
. Tweet: "Tweet Maria Alicia Alas Moreno "
   - Reasoning:The tweet states a fact about the need for a certain type of president without strong positive or negative sentiment.

. Tweet: "Tweet Nuestros obstaculos son mentales"
   - Reasoning: The tweet presents a factual statement about the astronaut's activities without expressing an opinion.

. Tweet: "¿Qué opinan sobre las últimas noticias del presidente?"
   - Reasoning: The tweet is asking a question and does not convey a positive or negative sentiment.

-1 (NEG):
. Tweet: "No estoy de acuerdo con las políticas actuales."
   - Reasoning: The tweet expresses disagreement with current policies, indicating a negative sentiment.

. Tweet: "Es una vergüenza que esto esté sucediendo en nuestro país."
   - Reasoning: The tweet expresses disappointment and criticism about a situation in the country.

. Tweet: "Las decisiones del presidente están dañando la economía."
   - Reasoning: The tweet criticizes the president's decisions, showing a negative sentiment about their impact on the economy.


    Instructions:

      -Read the tweet carefully.

      - Determine the sentiment expressed based on the content, using the following definitions and references:

      - Generate a Json output followin this pattern:

              (
                "classification": 1,0,-1.
                "reasoning: reasoning of the answer in 50 words.
              ).

        """
        self.prompt_sentiment_agent = PromptTemplate(
            template=self.sentiment_agent, input_variables=["tweet"]
        )

        # Corrected the LLMChain instantiation
        self.llm_chain_sentiment_agent = LLMChain(
            prompt=self.prompt_sentiment_agent, llm=self.client
        )

    def run_agent(self, tweet):
        self.tweet = tweet

        try:
            sentiment_output = self.llm_chain_sentiment_agent.run({"tweet": self.tweet})
            return sentiment_output
        except Exception as e:
            print(e)
            return "Error sentiment agent"


class EvaluationProcessPipeline:

    def __init__(self, client, text):
        self.client = client
        self.text = text
        self.agent = Sentiment_Agent(self.client)

    def process_tweet(self):
        max_attempts = 1  # Define maximum number of retry attempts
        attempts = 0

        while attempts < max_attempts:
            try:
                # Run the selected Agent
                entry_result = self.agent.run_agent(self.text)
                print(
                    "Raw entry_result:", entry_result
                )  # Debugging line to see what is being returned

                # Attempt to parse the JSON data
                data = json.loads(entry_result)
                labels = data["classification"]
                print(
                    "Extracted labels:", labels
                )  # Debugging line to confirm label extraction
                return labels  # Successful parsing and return, exit loop
            except (json.JSONDecodeError, Exception) as error:
                # Handle both JSON parsing errors and other exceptions
                print(f"Error encountered (attempt {attempts + 1}):", error)
                attempts += 1
                time.sleep(1)  # Sleep before retrying

        print("Failed after", max_attempts, "attempts.")
        return []


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def add_logo():
    st.image("assets/images/Omdena.png", width=250)


def main():
    add_logo()

    st.markdown(
        "<h1 style='text-align: center; color:white;'>IREX-Sentiment-Analyzer</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            "<h4 style color:black;'>Chat-GPT based sentiment analysis</h4>",
            unsafe_allow_html=True,
        )

    user_input = st.text_input("", key="input")
    if st.button("Classify GPT"):
        # agent = EvaluationProcessPipeline(llm, user_input)
        # label = agent.process_tweet()
        # st.write(label)
        st.write("hello world")


if __name__ == "__main__":
    main()
