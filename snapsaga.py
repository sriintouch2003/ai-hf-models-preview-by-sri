# Import all the needed Libraries

from dotenv import load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import os
import requests
import streamlit as st

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

## Function that converts any Image to Text using Hugginggface Open Source Model


def img2text(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-large"
    )

    text = image_to_text(url)[0]["generated_text"]

    print(text)

    return text


## Lets Import Libraries Langchain and OpenAI for Story Generation using the context that came from the previous step


## Function that creates a Story from the Context Text using Open AI GTP3.5 Model


def generate_story(scenario):
    template = """
  You are a crime thriller story teller;
  You can generate a very engaging short story based on a simple narrative, the story should be no more than 50 words;

  CONTEXT: {scenario}
  STORY:
  """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(
        llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1),
        prompt=prompt,
        verbose=True,
    )

    story = story_llm.predict(scenario=scenario)

    print(story)

    return story


## Function that converts this text into an Audio story file using Open Source Huggingface Model


def text2speech(message):
    API_URL = (
        "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    )
    headers = {"Authorization": "Bearer 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"}
    payloads = {"inputs": message}

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open(
        "audio.flac",
        "wb",
    ) as file:
        file.write(response.content)


def main():
    from PIL import Image
    # Loading Image using PIL
    ai_icon = Image.open("ai.png")
    st.set_page_config(page_title="Covert a Picture into a Thrilling Story", page_icon=ai_icon)
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>

    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;'><i>SnapSaga</i></h2><br><h4 style='text-align: center;'><i>Transform Your Images into Thrilling Tales and Mesmerizing Audio</i></h4>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image....", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Generated Story"):
            st.write(story)

        st.audio("audio.flac")


if __name__ == "__main__":
    main()
