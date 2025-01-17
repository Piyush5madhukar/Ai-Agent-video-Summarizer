import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file,get_file
import google.generativeai as genai

import time
from pathlib import Path

import tempfile

from dotenv import load_dotenv
load_dotenv()

import os

API_KEY=os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)


st.set_page_config(
    page_title="Phidata Mutlimodel ai agent",
    page_icon="🧊",
    layout="wide"
)

st.title("Phidata Mutlimodel ai agent")
st.header("Powered by Gemini")


@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video Ai summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )


#initialize the agent
multi_model_agent = initialize_agent()

video_file=st.file_uploader("Upload a video file", type=["mp4","mov","avi","mkv","webm"],help="Upload a video file to generate a summary")

if video_file:
    with tempfile.NamedTemporaryFile(delete=False,suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path,format="video/mp4",start_time=0)

    user_query=st.text_area(
        "What insighs do you want to get from the video?",
        placeholder="Ask anything about the video",
        help="Provide specific questions or insights you want to get from the video."

    )

    if st.button("Analyze Video",key="analyze_video_button"):
        if not user_query:
            st.warning("Please provide a query to analyze the video")
        else:
            try:
                with st.spinner("Analyzing video..."):
                    processed_video=upload_file(video_path)
                    while processed_video.state.name=="PROCESSING":
                        time.sleep(1)
                        processed_video=get_file(processed_video.name)

                    analysis_prompt=(
                        f"""
                        Analyzing video.
                        Respond to the following query using video insights
                        {user_query}

                        Provide a detailed,user-friendly and actionalbe response.
                        """
                    )
                    
                    response=multi_model_agent.run(analysis_prompt,videos=[processed_video])

                st.subheader("Analysis Results")
                st.markdown(response.content)

            except Exception as e:
                st.error(f"An error occured: {e}")  
            finally:
                Path(video_path).unlink(missing_ok=True) 

else:
    st.info("Please upload a video file to get started")

st.markdown(   
   """
<style>
.stTextArea textarea {
        height:100px;
}
</style>
""",
unsafe_allow_html=True
)     

