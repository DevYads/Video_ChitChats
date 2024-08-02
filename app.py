from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerationConfig, GenerativeModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import streamlit as st
import os
import json

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt_text = """You are YouTube Video Chats.
            You will be taking transcript text and summarizing the entire video
            and providing the important summary in points within 600-800 Words in Marathi.
            
            Here is the transcript text:\n"""


def generate_gemini_content(transcript_text, prompt):

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
    )
    response = model.generate_content(prompt + transcript_text)
    # print(response)
    return response.text


def extract_transcript_details(video_id):
    # print(f"Video ID: {video_id}")
    try:
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "hn"])
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
        return None
    except VideoUnavailable:
        st.error("Video is unavailable.")
        return None
    except Exception as e:
        st.error(f"Internal Error Occurred\n Details- {str(e)}")
        return None


def get_text_chunks(text):
    text_splitters = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitters.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_stores = FAISS.from_texts(text_chunks, embeddings)
    vector_stores.save_local("yt_trans_faiss_index")


def get_conversational_chat_chain(model):
    prompt_template = """
    You are ChatBot for YouTube Video Transcripts.
    You will be taking transcript text.
    Answer the question as detailed as possible from the provided video context transcript.
    If the question is related to the topic discussed but does not have answers in the provided context
    Then say that provided context does not have answer but I can give the answer and then you will provide
    the answers from your database.
    Context:\n {context}?\n
    Question:\n{question}\n
    
    Answer: 
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['context', 'question']
    )

    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


def get_user_input(model, user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        new_db = FAISS.load_local("yt_trans_faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chat_chain(model)
        # print(chain)

        response = chain(
            inputs={
                "input_documents": docs,
                "question": user_question
            }
        )

        return response
    except Exception as e:
        st.error(f"Error in Generating AI Response: {e}")
        return None


def generate_suggested_questions(description):
    required_response_schema = {
        "title": "Suggested Questions Schema",
        "description": "Schema for representing AI-generated suggested questions based on a video transcript",
        "type": "object",
        "properties": {
            "video_description": {
                "type": "string",
                "description": "A brief description or summary of the video content"
            },
            "suggested_questions": {
                "type": "array",
                "description": "List of AI-generated suggested questions based on the video content",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "A suggested question"
                        }
                    },
                    "required": ["question"]
                }
            }
        },
        "required": ["description", "suggested_questions"]
    }
    try:

        model = GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=GenerationConfig(response_mime_type="application/json")
        )

        prompt = f"""
        Given the following summary of transcript of the video,
        **Summary:**
        {description}
        
        generate three thoughtful and engaging questions that might be asked from the video.
        The questions should focus on key topics, interesting points, or areas for further exploration.
        
        Follow the JSON schema.<JSONSchema>{json.dumps(required_response_schema)}</JSONSchema>
        """

        response = model.generate_content(prompt)
        response_text = response.candidates[0].content.parts[0].text
        return response_text
    except json.JSONDecodeError as e:
        st.error(f"JSON decoding error in generate_suggested_questions: {e}")
        return None
    except Exception as e:
        st.error(f"Error in Generating Suggested Questions: {e}")
        return None


def main():
    st.set_page_config("Video-ChitChats")

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
        max_tokens=800,
    )

    youtube_link = st.text_input("Enter YouTube Video Link:", placeholder="https://www.youtube.com/watch?v=video_url_id")
    youtube_video_id = ""
    if youtube_link.startswith("https://www.youtube.com/watch?v="):
        youtube_video_id = youtube_link.split("=")[1]
    elif youtube_link.startswith("https://youtu.be/"):
        youtube_video_id = youtube_link.split("/")[-1]
    # print(youtube_video_id)

    if youtube_link:
        st.video(f"https://youtu.be/{youtube_video_id}")

    st.subheader("Chat with Video")

    if 'response' not in st.session_state:
        st.session_state.response = ""
    if 'analyzed_video' not in st.session_state:
        st.session_state.analyzed_video = False

    if st.button("Analyze The Video"):
        st.session_state.analyzed_video = False
        with st.spinner("Analyzing..."):
            try:
                transcription = extract_transcript_details(youtube_video_id)
                if transcription:
                    with open(f"transcript_files/YouTube-ID-{youtube_video_id}-transcript.txt", "w") as f:
                        f.write(transcription)
                    # print("Transcription Saved!")
                    if transcription:
                        text_chunks = get_text_chunks(transcription)
                        get_vector_store(text_chunks)

                        desc_model = ChatGoogleGenerativeAI(
                            model="gemini-1.5-flash",
                            temperature=0.4,
                            max_tokens=300,
                            max_retires=2,
                        )
                        summary_prompt = "Describe the content and summarize the video in 500 words"
                        desc_res = get_user_input(desc_model, summary_prompt)
                        desc = desc_res['output_text']
                        # print(desc)
                        response = generate_suggested_questions(description=desc)
                        st.session_state.response = response
                        st.success("Analyzed the Video")
                        st.session_state.analyzed_video = True

                else:
                    st.error("Unable To Analyze the Video!")
            except Exception as e:
                st.error(f"Error during video analysis: {e}")

    if st.session_state.analyzed_video:
        try:
            json_response = json.loads(st.session_state.response)
            if 'suggested_questions' in json_response:
                suggested_questions = [q['question'] for q in json_response['suggested_questions']]
            else:
                suggested_questions = []
                st.warning("No suggested questions found.")
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}")
            suggested_questions = []

        selected_question = st.selectbox(
            label="Suggested Questions",
            options=suggested_questions,
            index=None,
            placeholder="Ask from Suggested Questions"
        )

        if selected_question:
            query_response = get_user_input(model, selected_question)

            if query_response is not None:
                if 'output_text' in query_response:
                    st.write(f"{query_response['output_text']}")
                else:
                    st.write("No output text found in the response.")
            else:
                st.write("The response from the chain is None.")

    user_query = st.text_area("Ask me Anything about the PDF", placeholder="Hey, Ask me anything about your PDF.")

    if st.button("Ask Me"):
        if user_query.strip():
            try:
                query_response = get_user_input(model, user_query)

                if query_response and 'output_text' in query_response:
                    st.write(f"{query_response['output_text']}")
                else:
                    st.write("No output text found in the response.")
            except Exception as e:
                st.error(f"Error getting response for user query: {e}")
        else:
            st.warning("Please enter a question to ask.")


if __name__ == "__main__":
    main()
    # Common User Prompt
    # what is explained in the video and summarize it in points and focus on the important points to take notes
    # what is explained in the pdf  and summarize most important points to note
    # What example the video takes and provide the codes for different functions, expressions mention in the video

