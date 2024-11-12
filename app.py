import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import io
import tempfile

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from google.generativeai import GenerationConfig, GenerativeModel
from langchain_community.vectorstores import FAISS  # For Vector Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable


def get_pdf_text(pdf_docs):
    text = ""
    try:
        print(pdf_docs)
        for i, pdf in enumerate(pdf_docs):
            # pdf_stream = io.BytesIO(pdf)
            pdf_reader = PdfReader(pdf)
            print(f"PDF {i + 1} with Number of Pages: {len(pdf_reader.pages)}")
            for page in pdf_reader.pages:
                text += page.extract_text().strip()
        print(f"Number of Characters: {len(text)}")
        # with open("PDF_text.txt", "w", encoding="utf-8") as f:
        #     f.write(text)
        return text
    except Exception as e:
        st.error(f"Internal Error Occurred\n Details- {str(e)}")
        return None


def extract_transcript_details(video_id):
    print(f"Video ID: {video_id}")
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")


def get_conversation_chat_chain():
    # If the user’s question relates to the topic discussed in the document or video but the specific answer is not available in the provided context, explicitly state that the provided context does not contain the answer.
    prompt_template = """
    You are a highly advanced AI agent designed to assist users by providing detailed and comprehensive answers based on the context of a PDF documents or Video. 
    Your goal is to deliver in-depth responses that thoroughly cover the user's question, referencing relevant sections, data, or insights from the documents or Video content.
    Then, offer the correct and detailed answer based on your knowledge or database in 500-750 words without providing incorrect information.
    
    Additional Instructions:

    1. If the question requires interpretation, make sure to provide your analysis in a clear, structured manner.
    2. If multiple sections of the document are relevant, ensure you reference and connect them to build a coherent answer.
    3. If data or figures are involved, include them and explain their significance within the context of the broader document.
    4. Strive to create responses that are both informative and insightful, helping the user gain a deep understanding of the subject matter.
    
    
    Prompt Structure:

    Context:
    <CONTEXT>{context}</CONTEXT>
    
    Conversation History:
    {conversation_history}
    
    Question: {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template=prompt_template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        temperature=0.2,
    )

    # chain = load_qa_chain(
    #     llm=model,
    #     chain_type="stuff",
    #     prompt=prompt,
    # )
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain


def get_user_input(user_question, chat_history):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversation_chat_chain()

        ai_response = chain.invoke({
            "question": user_question,
            "context": docs,
            "conversation_history": chat_history
        })

        return ai_response
    except Exception as exception:
        st.error(f"Error in Generating AI Response: {exception}")
        return None


def generate_suggested_questions(description, num_ques):
    required_response_schema = {
        "title": "Suggested Questions Schema",
        "description": "Schema for representing AI-generated suggested questions based on a Document Content",
        "type": "object",
        "properties": {
            "video_description": {
                "type": "string",
                "description": "A brief description or summary of the Document Content"
            },
            "suggested_questions": {
                "type": "array",
                "description": "List of AI-generated suggested questions based on the Document Content",
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
            model_name="gemini-1.5-flash-8b",
            generation_config=GenerationConfig(response_mime_type="application/json")
        )

        prompt = f"""
        Given the following summary of content of the document or video,
        **Summary:**
        {description}

        Generate {num_ques} questions that might be asked and their answer can be found in the document or video content.
        The questions should focus on key topics, interesting points.

        Follow the JSON schema.<JSONSchema>{json.dumps(required_response_schema)}</JSONSchema>
        """

        ai_response = model.generate_content(prompt)
        response_text = ai_response.candidates[0].content.parts[0].text
        return response_text
    except json.JSONDecodeError as e:
        st.error(f"JSON decoding error in generate_suggested_questions: {e}")
        return None
    except Exception as e:
        st.error(f"Error in Generating Suggested Questions: {e}")
        return None


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(
                content="Hi There, I'm Chat with Documents or Video AI Assistant. Ask me anything about your Documents/Video."),
        ]

    if 'sq_response' not in st.session_state:
        st.session_state.sq_response = None
    if 'analyzed_video' not in st.session_state:
        st.session_state.analyzed_video = False
    if 'num_sq' not in st.session_state:
        st.session_state.num_sq = 0
    if 'wide_layout' not in st.session_state:
        st.session_state.wide_layout = "wide"
    if 'user_question_input' not in st.session_state:
        st.session_state.user_question_input = "Display"

    st.set_page_config("Chat With Multiple PDFs or Video", layout=st.session_state.wide_layout)

    with st.sidebar:
        layout_toggle = st.toggle("Toggle the Layout To Save a File")
        if layout_toggle:
            st.session_state.wide_layout = "centered"
            st.session_state.user_question_input = "Display"
        else:
            st.session_state.wide_layout = "wide"
            st.session_state.user_question_input = "None"
        st.title("Set Up:")

        video_toggle = st.toggle("Chat With Video")
        pdf_docs = None
        youtube_video_id = None

        if video_toggle:
            youtube_link = st.text_input("Enter YouTube Video Link:",
                                         placeholder="https://www.youtube.com/watch?v=video_url_id")
            youtube_video_id = ""
            if youtube_link.startswith("https://www.youtube.com/watch?v="):
                youtube_video_id = youtube_link.split("=")[1]
            elif youtube_link.startswith("https://youtu.be/"):
                youtube_video_id = youtube_link.split("/")[-1]
            # print(youtube_video_id)

            if youtube_link:
                st.video(f"https://youtu.be/{youtube_video_id}")
        else:
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on Analyze The PDFs", accept_multiple_files=True, type=['pdf'])
        want_suggested_questions = st.toggle(label="Suggested Questions")
        num_of_ques = 0
        if want_suggested_questions:
            num_of_ques = st.slider(label="Number of Suggested Questions", min_value=0, max_value=20, value=0)
        # print(len(pdf_docs.read()))
        if st.button("Analyze"):
            if pdf_docs:
                st.session_state.analyzed_video = False
                with st.spinner("Analyzing with AI..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)

                            st.success("Analyzed the PDF")
                            st.session_state.analyzed_video = True
                            st.session_state.chat_history = [
                                AIMessage(
                                    content="Hi There, I'm Chat with Documents AI Assistant. Ask me anything about your Documents."),
                            ]
                            # st.balloons()
                        else:
                            st.error("Unable To Analyze the Document!")
                    except Exception as e:
                        st.error(f"Error during Document analysis: {e}")
            elif youtube_video_id:
                st.session_state.analyzed_video = False
                with st.spinner("Analyzing with AI..."):
                    try:
                        transcription = extract_transcript_details(youtube_video_id)
                        # if transcription:
                        #     with open(f"transcript_files/YouTube-ID-{youtube_video_id}-transcript.txt", "w") as f:
                        #         f.write(transcription)
                        if transcription:
                            text_chunks = get_text_chunks(transcription)
                            get_vector_store(text_chunks)

                            st.success("Analyzed the Video")
                            st.session_state.analyzed_video = True
                            st.session_state.num_sq = 0
                            st.session_state.chat_history = [
                                AIMessage(
                                    content="Hi There, I'm Chat with Video AI Assistant. Ask me anything about your Documents."),
                            ]
                            # st.balloons()
                        else:
                            st.error("Unable To Analyze the Video!")
                    except Exception as e:
                        st.error(f"Error during Video analysis: {e}")
            else:
                st.warning(body="Please, Upload the Document / Video Url First")
        if st.session_state.num_sq != num_of_ques and st.session_state.analyzed_video:
            st.session_state.num_sq = num_of_ques
            desc_res = None
            if want_suggested_questions and num_of_ques > 0:
                summary_prompt = "Describe the content and summarize the document/video in important points in 1000 words."
                desc_res = get_user_input(
                    user_question=summary_prompt,
                    chat_history=[
                        AIMessage(
                            content="Hi There, I'm Chat with Documents AI Assistant. Ask me anything about your Documents/Video.")
                    ]
                )

            sq_response = generate_suggested_questions(description=desc_res, num_ques=num_of_ques)
            st.session_state.sq_response = sq_response

    st.title("Chat with PDFs or Video")

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", ):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_ques = None
    if st.session_state.user_question_input == 'Display':
        user_ques = st.chat_input(placeholder="Ask me anything about your Document/Video.")

    if st.session_state.analyzed_video and want_suggested_questions and num_of_ques > 0 and st.session_state.sq_response:
        try:
            json_response = json.loads(st.session_state.sq_response)
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
            selected_ques = selected_question
            if user_ques is None:
                user_ques = selected_ques

    if user_ques is not None and user_ques.strip() != "":
        # print(user_ques)
        st.session_state.chat_history.append(HumanMessage(content=user_ques))
        # print(st.session_state.chat_history)

        with st.chat_message("Human"):
            st.markdown(user_ques)

        with st.chat_message("AI"):
            try:
                response = get_user_input(
                    user_question=user_ques,
                    chat_history=st.session_state.chat_history
                )
                # print(response)
                if response is not None:
                    st.markdown(response)
                    st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                error_msg = str(e)
                st.error(error_msg)


if __name__ == "__main__":
    main()
