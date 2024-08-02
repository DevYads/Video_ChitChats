# Video-ChitChats: Chat with a Website from URL - AI Chatbot with Streamlit GUI

## Overview
Welcome to the GitHub repository for the LangChain AI Chatbot with Streamlit GUI! 

Video-ChitChats is a Streamlit-based web application that allows users to interact with video transcripts using AI-generated responses. Users can input a YouTube video link, receive a brief description of the video's content, get suggested questions, and chat with an AI to ask further questions related to the video's content. 

This project is a comprehensive guide to building a chatbot interacting with video transcripts, extracting information. It leverages the power of LangChain and integrates it with a Streamlit GUI for an enhanced user experience.

## Features

- **Video Transcription**: Analyze YouTube videos to generate transcripts.
- **Suggested Questions**: Get AI-generated suggested questions based on the video content.
- **Interactive Chat**: Chat with an AI to ask questions and get responses related to the video.


## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:

    ```sh
    git clone https://github.com/DevYads/Video-ChitChats.git
    cd Video-ChitChats
    ```

2. **Create and activate a virtual environment** (optional but recommended):

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app**:

    ```sh
    streamlit run app.py
    ```

2. **Enter a YouTube Video Link**: Paste the link in the provided text input box.

3. **Analyze the Video**: Click on "Analyze The Video" to process the video and generate a transcript.

4. **Suggested Questions**: View and select from the AI-suggested questions.

5. **Ask Questions**: Type in your own questions and interact with the AI.

## Configuration

- **Model Configuration**: The app uses Google Generative AI with the `gemini-1.5-flash` model. You can adjust the model settings such as temperature and max tokens in the `main()` function.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or feedback, please contact [Yadvendra Garg](mailto:yadvendragarg123@gmail.com).
