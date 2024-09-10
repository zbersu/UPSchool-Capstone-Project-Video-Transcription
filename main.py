import streamlit as st
import tempfile
import os
from openai import OpenAI
from dotenv import load_dotenv
import subprocess

# Load environment variables
load_dotenv()

# Get the FFmpeg path from the environment
ffmpeg_path = os.getenv('FFMPEG_PATH')

# Add FFmpeg to the system PATH
os.environ["PATH"] += os.pathsep + ffmpeg_path

# Initialize OpenAI client
client = OpenAI()


def main():
    st.set_page_config(page_title="AI-Powered Video Translation", page_icon="🎥", layout="wide")

    custom_css()

    st.title("🎥 AI-Powered Video Translation App")
    st.markdown("Translate video audio into multiple languages effortlessly.")

    # Initialize session state for transcripts and translations
    if 'transcript' not in st.session_state:
        st.session_state['transcript'] = None
    if 'translations' not in st.session_state:
        st.session_state['translations'] = {}
    if 'video_name' not in st.session_state:
        st.session_state['video_name'] = None

    # Sidebar for customization
    st.sidebar.title("Subtitle Customization")
    use_default = st.sidebar.checkbox("Use default settings", value=False)

    if use_default:
        color = "#FFFFFF"
        position = "Bottom"
        font_size = "16px"
        font_style = "Arial"
    else:
        color = st.sidebar.color_picker("Pick a subtitle color", "#FFFFFF")
        position = st.sidebar.selectbox(
            "Choose subtitle position",
            ["Bottom", "Top", "Top-Left", "Top-Right", "Middle", "Bottom-Left", "Bottom-Right"]
        )
        font_size = st.sidebar.slider("Select font size", 12, 32, 16)
        font_style = st.sidebar.selectbox(
            "Choose a font style",
            ["Arial", "Courier New", "Georgia", "Times New Roman", "Verdana", "Comic Sans MS"]
        )

    # Preview section for subtitle customization
    st.sidebar.markdown("### Preview")
    styled_preview = style_preview("Lorem Ipsum", color, position, font_size, font_style)
    st.sidebar.markdown(styled_preview, unsafe_allow_html=True)

    languages = st.multiselect(
        "Select languages for translation",
        ["Turkish", "English", "French", "German"],
        key="language_selection"
    )

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    if uploaded_file and languages:
        if st.button("Process Video", key="process_button"):
            process_video(uploaded_file, languages, color, position, font_size, font_style)

    # If there's already a transcript or translations in session state, display them
    if st.session_state['transcript'] or st.session_state['translations']:
        display_results(st.session_state['video_name'], st.session_state['transcript'],
                        st.session_state['translations'])



def custom_css():
    st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(to right, #141E30, #243B55);
        color: #FFFFFF;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #4CAF50;
        border-radius: 20px;
        font-weight: bold;
    }
    .stMultiSelect>div>div>div {
        background-color: rgba(255, 255, 255, 0.1);
    }
    .stFileUploader>div>div>div>button {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .stTextArea>div>textarea {
        background-color: rgba(255, 255, 255, 0.1);
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)


def save_uploaded_file(uploaded_file):
    file_path = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]).name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return file_path


def extract_audio(video_path):
    audio_path = tempfile.mktemp(suffix=".mp3")
    try:
        subprocess.run(['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path], check=True,
                       capture_output=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error extracting audio: {e.stderr.decode()}")
        raise
    return audio_path


def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="srt"
        )
    return transcription


def translate_transcript(transcript, target_language):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a skilled translator who can translate SRT files accurately."},
            {"role": "user",
             "content": f"Translate the following SRT content to {target_language}. Maintain the SRT format and timing:\n\n{transcript}"}
        ]
    )
    return response.choices[0].message.content


def style_srt(transcript, color, position, font_size, font_style):
    # Convert color and font style to SRT tags
    color_tag = f"<font color='{color}' size='{font_size}' face='{font_style}'>" if color else ""
    end_tag = "</font>" if color else ""

    # Add position based on selection
    position_tag = ""
    if position == "Top":
        position_tag = "{\\an8}"
    elif position == "Top-Left":
        position_tag = "{\\an7}"
    elif position == "Top-Right":
        position_tag = "{\\an9}"
    elif position == "Middle":
        position_tag = "{\\an5}"
    elif position == "Bottom-Left":
        position_tag = "{\\an1}"
    elif position == "Bottom-Right":
        position_tag = "{\\an3}"

    # Add styling to the transcript
    styled_transcript = ""
    for line in transcript.splitlines():
        if '-->' in line:
            styled_transcript += f"{line}\n"
        elif line.strip().isdigit() or line.strip() == '':
            styled_transcript += f"{line}\n"
        else:
            # Check if the line contains an exclamation mark
            if '!' in line:
                line = line.upper()  # Convert to uppercase if it has an exclamation mark

            # Apply color, position, font size, and style tags
            styled_transcript += f"{position_tag}{color_tag}{line}{end_tag}\n"

    return styled_transcript



def style_preview(text, color, position, font_size, font_style):
    # Simple function to generate HTML for preview
    color_tag = f"<span style='color:{color}; font-size:{font_size}px; font-family:{font_style};'>"
    end_tag = "</span>"

    # Position the preview correctly
    preview_html = "<div style='position:relative; width:100%; height:100px;'>"

    # Define text alignment based on the position
    alignment = "center"
    if "Left" in position:
        alignment = "left"
    elif "Right" in position:
        alignment = "right"

    if "Top" in position:
        vertical_position = "top:0;"
    elif "Middle" in position:
        vertical_position = "top:50%; transform:translateY(-50%);"
    else:  # Bottom positions
        vertical_position = "bottom:0;"

    preview_html += f"<div style='position:absolute; {vertical_position} width:100%; text-align:{alignment};'>"
    preview_html += f"{color_tag}{text}{end_tag}</div></div>"

    return preview_html


def process_video(uploaded_file, languages, color, position, font_size, font_style):
    with st.spinner("Processing video..."):
        try:
            # Save uploaded file
            temp_video_path = save_uploaded_file(uploaded_file)

            # Extract audio
            with st.spinner("Extracting audio..."):
                audio_path = extract_audio(temp_video_path)

            # Transcribe audio
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(audio_path)

            # Extract video name without extension
            video_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
            video_name = video_name.replace(" ", "_")  # Replace spaces with underscores

            # Apply styling to the transcript
            styled_transcript = style_srt(transcript, color, position, font_size, font_style)

            st.session_state['transcript'] = styled_transcript  # Store in session state
            st.session_state['video_name'] = video_name  # Store video name in session state
            st.success("Audio transcribed and styled successfully!")

            # Translate and style transcript for each language
            translations = {}
            for lang in languages:
                with st.spinner(f"Translating to {lang}..."):
                    translation = translate_transcript(transcript, lang)
                    styled_translation = style_srt(translation, color, position, font_size, font_style)
                    translations[lang] = styled_translation
                st.success(f"Translation to {lang} completed!")

            st.session_state['translations'] = translations  # Store translations in session state

            # Clean up temporary files
            os.unlink(temp_video_path)
            os.unlink(audio_path)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


def display_results(video_name, transcript, translations):
    st.subheader("Styled Transcript")
    st.text_area("Styled SRT Content", transcript, height=200)
    st.download_button("Download Styled SRT", transcript, f"{video_name}_styled_transcript.srt", "text/plain")

    for lang, translation in translations.items():
        with st.expander(f"{lang} Translation"):
            st.text_area(f"{lang} SRT Content", translation, height=200)
            st.download_button(f"Download {lang} SRT", translation, f"{video_name}_{lang.lower()}_translation.srt",
                               "text/plain")


if __name__ == "__main__":
    main()
