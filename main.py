import streamlit as st
import tempfile
import os
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pysrt

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
    if 'dynamic_transcripts' not in st.session_state:
        st.session_state['dynamic_transcripts'] = {}
    if 'dynamic_translation_transcripts' not in st.session_state:
        st.session_state['dynamic_translation_transcripts'] = {}

    # Sidebar for functionality selection
    st.sidebar.title("Select Functionality")
    functionality = st.sidebar.radio("Choose a functionality", ("Subtitle Customization", "Dynamic Color-Changing Subtitles"))

    # Video upload
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    # Language selection
    languages = st.multiselect(
        "Select languages for translation",
        ["Turkish", "English", "French", "German"],
        key="language_selection"
    )

    if functionality == "Subtitle Customization":
        # Subtitle Customization Functionality
        with st.sidebar.expander("Subtitle Customization Options", expanded=True):
            use_default = st.checkbox("Use default settings", value=False)

            if use_default:
                color = "#FFFFFF"
                position = "Bottom"
                font_size = "16px"
                font_style = "Arial"
            else:
                color = st.color_picker("Pick a subtitle color", "#FFFFFF")
                position = st.selectbox(
                    "Choose subtitle position",
                    ["Bottom", "Top", "Top-Left", "Top-Right", "Middle", "Bottom-Left", "Bottom-Right"]
                )
                font_size = st.slider("Select font size", 12, 32, 16)
                font_style = st.selectbox(
                    "Choose a font style",
                    ["Arial", "Courier New", "Georgia", "Times New Roman", "Verdana", "Comic Sans MS"]
                )

            # Preview section for subtitle customization
            st.markdown("### Preview")
            styled_preview = style_preview("Lorem Ipsum", color, position, font_size, font_style)
            st.markdown(styled_preview, unsafe_allow_html=True)

        if uploaded_file and languages:
            if st.button("Process Video", key="process_button"):
                process_video(uploaded_file, languages, color, position, font_size, font_style, functionality)

        # Display results
        if st.session_state['transcript'] or st.session_state['translations']:
            display_results(st.session_state['video_name'], st.session_state['transcript'],
                            st.session_state['translations'], functionality)

    elif functionality == "Dynamic Color-Changing Subtitles":
        # Dynamic Color-Changing Subtitles Functionality
        with st.sidebar.expander("Dynamic Subtitle Options", expanded=True):
            #font_size = st.slider("Select font size", 12, 32, 24)
            # Set a fixed font size for dynamic subtitles
            font_size = 16
            st.markdown("""
                <style>
                .note-text {
                    font-family: 'Arial', sans-serif;  /* Set the same font as the title */
                    font-size: 16px;
                    opacity: 0.8;  /* Adjust the opacity to make it less opaque */
                    color: #FFFFFF;  /* You can adjust the text color as needed */
                }
                </style>
                <p class="note-text"><strong>Note:</strong> The font size for dynamic subtitles is fixed at 16 for optimal readability and placement within the bottom region of the video.</p>
                """, unsafe_allow_html=True)

        if uploaded_file and languages:
            if st.button("Process Video", key="process_button_dynamic"):
                process_video(uploaded_file, languages, None, None, font_size, None, functionality)

        # Display dynamic transcripts
        if st.session_state['dynamic_transcripts'] or st.session_state['dynamic_translation_transcripts']:
            display_results(st.session_state['video_name'], st.session_state['dynamic_transcripts'],
                            st.session_state['dynamic_translation_transcripts'], functionality)


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


def process_video(uploaded_file, languages, color, position, font_size, font_style, functionality):
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

            if functionality == "Subtitle Customization":
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

            elif functionality == "Dynamic Color-Changing Subtitles":
                # Process dynamic color-changing subtitles
                with st.spinner("Processing dynamic color-changing subtitles..."):
                    dynamic_transcript = process_dynamic_subtitles(temp_video_path, transcript, font_size)
                    st.session_state['dynamic_transcripts'] = dynamic_transcript  # Store in session state

                    dynamic_translations = {}
                    for lang in languages:
                        with st.spinner(f"Translating to {lang}..."):
                            translation = translate_transcript(transcript, lang)
                            dynamic_translation = process_dynamic_subtitles(temp_video_path, translation, font_size)
                            dynamic_translations[lang] = dynamic_translation
                        st.success(f"Translation to {lang} completed!")

                    st.session_state['dynamic_translation_transcripts'] = dynamic_translations  # Store in session state
                    st.session_state['video_name'] = video_name  # Store video name in session state
                    st.success("Dynamic color-changing subtitles processed successfully!")

            # Clean up temporary files
            os.unlink(temp_video_path)
            os.unlink(audio_path)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


def display_results(video_name, transcript, translations, functionality):
    if functionality == "Subtitle Customization":
        st.subheader("Styled Transcript")
        st.text_area("Styled SRT Content", transcript, height=200)
        st.download_button("Download Styled SRT", transcript, f"{video_name}_styled_transcript.srt", "text/plain")

        for lang, translation in translations.items():
            with st.expander(f"{lang} Translation"):
                st.text_area(f"{lang} SRT Content", translation, height=200)
                st.download_button(f"Download {lang} SRT", translation, f"{video_name}_{lang.lower()}_translation.srt",
                                   "text/plain")

    elif functionality == "Dynamic Color-Changing Subtitles":
        st.subheader("Dynamic Color-Changing Transcript")
        st.text_area("Dynamic SRT Content", transcript, height=200)
        st.download_button("Download Dynamic SRT", transcript, f"{video_name}_dynamic_transcript.srt", "text/plain")

        for lang, translation in translations.items():
            with st.expander(f"{lang} Dynamic Translation"):
                st.text_area(f"{lang} Dynamic SRT Content", translation, height=200)
                st.download_button(f"Download {lang} Dynamic SRT", translation,
                                   f"{video_name}_{lang.lower()}_dynamic_translation.srt", "text/plain")


def process_dynamic_subtitles(video_path, srt_content, font_size):
    # Convert SRT content to pysrt.SubRipFile
    subs = pysrt.from_string(srt_content)

    # Load video
    video = cv2.VideoCapture(video_path)

    # Ensure the video starts from the first frame
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare a list to hold dominant colors for each subtitle
    dominant_colors = []

    # Process each subtitle
    for sub in subs:
        # Calculate frame numbers for the subtitle timing
        start_frame = int((sub.start.ordinal / 1000) * fps)
        end_frame = int((sub.end.ordinal / 1000) * fps)

        # Ensure frame numbers are within video frame count
        start_frame = min(start_frame, frame_count - 1)
        end_frame = min(end_frame, frame_count - 1)

        # Set video to the middle frame of the subtitle duration
        mid_frame = (start_frame + end_frame) // 2
        video.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)

        ret, frame = video.read()
        if ret:
            # Get dominant color from the bottom region of the frame
            height = frame.shape[0]
            bottom_region = frame[int(height * 0.7):, :]
            dominant_color = get_dominant_color(bottom_region)
            text_color = get_contrasting_color(dominant_color)
            dominant_colors.append(text_color)
        else:
            dominant_colors.append((255, 255, 255))  # Default to white if frame not read

    # Close video
    video.release()

    # Convert subtitles to ASS format with dynamic colors
    ass_content = convert_subs_to_ass(subs, dominant_colors, font_size)

    return ass_content



def get_dominant_color(image, num_colors=1):
    pixels = np.float32(image.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, num_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant_color = palette[np.argmax(counts)]
    return tuple(map(int, dominant_color))


def get_contrasting_color(color):
    # Convert BGR to RGB for PIL
    color_rgb = (color[2], color[1], color[0])
    # Calculate luminance
    luminance = (0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]) / 255
    return (255, 255, 255) if luminance < 0.5 else (0, 0, 0)


def convert_subs_to_ass(subs, colors, font_size):
    # Start building ASS content
    ass_header = """[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayResX: 384
PlayResY: 288
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(font_size=font_size)

    ass_events = ""
    for sub, color in zip(subs, colors):
        # Convert time format
        start_time = format_ass_time(sub.start)
        end_time = format_ass_time(sub.end)
        # Convert color to ASS hex format (BGR)
        color_hex = '&H{0:02X}{1:02X}{2:02X}'.format(color[2], color[1], color[0])
        # Build dialogue line
        dialogue = "Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\c{color}}}{text}\n".format(
            start=start_time,
            end=end_time,
            color=color_hex,
            text=sub.text.replace('\n', '\\N')
        )
        ass_events += dialogue

    return ass_header + ass_events


def format_ass_time(time):
    hours = time.hours
    minutes = time.minutes
    seconds = time.seconds
    milliseconds = int(time.milliseconds / 10)
    return "{:01d}:{:02d}:{:02d}.{:02d}".format(hours, minutes, seconds, milliseconds)


if __name__ == "__main__":
    main()