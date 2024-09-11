# AI-Powered Video Translation - README

## Overview

**AI-Powered Video Translation** is a Streamlit-based web application that processes video files to generate customized and dynamic subtitles. It offers functionalities to translate video audio into multiple languages and apply dynamic, color-changing subtitles based on the dominant colors in the video frame. The app provides users with a visually appealing interface with customizable subtitle styling and dynamic subtitle handling.

## Features

- **Subtitle Customization**: Users can customize subtitles by choosing colors, fonts, sizes, and positions.
- **Dynamic Color-Changing Subtitles**: Subtitles change color dynamically based on the dominant color in the video frame.
- **API Integration with OpenAI**: Utilizes OpenAI's API for transcribing and translating video audio.
- **Multi-Language Translations**: Supports translations into various languages such as Turkish, English, French, and German.
- **User-Friendly Interface**: Built with Streamlit for an interactive and easy-to-use experience.
- **Customizable Subtitle Appearance**: Users can adjust subtitle styles or use default settings.

## Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)
- Streamlit
- OpenAI API key
- FFmpeg (for video and audio processing)
- `.env` file with the following variables:
  - `OPENAI_API_KEY=your_openai_api_key`
  - `FFMPEG_PATH=your_ffmpeg_directory_path`

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-repo/video-translation-app.git
    ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Navigate to the Project Directory**

    ```bash
    cd path/to/video-translation-app
    ```

4. **Fill in the `.env` File**

   Ensure your `.env` file contains the following environment variables:

    ```bash
    OPENAI_API_KEY=your_openai_api_key
    FFMPEG_PATH=your_ffmpeg_directory_path
    ```

5. **Ensure FFmpeg is Installed**

   If you do not have FFmpeg installed, you can follow the instructions for your operating system:

   - **On macOS**: Install with Homebrew:
     ```bash
     brew install ffmpeg
     ```

   - **On Ubuntu**: Install with apt:
     ```bash
     sudo apt install ffmpeg
     ```

   - **On Windows**: Download the FFmpeg executable from [FFmpeg's website](https://ffmpeg.org/download.html) and add it to your system PATH.

## Running the App

To run the Streamlit application, use the following command in your terminal:

```bash
streamlit run main.py
```
The application will open in your default web browser.

To stop the app, press `Ctrl+C` in the terminal.

## Customization

- **Subtitle Customization**: Modify subtitle colors, font size, and position by changing the values in the sidebar.
- **Background & Styling**: You can adjust the custom CSS in the app to modify the layout and appearance (such as background gradients and buttons).
- **Change the Default Font Size for Dynamic Subtitles**: The font size for dynamic subtitles is fixed at 16, but you can adjust this in the code by modifying the `font_size` parameter in the dynamic subtitle section.

## Troubleshooting

- **Audio Transcription Issues**: Ensure that the OpenAI API key is correctly set and that you have an active subscription with access to the transcription model.
- **Missing Subtitles or Delays**: If subtitles do not appear at the correct time, ensure that FFmpeg is properly extracting the audio and that your video file is in a supported format.
- **Subtitle Not Displaying**: Check that the video file is uploaded correctly, and the subtitles are processed successfully. Ensure that the subtitle SRT or ASS files are generated.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request if you'd like to improve the project.

## Acknowledgements

- **OpenAI API**: Used for audio transcription and translation.
- **FFmpeg**: For video and audio processing.
- **Streamlit**: The framework used to create the web application.
