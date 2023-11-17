import pickle
from datetime import timedelta

import whisper
from tqdm import tqdm

OFFSET = 24
INPUT_FILE = "./sin_muertos.mp3"
OUTPUT_FILE = "./sin_muertos_mp3_24s_forward.srt"


def transcribe_audio(path):
    try:
        with open("segments.pkl", "rb") as f:
            segments = pickle.load(f)
        print("Segments loaded from pickle.")
    except:
        model = whisper.load_model("base")  # Change this to your desired model
        print("Whisper model loaded.")
        transcribe = model.transcribe(audio=path, verbose=True, fp16=False, language="es")
        segments = transcribe['segments']

        with open("segments.pkl", "wb") as f:
            pickle.dump(segments, f)

    for i in tqdm(range(len(segments))):
        # for segment in segments:
        segment = segments[i]
        startTime = str(0) + str(timedelta(seconds=int(segment['start']) + OFFSET)) + ',000'
        endTime = str(0) + str(timedelta(seconds=int(segment['end']) + OFFSET)) + ',000'
        text = segment['text']
        segmentId = segment['id'] + 1
        segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"

        srtFilename = OUTPUT_FILE
        with open(srtFilename, 'a', encoding='utf-8') as srtFile:
            srtFile.write(segment)

    return srtFilename


transcribe_audio(INPUT_FILE)
