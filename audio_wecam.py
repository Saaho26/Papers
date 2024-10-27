import asyncio
import cv2
import pyaudio
import wave
from concurrent.futures import ThreadPoolExecutor
import os
import sounddevice  as sd 
import numpy as np
from scipy.io.wavfile  import write
from moviepy.editor import VideoFileClip ,AudioFileClip
import time     

os.environ['QT_QPA_PLATFORM'] = 'xcb'
device_index = 4  # Give Device index based on your machine 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Lower sample rate to reduce data volume
CHUNK = 2048  # Increase chunk size to prevent overflow
WAVE_OUTPUT_FILENAME = "output.wav"
video_output_file= 'output.mp4'
def save_video():
   
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    frame_rate = 20.0
    frame_size = (640 , 480)
    out = cv2.VideoWriter(video_output_file , fourcc , frame_rate , frame_size )
    return out 

audio = pyaudio.PyAudio()
video_timestamp = []
# Task 1: Capture and display webcam frames
async def task1():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    out = save_video()
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read Videoframe.")
                break
            current_time  = time.time() - start_time 
            video_timestamp.append(time.time())
            out.write(frame)
            cv2.imshow('Webcam', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.005)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        out.release()

# Task 2: Record audio and save it as a WAV file
async def task2(executor):
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK ,input_device_index= device_index)
    print("Recording.....")
    frames = []
    overflow_count  = 0
    audio_timestamp = []
    def record_audio():
        nonlocal overflow_count
        start_time = time.time()

        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            current_time = time.time() - start_time
            audio_timestamp.append(time.time())
            # print(frames)
        except OSError as e:
            overflow_count+=1
            print("Audio stream overflow:{},(count :  {})".format(e , overflow_count))

    try:
        while True:
            await asyncio.get_event_loop().run_in_executor(executor, record_audio)
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        print("Recording stopped.")
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        if frames:
            with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))

        merage_audio_with_video(video_output_file, WAVE_OUTPUT_FILENAME, video_timestamp , audio_timestamp  ,'VideoFile.mp4')


def merage_audio_with_video(video_path , audio_path ,video_timestamp , audio_timestamp, output_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    audio_offset = audio_timestamp[0] - video_timestamp[0]
    synced_audio_clip = audio_clip.set_start(audio_offset)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_path , codec = 'libx264' , audio_codec  = 'aac')
    video_clip.close()
    audio_clip.close()
    final_clip.close()
    print(f"Merged Audio With Video is Saved IN {output_path}")







# Main function to run both tasks concurrently
async def main():
    with ThreadPoolExecutor() as executor:
        await asyncio.gather(task1(), task2(executor))

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Tasks terminated.")
