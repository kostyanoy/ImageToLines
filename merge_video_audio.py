from moviepy.editor import VideoFileClip, AudioFileClip

video_path = "result/bad_apple/bad_apple.mp4"
audio_path = "images/bad_apple.mp4"
raw_audio_path = "result/bad_apple/bad_apple_audio.mp3"
result = "result/bad_apple/bad_apple_full.mp4"

video_clip = VideoFileClip(video_path)
audio_clip = AudioFileClip(raw_audio_path)

video_clip.audio = audio_clip
video_clip.write_videofile(result, fps=30)
