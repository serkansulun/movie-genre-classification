import os
import subprocess
from time import time
import pandas as pd
from pathlib import Path
import preprocessing.src.video_utils as u_video
import utils as u

""" 
Download Movienet trailers hosted on YouTube.
Takes a long time. If you have lots of space,
you can remove the re-encoding part and save 
the videos in their original forms to save time.
"""
os.makedirs("preprocessing/output", exist_ok=True)
log = u.Logger("preprocessing/output/log.txt")

work_dir = Path('preprocessing/data/trailers')

video_dir = work_dir / "downloaded"
os.makedirs(video_dir, exist_ok=True)
fail_dir = os.path.join(work_dir / "failed")
os.makedirs(fail_dir, exist_ok=True)
temp_dir = work_dir / "temp"
os.makedirs(temp_dir, exist_ok=True)

min_height = 300
crf = 23    # 28 is smaller, but bad quality

metadata = pd.read_csv('preprocessing/data/labels/trailers_genres_clean.csv')
yt_ids = metadata['youtube_id'].tolist()

# remove already downloaded
downloaded = video_dir.glob('*')
downloaded = [item.stem for item in downloaded if item.suffix not in (".part", ".ytdl")]
n_total = len(yt_ids)
yt_ids = [yt_id for yt_id in yt_ids if yt_id not in downloaded]
yt_ids = sorted(yt_ids)
data = enumerate(yt_ids)    # for indexing

t0 = time()
n0 = len(os.listdir(video_dir))

def download_video(args):
    idx = args[0]
    youtube_id = args[1]
    youtube_trailer_url = 'https://www.youtube.com/watch?v=' + youtube_id
    
    if idx % 100 == 0:
        t1 = time()
        n1 = len(os.listdir(video_dir))
        hours_elapsed = (t1 - t0) / 3600.0
        videos_processed = n1 - n0
        if videos_processed > 0:
            videos_remaining = n_total - n1
            hours_remaining = hours_elapsed / videos_processed * videos_remaining
            seconds_per_video = hours_elapsed / videos_processed * 3600.0
            log(f"{videos_processed} downloaded, {videos_remaining} remain. "
                f"{hours_elapsed:.1f}h passed, {hours_remaining:.1f}h remain. "
                f"{seconds_per_video:.1f}s/vid")
            
    streams = u_video.select_smallest_yt_video(youtube_trailer_url, min_height, temp_dir=temp_dir)
    fail_output_path = os.path.join(fail_dir, youtube_id)
    if streams == None or streams["video"] == None:   # Write as fail
        with open(fail_output_path, "w") as f_out:
            f_out.write(youtube_trailer_url)
        return False
    else:
        video = streams["video"]
        audio = streams["audio"]

        video_extension = ".%(ext)s"    # determined by yt-dlp
        if video["ACODEC"] == "video only":
            stream_id = video["ID"] + "+" + audio["ID"]
        else:
            stream_id = video["ID"]
        temp_video_output_path = os.path.join(temp_dir, youtube_id + "_temp" + video_extension)
        # download video
        yt_args = (
            "yt-dlp "
            f"-f {stream_id} "
            "--force-overwrites "
            "--print filename --no-simulate "
            f"-o \"{temp_video_output_path}\" "
            f"\"{youtube_trailer_url}\""
        )

        # get full filename with correct extension
        temp_video_output_path_orig = subprocess.getoutput(yt_args)
        # remove if there are warning messages
        temp_dir_str = str(temp_dir)
        if temp_dir_str != temp_video_output_path_orig[:len(temp_dir_str)]:
            loc = temp_video_output_path_orig.find(str(temp_dir))
            temp_video_output_path = temp_video_output_path_orig[loc:]
        else:
            temp_video_output_path = temp_video_output_path_orig

        video_output_path = os.path.join(video_dir, youtube_id + ".mkv")

        # Scale and re-encode to save space

        ff_args = ["ffmpeg", "-i", temp_video_output_path, "-y", "-c:v", "libx265", "-crf", str(crf),
            "-vf", f"scale=-1:{min_height},pad=ceil(iw/2)*2:ceil(ih/2)*2", "-max_muxing_queue_size", "9999",
            "-ac", "1", "-c:a", "libopus", "-b:a", "16k", "-preset", "ultrafast", video_output_path]
        subprocess.call(ff_args)
        try:
            os.remove(temp_video_output_path)
        except:
            log("Failed to remove\n" + temp_video_output_path_orig + "\n" + temp_video_output_path)


if __name__ == "__main__":
    u.run_parallel(download_video, data, 'thread')
