import os
import subprocess
import ffmpeg
import re
import numpy as np
import yt_dlp


def select_smallest_yt_video(url, min_height, temp_dir=""):
    # selects smallest format adhering to minimum height requirement
    # with both audio and video streams
    streams = parse_ytdlp_format(url, temp_dir=temp_dir)
    videos = []
    audios = []
    mixes = []
    for stream in streams:
        stream["FILESIZE"] = stream["FILESIZE"].replace("~", "")
        if "GiB" in stream["FILESIZE"]:
            stream["FILESIZE"] = float(stream["FILESIZE"].replace("GiB", "")) * 1000000
        elif "MiB" in stream["FILESIZE"]:
            stream["FILESIZE"] = float(stream["FILESIZE"].replace("MiB", "")) * 1000
        elif "KiB" in stream["FILESIZE"]:
            stream["FILESIZE"] = float(stream["FILESIZE"].replace("KiB", ""))
        else:
            continue
        
        if stream["RESOLUTION"] not in ("", "audio only"):
            stream["HEIGHT"] = int(stream["RESOLUTION"][stream["RESOLUTION"].find("x")+1:])
        else:
            stream["HEIGHT"] = None

        if stream["ACODEC"] == "video only":
            videos.append(stream)
        elif stream["VCODEC"] == "audio only":
            audios.append(stream)
        elif stream["VCODEC"] not in ("", "images") and stream["ACODEC"] != "":
            mixes.append(stream)

    # now choose
    video_or_mix = videos + mixes
    large_ones = [item for item in video_or_mix if item["HEIGHT"] >= min_height]
    
    if large_ones != []:
        # if there are large video streams, take the smallest file
        large_ones = sorted(large_ones, key=lambda x:x["FILESIZE"])
        selected_video = large_ones[0]
    elif video_or_mix != []:
        # take best resolution
        video_or_mix = sorted(video_or_mix, key=lambda x:x["HEIGHT"])
        video_or_mix.reverse()
        max_height = video_or_mix[0]["HEIGHT"]
        video_or_mix = [video for video in video_or_mix if video["HEIGHT"] == max_height]
        # take smallest file
        video_or_mix = sorted(video_or_mix, key=lambda x:x["FILESIZE"])
        selected_video = video_or_mix[0]
    else:
        selected_video = None
    if selected_video != None:
        if selected_video["ACODEC"] in ("" or "video only"):
            # no audio, need to get it also (smallest file)
            audios = sorted(audios, key=lambda x:x["FILESIZE"])
            selected_audio = audios[0]
        else:
            selected_audio = None
    else:
        selected_audio = None

    return {"video": selected_video, "audio": selected_audio}


def parse_ytdlp_format(url, temp_dir="../data/temp"):
    # parse the output of ytdlp format
    
    id = url.replace("https://www.youtube.com/watch?v=", "")
    info_output_path = os.path.join(temp_dir, id + ".txt")

    if "http" not in url:
        url = f"\"{url}\""  # some youtube ids require quotes

    keys_l = ["ID", "EXT", "RESOLUTION"]    # left aligned
    keys_r = ["FILESIZE"]   # right aligned
    keys_s = ["VCODEC", "ACODEC"]     # special, bit problematic

    # Get available formats
    args = ["yt-dlp", "-F", "--", url, ">", info_output_path]

    subprocess.call(" ".join(args), shell=True)
    with open(info_output_path, "r") as f:
        text = f.readlines()
        
        list_started = False
        videos = []
        for line in text:
            if line[:len(keys_l[0])] == keys_l[0]:
                # header, find indices
                inds = {}
                for key in keys_l:
                    a = line.find(key)
                    b = a + len(key)
                    while line[b] == " ":
                        b += 1
                    inds[key] = (a, b)

                for key in keys_r:
                    a = line.find(key)
                    b = a + len(key)
                    while line[a] == " ":
                        a -= 1
                    inds[key] = (a, b)

                for key in keys_s:
                    a = line.find(key)
                    inds[key] = (a, None)

            if line[:3] == "---":
                list_started = True
                continue

            if list_started:
                video = {}
                for key in keys_l + keys_r:
                    val = line[inds[key][0]:inds[key][1]]
                    video[key] = val.strip()

                for key in keys_s:
                    val = line[inds[key][0]:]
                    val = val.replace(" only", "_only")
                    loc = val.find(" ")
                    val = val[:loc]
                    val = val.replace("_only", " only")
                    video[key] = val

                videos.append(video)
                
    os.remove(info_output_path)
    return videos



def video_to_midscenes(input_path,  threshold=0.27, write=False, output_dir=None, frame_size=None):
    # Returns frames in the middle of pair of scenecuts
    scenecut_times = ffmpeg_scene_detect(input_path, threshold=threshold)
    scenecut_times = [scene[0] for scene in scenecut_times]

    mid_scene_numbers, timestamps = scenecuts_to_median_frames(input_path, scenecut_times)

    if mid_scene_numbers == None:
        return None
    frames = get_frames_by_number(input_path, mid_scene_numbers, write=write, output_dir=output_dir, frame_size=frame_size)
    if not write:
        frames = frames.copy()  # make it writable
    return frames, timestamps

def ffmpeg_scene_detect(input_, threshold=0.3):
    # Returns timestamps of scenecuts
    commands_flat = f"ffmpeg -i {input_} -vsync vfr -vf select=scene -loglevel debug -f null /dev/null 2>&1 | grep scene:" 
    output = subprocess.run(commands_flat, shell=True, capture_output=True, text=True).stdout

    output = output.split("\n")
    scenes_times = []
    for line in output:
        if "scene" in line:
            line = line.split(" ")
            line = [item for item in line if ":" in item]
            line_dict = {}
            for item in line:
                key, val = item.split(":")[:2]
                if key in ("t", "scene"):
                    try:
                        val_found = float(val)
                    except:     # remove non-numeric characters
                        val_found = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", val)
                        if val_found == []:
                            continue
                        else:
                            val_found = float(val[0])

                    line_dict[key] = val_found

            if "t" in line_dict.keys() and "scene" in line_dict.keys():
                if line_dict["scene"] > threshold:
                    scenes_times.append((line_dict["t"], line_dict["scene"]))

    return scenes_times


def get_video_fps(video_path):
    try:
        probe = ffmpeg.probe(video_path, cmd='ffprobe', v='error', select_streams='v:0', count_packets=None, show_entries='stream=nb_read_packets,avg_frame_rate')
        fps = round(eval(probe["streams"][0]["avg_frame_rate"]), 2)
        return fps
    except:
        return None
    

def scenecuts_to_median_frames(input_, scenecut_times):
    # Takes scenecut timestamps and returns the numbers of frames
    # that are in the middle of pairs of scenecuts

    # get FPS, duration, number of frames
    try:
        probe = ffmpeg.probe(input_, cmd='ffprobe', v='error', select_streams='v:0', count_packets=None, show_entries='stream=nb_read_packets,avg_frame_rate')
    except:
        return None
    fps = round(eval(probe["streams"][0]["avg_frame_rate"]), 2)
    n_frames = int(probe["streams"][0]["nb_read_packets"]) - 1   # 0 indexing

    # convert times to frames
    scenecut_times.insert(0, 0)
    if "duration" in probe["format"].keys():
        duration = float(probe["format"]["duration"])
        scenecut_times.append(duration)
    median_seconds = [((scenecut_times[i] + scenecut_times[i+1]) / 2) \
        for i in range(len(scenecut_times)-1)]
    median_frames = [min(int(round(item*fps)), n_frames) for item in median_seconds]
    median_frames = sorted(list(set(median_frames)))    # get unique

    # create a list of timestamps
    timestamps = []
    for item in median_frames:
        seconds = item / fps
        minute, second = divmod(seconds, 60)
        minute = int(minute)
        second, milisecond = divmod(second, 1)
        second = int(second)
        millisecond = int(round(milisecond * 1000))
        timestamps.append(f"{minute:02d}:{second:02d}.{millisecond:03d}")
    return median_frames, timestamps


def get_frames_by_number(input_, frame_numbers, write=False, output_dir=None, frame_size=None):
    # Returns the sequence of frames as numpy array, given video and frame numbers
    probe = ffmpeg.probe(input_)
    video_stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"]
    video_stream = video_stream[0]
    h = video_stream["height"]
    w = video_stream["width"]

    filter_string = "'"
    for frame in frame_numbers:
        filter_string += r'eq(n\,{})+'.format(frame)
    filter_string = filter_string[:-1] + "'"     # replace final + with '

    if write:
        assert output_dir != None, "Output directory is required."
        if frame_size != None:
            commands = [
                "ffmpeg", "-i", input_,  "-y",
                "-vf", "select=" + filter_string + ",scale=-1:" + str(frame_size),
                "-vsync", "0",        
                "-pix_fmt", "rgb24",
                f"{output_dir}/%03d.png"
            ]
        else:
            commands = [
                "ffmpeg", "-i", input_,  "-y",
                "-vf", "select=" + filter_string,
                "-vsync", "0",        
                "-pix_fmt", "rgb24",
                f"{output_dir}/%03d.png"
            ]
        subprocess.run(commands)
        return
    else:
        commands = [
            "ffmpeg", "-i", input_,  "-y",
            "-vf", "select=" + filter_string,
            "-vsync", "0",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo", 
            "pipe:1"
        ]
        ret = subprocess.run(commands,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            bufsize=10**8
                            )
        output = ret.stdout
        output = np.frombuffer(output, np.uint8).reshape(-1, h, w, 3)

        return output


def extract_audio(video_path):
    probe = ffmpeg.probe(video_path)
    audio_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    if audio_info:
        audio_sample_rate = int(audio_info['sample_rate'])  # Get original audio sampling rate
        num_audio_channels = int(audio_info.get('channels', 1))
    else:
        audio_sample_rate = None
        num_audio_channels = None

    if audio_info:      # if available
        bits = 32
        dtype_ = eval(f"np.int{bits}")
        max_ = np.iinfo(dtype_).max
        format = f"s{bits}le"

        out_audio, _ = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format=format)
            .run(capture_stdout=True, quiet=True)
        )
        # Convert the audio to a numpy array
        audio_array = np.frombuffer(out_audio, dtype=dtype_).astype(np.float32)
        audio_array = audio_array.reshape((-1, num_audio_channels)).T / max_
    else:
        audio_array = None

    return audio_array, audio_sample_rate

def download_youtube(url, target_dir='.', size=360):
    target_dir = str(target_dir)
    # Downloads Youtube video
    ydl_opts = {
        'outtmpl': f'{target_dir}/%(title)s.%(ext)s',
        'noplaylist': True,
        'overwrites': True,
        'format': 'bestaudio/best',  # Default format selection
        'format_sort': 'height',     # Sort formats by height
        'format': f'best[height<={size}]/best',  # Select the best format with height <= 360p
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        # ext = info_dict.get('ext', 'mp4')  # Default to mp4 if extension not foundm
        filename = ydl.prepare_filename(info_dict)

    return filename