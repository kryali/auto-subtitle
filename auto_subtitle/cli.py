import os
import sys
import ffmpeg
import faster_whisper
from faster_whisper import WhisperModel
import argparse
import warnings
import tempfile
import subprocess
from .utils import filename, str2bool, write_srt


def check_gpu_encoders():
    """Check which GPU encoders are available on the system"""
    available_encoders = []
    
    try:
        # Check ffmpeg encoders
        result = subprocess.run(['ffmpeg', '-encoders'], 
                              capture_output=True, text=True, timeout=10)
        encoders_output = result.stdout
        
        if 'h264_nvenc' in encoders_output:
            available_encoders.append('nvenc')
        if 'h264_qsv' in encoders_output:
            available_encoders.append('qsv')
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg not available or other error
        pass
    
    return available_encoders


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--model", default="small",
                        choices=faster_whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"], help="device to use for inference (auto, cpu, or cuda)")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--gpu_accel", type=str2bool, default=True,
                        help="whether to use GPU acceleration for video encoding (if available)")
    parser.add_argument("--output_srt", type=str2bool, default=False,
                        help="whether to output the .srt file along with the video files")
    parser.add_argument("--srt_only", type=str2bool, default=False,
                        help="only generate the .srt file and not create overlayed video")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], 
    help="What is the origin language of the video? If unset, it is detected automatically.")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    task: str = args.pop("task")
    language: str = args.pop("language")
    device: str = args.pop("device")
    gpu_accel: bool = args.pop("gpu_accel")
    verbose: bool = args.pop("verbose")
    
    # Pre-flight: make sure every supplied video file is accessible before doing any heavy work.
    video_paths = args.pop("video")
    missing = [p for p in video_paths if not os.path.isfile(p)]
    if missing:
        print("The following input file(s) were not found:")
        for p in missing:
            print(f"  {p}")
        print("Aborting.")
        sys.exit(1)
    
    # All files exist; continue processing.
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        args["language"] = language
        
    # Keep track of the final language value that will be supplied to Whisper
    final_language = args.get("language", "auto")

    model = WhisperModel(model_name, device=device)
    audios = get_audio(video_paths, verbose=verbose)

    # Build a small wrapper so we cleanly forward the user-requested parameters to
    # WhisperModel.transcribe while keeping get_subtitles unaware of them.
    def transcribe_fn(audio_path: str):
        kwargs = {"task": task}
        # Only pass language if the user gave an explicit non-auto value.  For
        # translation, Whisper works fine without forcing the source language,
        # and some versions behave oddly when both options are combined.
        if language != "auto":
            kwargs["language"] = final_language
        return model.transcribe(audio_path, **kwargs)

    subtitles = get_subtitles(
        audios,
        output_srt or srt_only,
        output_dir,
        transcribe_fn,
    )

    if srt_only:
        return

    # Check available GPU encoders once
    available_gpu_encoders = check_gpu_encoders() if gpu_accel else []
    if gpu_accel and available_gpu_encoders and verbose:
        print(f"GPU encoders available: {', '.join(available_gpu_encoders)}")
    elif gpu_accel and not available_gpu_encoders:
        print("GPU acceleration requested but no compatible encoders found, using CPU encoding")

    for path, srt_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}.mp4")

        print(f"Adding subtitles to {filename(path)}...")

        video = ffmpeg.input(path)
        audio = video.audio

        # Configure GPU acceleration if available
        output_kwargs = {}
        if gpu_accel and available_gpu_encoders:
            if 'nvenc' in available_gpu_encoders:
                output_kwargs.update({
                    'c:v': 'h264_nvenc',  # Use NVIDIA hardware encoder
                    'preset': 'p4',       # Fast preset for NVENC
                    'cq': '23'           # Constant quality
                })
                if verbose:
                    print("Using NVIDIA GPU acceleration for video encoding...")
            elif 'qsv' in available_gpu_encoders:
                output_kwargs.update({
                    'c:v': 'h264_qsv',
                    'preset': 'fast',
                    'global_quality': '23'
                })
                if verbose:
                    print("Using Intel QSV GPU acceleration for video encoding...")
        else:
            # Use CPU encoding
            output_kwargs.update({
                'c:v': 'libx264',
                'preset': 'fast',
                'crf': '23'
            })
            if verbose and gpu_accel:
                print("Using CPU encoding (no GPU encoders available)...")

        ffmpeg.concat(
            video.filter('subtitles', srt_path, force_style="OutlineColour=&H40000000,BorderStyle=3"), audio, v=1, a=1
        ).output(out_path, **output_kwargs).run(quiet=not verbose, overwrite_output=True)

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")


def get_audio(paths, *, verbose: bool = False):
    temp_dir = tempfile.gettempdir()

    audio_paths = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        try:
            ffmpeg.input(path).output(
                output_path,
                acodec="pcm_s16le", ac=1, ar="16k"
            ).run(quiet=not verbose, overwrite_output=True)
        except ffmpeg.Error as e:
            # Surface the underlying ffmpeg stderr so users know what went wrong.
            print(f"Failed to extract audio from {path} (ffmpeg execution failed).")
            if e.stderr:
                try:
                    print(e.stderr.decode("utf-8", errors="ignore"))
                except AttributeError:
                    # stderr might already be str depending on ffmpeg-python version.
                    print(e.stderr)
            # Re-raise so the CLI exits with a non-zero status code.
            raise

        audio_paths[path] = output_path

    return audio_paths


def get_subtitles(audio_paths: list, output_srt: bool, output_dir: str, transcribe: callable):
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(path)}.srt")
        
        print(
            f"Generating subtitles for {filename(path)}... This might take a while."
        )

        warnings.filterwarnings("ignore")
        segments, _ = transcribe(audio_path)
        warnings.filterwarnings("default")

        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(segments, file=srt)

        subtitles_path[path] = srt_path

    return subtitles_path


if __name__ == '__main__':
    main()
