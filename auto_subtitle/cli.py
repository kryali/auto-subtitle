import os
import sys
import time
import logging
import ffmpeg
import torch
from types import SimpleNamespace
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import argparse
import warnings
import tempfile
import subprocess
from tqdm import tqdm
from .utils import filename, str2bool, write_srt

# Module-level logger
logger = logging.getLogger(__name__)


def format_duration(seconds):
    """Format duration in seconds to a human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


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
    setup_logging()
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--model", default="openai/whisper-small",
                        help="HuggingFace repo or path of the Whisper model to use (e.g. openai/whisper-large-v3)")
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
                        help="print ffmpeg command output (otherwise ffmpeg runs in quiet mode)")

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
        logger.error("The following input file(s) were not found:")
        for p in missing:
            logger.error(f"  {p}")
        logger.error("Aborting.")
        sys.exit(1)
    
    # All files exist; continue processing.
    os.makedirs(output_dir, exist_ok=True)

    # Log start time
    logger.info("🚀 Starting auto-subtitle processing")
    logger.info(f"📁 Processing {len(video_paths)} file(s) with model '{model_name}' (task: {task})")
    if language != "auto":
        logger.info(f"🌐 Source language: {language}")
    
    overall_start_time = time.time()

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        args["language"] = language
        
    # Keep track of the final language value that will be supplied to Whisper
    final_language = args.get("language", "auto")

    logger.info(f"Loading Whisper model '{model_name}' using transformers pipeline…")
    model_start = time.time()

    # Choose device for pipeline
    if device == "cuda":
        device_id = 0
    elif device == "cpu":
        device_id = -1
    else:  # auto
        device_id = 0 if torch.cuda.is_available() else -1

    model_kwargs = {}
    
    # Only use Flash Attention 2 if available and on GPU
    if device_id != -1 and is_flash_attn_2_available():
        logger.info("🚀 Using Flash Attention 2 for maximum speed")
        model_kwargs["attn_implementation"] = "flash_attention_2"
        # Force model to initialize directly on GPU for Flash Attention 2
        model_kwargs["device_map"] = "auto"
    elif device_id != -1:
        logger.info("⚠️  Flash Attention 2 not available, using standard attention (slower)")
    else:
        logger.info("💻 Using CPU inference with standard attention")

    # Create pipeline - don't use device parameter when device_map is specified
    pipeline_kwargs = {
        "model": model_name,
        "torch_dtype": torch.float16 if device_id != -1 else torch.float32,
        "model_kwargs": model_kwargs,
    }
    
    # Add device parameter only if we're not using device_map
    if "device_map" not in model_kwargs:
        pipeline_kwargs["device"] = device_id
    
    pipe = pipeline("automatic-speech-recognition", **pipeline_kwargs)

    model_end = time.time()
    logger.info(f"✓ Model loaded in {format_duration(model_end - model_start)}")

    # Audio extraction phase
    logger.info(f"📥 Starting audio extraction for {len(video_paths)} file(s)...")
    audio_start = time.time()
    audios = get_audio(video_paths, verbose=verbose)
    audio_end = time.time()
    logger.info(f"✓ Audio extraction phase completed in {format_duration(audio_end - audio_start)}")

    # Build a small wrapper so we cleanly forward the user-requested parameters to
    # WhisperModel.transcribe while keeping get_subtitles unaware of them.
    def transcribe_fn(audio_path: str):
        call_kwargs = {
            "return_timestamps": True,
            "chunk_length_s": 30,
            "batch_size": 24,
        }

        # Pass task and language through generate_kwargs for transformers pipeline
        generate_kwargs = {}
        if task == "translate":
            generate_kwargs["task"] = "translate"
        elif task == "transcribe":
            generate_kwargs["task"] = "transcribe"
        
        if language != "auto":
            generate_kwargs["language"] = final_language
            
        # Clear any conflicting forced_decoder_ids when using task parameter
        generate_kwargs["forced_decoder_ids"] = None
        
        # Ensure proper timestamp handling by setting return_timestamps in generate_kwargs as well
        generate_kwargs["return_timestamps"] = True
        
        if generate_kwargs:
            call_kwargs["generate_kwargs"] = generate_kwargs
            
        logger.info(f"Pipeline call kwargs: {call_kwargs}")

        outputs = pipe(audio_path, **call_kwargs)

        # Convert pipeline chunks to faster-whisper compatible segments
        segments = []
        skipped_segments = 0
        
        for ch in outputs.get("chunks", []):
            start, end = ch["timestamp"]
            text = ch["text"]
            
            # Skip segments with None timestamps entirely - don't create fake ones
            if start is None or end is None:
                skipped_segments += 1
                logger.warning(f"Skipping segment with missing timestamp: '{text.strip()}'")
                continue
                
            segments.append(type("Segment", (), {"start": start, "end": end, "text": text}))

        # Log if we had to skip segments
        if skipped_segments > 0:
            logger.warning(f"Skipped {skipped_segments} segments due to missing timestamps")

        # Calculate duration from segments, with fallback
        if segments and segments[-1].end is not None:
            duration = segments[-1].end
        else:
            # Fallback: estimate duration from audio file if possible
            try:
                # Try to get duration using ffmpeg-python (already available)
                probe = ffmpeg.probe(audio_path)
                duration = float(probe['streams'][0]['duration'])
            except:
                try:
                    # Fallback to librosa if available
                    import librosa
                    y, sr = librosa.load(audio_path, sr=None)
                    duration = len(y) / sr
                except:
                    # Final fallback: estimate based on number of segments
                    duration = len(segments) * 2.0 if segments else 0.0
                    logger.warning(f"Could not determine audio duration for {audio_path}, using estimated duration: {duration}s")
        
        info = SimpleNamespace(duration=duration)
        return segments, info

    # Subtitle generation phase
    logger.info(f"🎯 Starting subtitle generation for {len(audios)} file(s)...")
    subtitle_start = time.time()
    subtitles = get_subtitles(
        audios,
        output_srt or srt_only,
        output_dir,
        transcribe_fn,
    )
    subtitle_end = time.time()
    logger.info(f"✓ Subtitle generation phase completed in {format_duration(subtitle_end - subtitle_start)}")

    if srt_only:
        # Calculate and display timing summary for SRT-only mode
        overall_end_time = time.time()
        total_time = overall_end_time - overall_start_time
        
        logger.info("🎉 SRT generation completed successfully!")
        logger.info(f"📊 Total time: {format_duration(total_time)}")
        logger.info(f"   • Model loading: {format_duration(model_end - model_start)}")
        logger.info(f"   • Audio extraction: {format_duration(audio_end - audio_start)}")
        logger.info(f"   • Subtitle generation: {format_duration(subtitle_end - subtitle_start)}")
        return

    # Check available GPU encoders once
    available_gpu_encoders = check_gpu_encoders() if gpu_accel else []
    if gpu_accel and available_gpu_encoders:
        logger.info(f"GPU encoders available: {', '.join(available_gpu_encoders)}")
    elif gpu_accel and not available_gpu_encoders:
        logger.warning("GPU acceleration requested but no compatible encoders found, using CPU encoding")

    # Video encoding phase
    logger.info(f"🎬 Starting video encoding for {len(subtitles)} file(s)...")
    encoding_start = time.time()

    for path, srt_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}.mp4")

        logger.info(f"Adding subtitles to {filename(path)}...")
        start_time = time.time()

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
                logger.info("Using NVIDIA GPU acceleration for video encoding...")
            elif 'qsv' in available_gpu_encoders:
                output_kwargs.update({
                    'c:v': 'h264_qsv',
                    'preset': 'fast',
                    'global_quality': '23'
                })
                logger.info("Using Intel QSV GPU acceleration for video encoding...")
        else:
            # Use CPU encoding
            output_kwargs.update({
                'c:v': 'libx264',
                'preset': 'fast',
                'crf': '23'
            })
            if gpu_accel:
                logger.info("Using CPU encoding (no GPU encoders available)...")

        ffmpeg.concat(
            video.filter('subtitles', srt_path, force_style="OutlineColour=&H40000000,BorderStyle=3"), audio, v=1, a=1
        ).output(out_path, **output_kwargs).run(quiet=not verbose, overwrite_output=True)

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"✓ Subtitle overlay completed in {format_duration(duration)}")
        logger.info(f"Saved subtitled video to {os.path.abspath(out_path)}.")

    encoding_end = time.time()
    logger.info(f"✓ Video encoding phase completed in {format_duration(encoding_end - encoding_start)}")
    
    # Calculate and display overall timing
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    
    logger.info("🎉 All processing completed successfully!")
    logger.info(f"📊 Total time: {format_duration(total_time)}")
    logger.info(f"   • Model loading: {format_duration(model_end - model_start)}")
    logger.info(f"   • Audio extraction: {format_duration(audio_end - audio_start)}")
    logger.info(f"   • Subtitle generation: {format_duration(subtitle_end - subtitle_start)}")
    logger.info(f"   • Video encoding: {format_duration(encoding_end - encoding_start)}")


def get_audio(paths, *, verbose: bool = False):
    temp_dir = tempfile.gettempdir()

    audio_paths = {}

    for path in paths:
        logger.info(f"Extracting audio from {filename(path)}...")
        start_time = time.time()
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        try:
            ffmpeg.input(path).output(
                output_path,
                acodec="pcm_s16le", ac=1, ar="16k"
            ).run(quiet=not verbose, overwrite_output=True)
        except ffmpeg.Error as e:
            # Surface the underlying ffmpeg stderr so users know what went wrong.
            logger.error(f"Failed to extract audio from {path} (ffmpeg execution failed).")
            if e.stderr:
                try:
                    logger.error(e.stderr.decode("utf-8", errors="ignore"))
                except AttributeError:
                    # stderr might already be str depending on ffmpeg-python version.
                    logger.error(e.stderr)
            # Re-raise so the CLI exits with a non-zero status code.
            raise

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"✓ Audio extraction completed in {format_duration(duration)}")

        audio_paths[path] = output_path

    return audio_paths


def get_subtitles(audio_paths: list, output_srt: bool, output_dir: str, transcribe: callable):
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(path)}.srt")
        
        logger.info(
            f"Generating subtitles for {filename(path)}... This might take a while."
        )
        start_time = time.time()

        warnings.filterwarnings("ignore")
        segments, info = transcribe(audio_path)
        warnings.filterwarnings("default")

        # Get audio duration for progress estimation
        audio_duration = getattr(info, 'duration', None)
        
        # Convert segments iterator to list with progress bar
        logger.info("Processing segments...")
        segments_list = []
        
        # Create progress bar
        if audio_duration:
            # If we know the duration, show progress based on timestamp
            progress_bar = tqdm(
                desc="Processing", 
                unit=" segments",
                total=None,  # We don't know total segments upfront
                bar_format="{desc}: {n} segments | {elapsed} | {postfix}"
            )
        else:
            # Simple segment counter if no duration available
            progress_bar = tqdm(
                desc="Processing", 
                unit=" segments",
                bar_format="{desc}: {n} segments | {elapsed}"
            )

        try:
            last_end_time = 0
            for segment in segments:
                segments_list.append(segment)
                progress_bar.update(1)
                
                # Update progress description with time info if available
                if audio_duration and hasattr(segment, 'end'):
                    last_end_time = segment.end
                    progress_percentage = (last_end_time / audio_duration) * 100
                    progress_bar.set_postfix_str(
                        f"{last_end_time:.1f}s/{audio_duration:.1f}s ({progress_percentage:.1f}%)"
                    )
                    
        finally:
            progress_bar.close()

        logger.info(f"Processed {len(segments_list)} segments")

        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(segments_list, file=srt)

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"✓ Subtitle generation completed in {format_duration(duration)}")

        subtitles_path[path] = srt_path

    return subtitles_path


if __name__ == '__main__':
    main()
