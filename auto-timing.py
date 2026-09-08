#!/usr/bin/env python3
"""
Automated audio timing-track generator
--------------------------------------
Produces groups of Audacity-compatible label tracks from an audio file:
  - drum / percussive onsets
  - general / melodic onsets
  - beats
  - word starts (from provided lyrics or auto-transcription)

Primary platform: Linux. Windows (native or WSL) also supported.

setup:
# Make executable (Linux)
chmod +x audio_timing_tracks.py
#nvidia-smi
#apt install nvidia-utils-580 
#CUDA lib error:
pip uninstall -y torch torchaudio torchvision torio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
#test:
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
#then re-run this script

Usage:
    python audio_timing_tracks.py path/to/song.mp3
    python audio_timing_tracks.py path/to/song.mp3 --lyrics lyrics.txt

# First run – will offer to install missing packages and ask questions
python audio_timing_tracks.py path/to/song.mp3
# With lyrics file
python audio_timing_tracks.py path/to/song.mp3 --lyrics path/to/lyrics.txt
# Subsequent runs on the same file reuse answers from song.json
python audio_timing_tracks.py path/to/song.mp3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
#import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from my_utils import ANSI, good, error, warn, info, debug, memo, relpath, elapsed

APP_VERSION = "0.1.0"
#APP_NAME = "LightBench"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #Path(__file__).parent

#dbg(os.getcwd())
debug("basedir", BASE_DIR)


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES = [
    "numpy",
    "soundfile",
    "librosa",
    "demucs",
    "faster-whisper",
    "torch",          # required by demucs / faster-whisper
    "torchaudio",
]

OPTIONAL_PACKAGES = [
    "essentia",        # genre classification (may need special install)
]

def check_and_offer_install() -> None:
    """Check for missing packages and offer to install them."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg.replace("-", "_").split("[")[0])
        except ImportError:
            missing.append(pkg)

    if not missing:
        good("All core dependencies are present.")
        return

    warn("\nMissing required packages:")
    for p in missing:
        warn(f"  - {p}")

    answer = input("\nInstall them now with pip? [y/N]: ").strip().lower()
    if answer in ("y", "yes"):
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + missing
        debug("Running:", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
            good("Installation finished. Please re-run the script.")
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            error("Installation failed:", e)
            sys.exit(1)
    else:
        error("Cannot continue without the required packages.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Config persistence (sidecar JSON)
# ---------------------------------------------------------------------------

def config_path_for(audio_path: Path) -> Path:
#    return audio_path.with_suffix(".json")
    return audio_path.parent / "auto_seq" / f"{audio_path.stem}.json"

def load_config(audio_path: Path) -> Dict[str, Any]:
    cfg_file = config_path_for(audio_path)
    if cfg_file.is_file() and ask_bool({}, "use_answers", f"{ANSI['memo']}Use previous answers?", True):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            info(f"Loaded previous settings from {relpath(cfg_file, audio_path)}")
            return data
        except Exception as e:
            warn(f"Warning: could not read {relpath(cfg_file, audio_path)}: {e}")
    return {}

def save_config(audio_path: Path, config: Dict[str, Any]) -> None:
    cfg_file = config_path_for(audio_path)
    try:
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        info(f"Saved settings to {relpath(cfg_file, audio_path)}")
    except Exception as e:
        error(f"Warning: could not write config: {e}")

def ask(config: Dict[str, Any], key: str, prompt: str, default: str = "") -> str:
    """Ask user only if the key is not already in the config."""
    if key in config and config[key] is not None:
        debug(f"{prompt} [saved: {config[key]}]")
        return str(config[key])

    if default:
        full_prompt = f"{prompt} [{default}]: {ANSI['reset']}"
    else:
        full_prompt = f"{prompt}: {ANSI['reset']}"

    value = input(full_prompt).strip()
    if not value and default:
        value = default
    config[key] = value
    return value

def ask_bool(config: Dict[str, Any], key: str, prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    if key in config:
        debug(f"{prompt} [saved: {config[key]}]")
        return bool(config[key])

    answer = input(f"{prompt} [{default_str}]: {ANSI['reset']}").strip().lower()
    if not answer:
        value = default
    else:
        value = answer in ("y", "yes", "1", "true")
    config[key] = value
    return value

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    debug("  $", " ".join(cmd), depth=+1)
    return subprocess.run(cmd, check=check, capture_output=False)

def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        error("ERROR: ffmpeg not found in PATH.")
        error("Install it with your package manager (apt install ffmpeg, etc.)")
        sys.exit(1)

def write_audacity_labels(
    times: List[float],
    outfile: Path,
    labels: Optional[List[str]] = None,
    as_regions: bool = False,
    durations: Optional[List[float]] = None,
    basedir: Path = None,
) -> None:
    """Write a standard Audacity label track file."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        for i, t in enumerate(times):
            if as_regions and durations is not None:
                end = t + durations[i]
            else:
                end = t
            lab = labels[i] if labels and i < len(labels) else ""
            f.write(f"{t:.6f}\t{end:.6f}\t{lab}\n")
    good(f"  Wrote {len(times)} labels → {relpath(outfile, basedir)}")

# ---------------------------------------------------------------------------
# Core processing steps
# ---------------------------------------------------------------------------

def prepare_audio(audio_path: Path, work_dir: Path) -> Path:
    """Convert to mono 44.1 kHz WAV."""
    out = work_dir / "prepared.wav"
    if out.exists():
        return out
    debug("\n[1/7] Preparing audio …")
    run_cmd([
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ac", "1", "-ar", "44100",
        "-sample_fmt", "s16",
        str(out)
    ])
    return out

def separate_sources(prepared: Path, work_dir: Path, do_separate: bool) -> Dict[str, Path]:
    """Run Demucs and return paths to stems."""
    stems = {
        "mix": prepared,
        "vocals": prepared,
        "drums": prepared,
        "other": prepared,
    }
    if not do_separate:
        debug("\n[2/7] Skipping source separation.")
        return stems

    warn("\n[2/7] Running Demucs source separation (this may take a while) …")
    out_dir = work_dir / "demucs"
    # Use htdemucs (good quality / speed balance)
    run_cmd([
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",          # faster; change to full 4-stem if desired
        "-o", str(out_dir),
        str(prepared)
    ])

    # Demucs output layout: demucs/htdemucs/prepared/{vocals,no_vocals}.wav
    # or full model: drums, bass, other, vocals
    model_dir = next(out_dir.glob("*/*"), None)
    if model_dir and model_dir.is_dir():
        for name in ("vocals", "drums", "bass", "other", "no_vocals"):
            candidate = model_dir / f"{name}.wav"
            if candidate.exists():
                stems[name if name != "no_vocals" else "other"] = candidate
        if "drums" not in stems and "other" in stems:
            stems["drums"] = stems["other"]  # fallback
    good("  Stems ready.")
    return stems

def detect_onsets(audio_path: Path, sr: int = 44100) -> List[float]:
    """Simple robust onset detection with librosa."""
    import librosa
    import numpy as np

    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    # Prefer backtracking for tighter placement on the attack
    onsets = librosa.onset.onset_detect(
        y=y, sr=sr,
        units="time",
        backtrack=True,
        hop_length=512,
    )
    return sorted(float(t) for t in onsets)

def detect_beats(audio_path: Path, sr: int = 44100) -> Tuple[List[float], float]:
    """Beat tracking with librosa."""
    import librosa

    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    # tempo can be ndarray in newer librosa
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) else 120.0
    else:
        tempo = float(tempo)
    return [float(t) for t in beats], tempo

def transcribe_words(
    vocals_path: Path,
    language: str = "en",
    model_size: str = "medium",
) -> List[Dict[str, Any]]:
    """Return list of {word, start, end} using faster-whisper."""
    from faster_whisper import WhisperModel

    debug(f"  Loading Whisper model '{model_size}' …")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(vocals_path),
        word_timestamps=True,
        language=language if language else None,
        vad_filter=True,
    )

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word.strip(),
                    "start": float(w.start),
                    "end": float(w.end),
                })
    return words

def load_lyrics_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    # Simple normalisation – keep it readable
    return " ".join(text.split())

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Audacity timing label tracks from audio"
    )
    parser.add_argument("audio", type=Path, help="Input audio file (mp3, wav, …)")
    parser.add_argument("--lyrics", type=Path, default=None,
                        help="Optional plain-text lyrics file")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Working directory (default: <audio-dir>/auto_seq)")
    args = parser.parse_args()

    audio_path: Path = args.audio.resolve()
    if not audio_path.is_file():
        error(f"File not found: {audio_path}")
        sys.exit(1)

    info("=" * 60, srcline="")
    info("Audio Timing Track Generator")
    info("=" * 60, srcline="")

    # 0. Dependencies
    check_and_offer_install()
    ensure_ffmpeg()

    # 1. Config / user choices
    config = load_config(audio_path)

    work_dir = args.work_dir
    if work_dir is None:
#        work_dir = audio_path.with_name(audio_path.stem + "_timing")
        work_dir = audio_path.parent / "auto_seq"
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    debug(f"Working directory: {relpath(work_dir, audio_path)}")
    
    do_separate = ask_bool(config, "separate", f"{ANSI['memo']}Run source separation (Demucs)?", True)
    language = ask(config, "language", f"{ANSI['memo']}Language code for lyrics/transcription", "en")
    whisper_model = ask(config, "whisper_model",
                        f"{ANSI['memo']}Whisper model size (tiny/base/small/medium/large-v3)", "medium")
    gen_drums = ask_bool(config, "gen_drums", f"{ANSI['memo']}Generate drum/percussive onset track?", True)
    gen_onsets = ask_bool(config, "gen_onsets", f"{ANSI['memo']}Generate general onset track?", True)
    gen_beats = ask_bool(config, "gen_beats", f"{ANSI['memo']}Generate beat track?", True)
    gen_words = ask_bool(config, "gen_words", f"{ANSI['memo']}Generate word-start track?", True)

    lyrics_path = args.lyrics
    if lyrics_path is None and "lyrics_file" in config:
        lyrics_path = Path(config["lyrics_file"]) if config["lyrics_file"] else None
    if lyrics_path is not None:
        lyrics_path = lyrics_path.resolve()
        if not lyrics_path.is_file():
            error(f"File not found: {relpath(lyrics_path, audio_path)}")
            sys.exit(1)

    if gen_words and lyrics_path is None:
        has_lyrics = ask_bool(config, "has_lyrics",
                              f"{ANSI['memo']}Do you have a lyrics text file?", False)
        if has_lyrics:
            lyrics_str = ask(config, "lyrics_file", f"{ANSI['memo']}Path to lyrics .txt file")
            lyrics_path = Path(lyrics_str).expanduser().resolve()
            if not lyrics_path.is_file():
                warn(f"Lyrics file not found: {relpath(lyrics_path, audio_path)}")
                lyrics_path = None

    # Persist any new answers
    if lyrics_path:
        config["lyrics_file"] = str(lyrics_path)
    save_config(audio_path, config)

    # 2. Prepare
    prepared = prepare_audio(audio_path, work_dir)

    # 3. Separation
    stems = separate_sources(prepared, work_dir, do_separate)

    # 4. Onsets & beats
    label_dir = work_dir / "labels"
    label_dir.mkdir(exist_ok=True)

    if gen_drums:
        debug("\n[3/7] Detecting drum / percussive onsets …")
        drum_times = detect_onsets(stems.get("drums", prepared))
        write_audacity_labels(drum_times, label_dir / "drums.txt", basedir=work_dir)

    if gen_onsets:
        debug("\n[4/7] Detecting general / melodic onsets …")
        onset_times = detect_onsets(stems.get("other", prepared))
        write_audacity_labels(onset_times, label_dir / "onsets.txt", basedir=work_dir)

    if gen_beats:
        debug("\n[5/7] Detecting beats …")
        beat_times, tempo = detect_beats(prepared)
        debug(f"  Estimated tempo: {tempo:.1f} BPM")
        write_audacity_labels(beat_times, label_dir / "beats.txt", basedir=work_dir)
        config["tempo"] = tempo
        save_config(audio_path, config)

    # 5. Words
    if gen_words:
        debug("\n[6/7] Generating word timings …")
        vocals = stems.get("vocals", prepared)

        words = transcribe_words(vocals, language=language, model_size=whisper_model)

        if not words:
            warn("  No words detected.")
        else:
            # If user supplied lyrics, we still use the Whisper timestamps
            # (true forced alignment would require MFA / WhisperX extra setup).
            # We simply keep the detected words; user can edit later in Audacity.
            word_starts = [w["start"] for w in words]
            word_labels = [w["word"] for w in words]
            write_audacity_labels(
                word_starts,
                label_dir / "words.txt",
                labels=word_labels,
                basedir=work_dir,
            )

            # Also write a full-region version
            write_audacity_labels(
                [w["start"] for w in words],
                label_dir / "words_regions.txt",
                labels=[w["word"] for w in words],
                as_regions=True,
                durations=[w["end"] - w["start"] for w in words],
                basedir=work_dir,
            )

            # Optional: save raw transcript
            transcript_file = work_dir / "transcript.txt"
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(" ".join(w["word"] for w in words))
            info(f"  Transcript saved to {relpath(transcript_file, audio_path)}")

    # 6. Done
    memo("\n[7/7] Finished.")
    info(f"\nAudacity label files are in: {relpath(label_dir, audio_path)}")
    info("Import them via File → Import → Labels in Audacity.")
    info("Each .txt file becomes its own Label Track.")
    memo("\nRe-run on same audio file to reuse previous answers.")
    debug("elapsed:", elapsed())
    

if __name__ == "__main__":
    main()

#eof