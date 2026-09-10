#!/usr/bin/env python3
"""
Automated audio timing-track generator (memory-aware + vocals + features + genre/palette)
-----------------------------------------------------------------------------------------
Produces Audacity-compatible label tracks + genre guess + emotion-based color palette.
  - drum / percussive onsets
  - general / melodic onsets
  - beats
  - word starts / regions
  - contiguous vocal activity regions
  - structural sections + novelty events (features track)
  - Detects whether the audio is likely music
  - Estimates genre from audio features (+ optional lyric keywords)
  - Estimates predominant emotion
  - Suggests a 3- or 4-color palette (primary color = main emotion)
  - --force-music flag to force genre/palette analysis even if the
    music-detection heuristic fails.

Designed to run on machines with as little as ~4 GB free RAM (uses more when available).

Primary platform: Linux. Windows (native or WSL) also supported.

setup:
# Make executable (Linux)
chmod +x audio_timing_tracks.py
#nvidia-smi
#apt install nvidia-utils-580 
#CUDA lib error:
pip uninstall -y torch torchaudio torchvision torio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install whisperx
# Optional but recommended
   pip uninstall tensorflow tensorflow-cpu tensorflow-gpu -y
   pip install whisperx --force-reinstall
#test:
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
#then re-run this script

Usage:
    python audio_timing_tracks.py path/to/song.mp3
    python audio_timing_tracks.py path/to/song.mp3 --lyrics lyrics.txt
    python auto-timing.py path/to/song.mp3 --force-music
# Normal run – uses cache when possible
    python auto-timing.py song.mp3
# Force everything to be recomputed
    python auto-timing.py song.mp3 --force

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
#Force TensorFlow and CUDA to stay completely out of the way before WhisperX is imported:
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # hide all GPUs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"         # silence TF
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from my_utils import ANSI, good, error, warn, info, debug, memo, relpath, elapsed

APP_VERSION = "0.3.1"
#APP_NAME = "LightBench"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #Path(__file__).parent
NSTEPS = 9
#debug(os.getcwd())
debug("basedir", BASE_DIR)


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES = [
    "numpy",
    "soundfile",
    "librosa",
    "demucs",
#    "faster-whisper",
#    "whisperx",          # ← replaces faster-whisper
    "torch",          # required by demucs / faster-whisper
    "torchaudio",
]

# whisperx is strongly preferred; faster-whisper is the fallback
PREFERRED_PACKAGES = ["whisperx"]
FALLBACK_PACKAGES = ["faster-whisper"]

OPTIONAL_PACKAGES = [
    "essentia",        # genre classification (may need special install)
    "psutil",          # for available-RAM detection
]

def check_and_offer_install() -> None:
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg.replace("-", "_").split("[")[0])
        except ImportError:
            missing.append(pkg)

    # Try to detect whisperx / faster-whisper
    has_whisperx = False
    has_faster = False
    try:
        import whisperx
        has_whisperx = True
    except ImportError:
        pass
    try:
        import faster_whisper
        has_faster = True
    except ImportError:
        pass

    if not has_whisperx: #and not has_faster:
        missing.append("whisperx (recommended) or faster-whisper")

    if not missing:
        if has_whisperx:
            good("WhisperX available – will use high-quality alignment.")
        else:
            warn("WhisperX not found – falling back to faster-whisper (weaker timings).")
        return

    warn("\nMissing packages:")
    for p in missing:
        warn(f"  - {p}")

    answer = input(f"\n{ANSI['memo']}Install recommended packages now with pip? [y/N]: ").strip().lower()
    if answer in ("y", "yes"):
        to_install = REQUIRED_PACKAGES + ["whisperx"]
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + to_install
        debug("Running:", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
            good("Installation finished. Please re-run the script.")
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            error("Installation failed:", e)
            sys.exit(1)
    else:
        error("Cannot continue without at least one transcription backend.")
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
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
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

    full_prompt = f"{prompt} [{default}]: {ANSI['reset']}" if default else f"{prompt}: {ANSI['reset']}"
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
    value = default if not answer else answer in ("y", "yes", "1", "true")
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
            end = t + durations[i] if as_regions and durations is not None else t
            lab = labels[i] if labels and i < len(labels) else ""
            f.write(f"{t:.6f}\t{end:.6f}\t{lab}\n")
    good(f"  Wrote {len(times)} labels → {relpath(outfile, basedir)}")

def available_ram_gb() -> float:
    """Best-effort available RAM in GB. Returns a conservative default if psutil missing."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return 4.0   # assume low-RAM environment

def load_audio_safe(
    path: Path,
    sr: int = 44100,
    mono: bool = True,
    dtype="float32",
    max_duration: Optional[float] = None,
) -> Tuple[Any, int]:
    """
    Memory-conscious loader.
    Uses float32, optionally limits duration, and warns on very large files.
    """
    import librosa
    import numpy as np
    import soundfile as sf

    # Quick size estimate
    try:
        info = sf.info(str(path))
        duration = info.duration
        channels = info.channels
        est_bytes = duration * sr * (1 if mono else channels) * 4  # float32
        est_gb = est_bytes / (1024 ** 3)
        free = available_ram_gb()
        if est_gb > free * 0.6:
            warn(f"  Audio may use ~{est_gb:.2f} GB; only {free:.1f} GB free → conservative mode.")
    except Exception:
#        duration = None
        pass
    # Actual load – librosa handles resampling & mono
    y, sr_out = librosa.load(
        str(path),
        sr=sr,
        mono=mono,
        dtype=np.float32 if dtype == "float32" else None,
        duration=max_duration,
    )
    return y, sr_out

def is_up_to_date(
    output: Path | list[Path],
    *inputs: Path,
    version: str = "1",
    force: bool = False,
) -> bool:
    """
    Return True only if:
      - force is False
      - all output files exist
      - every output is newer than all inputs
      - the companion .version file matches the supplied logic version
    """
    if force:
        return False

    outputs = [output] if isinstance(output, Path) else list(output)

    for out in outputs:
        if not out.exists():
            return False
        version_file = out.with_suffix(out.suffix + ".version")
        if not version_file.exists():
            return False
        try:
            if version_file.read_text(encoding="utf-8").strip() != version:
                return False
        except Exception:
            return False

    try:
        newest_input = max(inp.stat().st_mtime for inp in inputs if inp.exists())
    except ValueError:
        return False

    for out in outputs:
        if out.stat().st_mtime < newest_input:
            return False

    return True

def mark_cache(output: Path | list[Path], version: str) -> None:
    """Write a small .version sidecar so future runs know the logic version."""
    outputs = [output] if isinstance(output, Path) else list(output)
    for out in outputs:
        version_file = out.with_suffix(out.suffix + ".version")
        version_file.write_text(version, encoding="utf-8")

# ---------------------------------------------------------------------------
# Core audio processing (prepare / separate / onsets / beats / words)
# ---------------------------------------------------------------------------

def prepare_audio(audio_path: Path, work_dir: Path, force: bool = False) -> Path:
    PREPARE_VERSION = "1"  # Bump when step logic changes
    """Convert to mono 44.1 kHz WAV (16-bit is fine for analysis)."""
    out = work_dir / "prepared.wav"
#    if out.exists():
    if is_up_to_date(out, audio_path, version=PREPARE_VERSION, force=force):
        debug("  [cache] prepared.wav is up-to-date → skipping")
        return out
    debug(f"\n[1/{NSTEPS}] Preparing audio …")
    run_cmd([
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ac", "1", "-ar", "44100",
        "-sample_fmt", "s16",
        str(out)
    ])
    mark_cache(out, PREPARE_VERSION)
    return out

def separate_sources(prepared: Path, work_dir: Path, do_separate: bool, force: bool = False) -> Dict[str, Path]:
    SEPARATE_VERSION = "1"  # Bump when step logic changes
    """Run Demucs (two-stem for lower RAM) and return paths to stems."""
    stems = {
        "mix": prepared,
        "vocals": prepared,
        "drums": prepared,
        "other": prepared,
    }
    if not do_separate:
        debug("\n[2/9] Skipping source separation.")
        return stems

    out_dir = work_dir / "demucs"
    # Typical Demucs layout
    possible_vocals = list(out_dir.glob("*/*/vocals.wav")) + list(out_dir.glob("*/*/*/vocals.wav"))
    
    if possible_vocals and is_up_to_date(possible_vocals[0], prepared, version=SEPARATE_VERSION, force=force):
        debug("  [cache] Demucs stems are up-to-date → skipping separation")
        model_dir = possible_vocals[0].parent
    else:
        warn(f"\n[2/{NSTEPS}] Running Demucs 2-stem source separation (this may take a while / use RAM) …")
        # Use htdemucs (good quality / speed balance)
        # two-stems is significantly lighter than full 4-stem
        run_cmd([
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",          # faster; change to full 4-stem if desired
            "-o", str(out_dir),
            str(prepared)
        ])
        # Demucs output layout: demucs/htdemucs/prepared/{vocals,no_vocals}.wav
        # or full model: drums, bass, other, vocals
        model_dir = next(out_dir.glob("*/*"), None)
        if model_dir:
            # mark the main stem as the cache representative
            mark_cache(model_dir / "vocals.wav", SEPARATE_VERSION)

    if model_dir and model_dir.is_dir():
        for name in ("vocals", "drums", "bass", "other", "no_vocals"):
            candidate = model_dir / f"{name}.wav"
            if candidate.exists():
                stems[name if name != "no_vocals" else "other"] = candidate
        if "drums" not in stems and "other" in stems:
            stems["drums"] = stems["other"]
    good("  Stems ready.")
    return stems

def detect_onsets(audio_path: Path, sr: int = 44100) -> List[float]:
    """Simple robust onset detection with librosa."""
    """Onset detection – loads with float32."""
    import librosa
#    import numpy as np
#    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    y, _ = load_audio_safe(audio_path, sr=sr)
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
#    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    y, _ = load_audio_safe(audio_path, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    # tempo can be ndarray in newer librosa
    tempo = float(tempo[0]) if hasattr(tempo, "__len__") and len(tempo) else float(tempo)
    return [float(t) for t in beats], tempo

def transcribe_words(
    vocals_path: Path,
    language: str = "en",
    model_size: str = "medium",
    genre: str = "unknown",
    user_lyrics: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Prefer WhisperX (with forced alignment). Fall back to faster-whisper.
    Tune alignment parameters according to detected music style.
    If user_lyrics is provided, prefer those words when it makes sense.
    """
    words: List[Dict[str, Any]] = []
    used_backend = "none"

    # ---------- 1. Try WhisperX first ----------
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Extra safety
        import whisperx
        import torch

#        compute_type = "default"   # or "float32"

        device = "cpu"
        # int8 is sometimes unstable on CPU with certain WhisperX builds
        compute_type = "int8"          # try this first
        # compute_type = "default"     # safer fallback if int8 still crashes
        # compute_type = "float32"     # slowest but most compatible

        free = available_ram_gb()
        batch_size = 4 if free > 6 else 2

        if free < 5.0 and model_size in ("medium", "large-v2", "large-v3"):
            model_size = "small"
            warn(f"  Low RAM → using WhisperX model '{model_size}'")

        debug(f"  Loading WhisperX '{model_size}' on {device} ({compute_type}) ({free:.1f} free) …")
    
        model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=compute_type,
            language=language if language else None,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        free = available_ram_gb()
        batch_size = 8 if free > 8 else 4 if free > 5 else 2

        if free < 5.0 and model_size in ("medium", "large-v2", "large-v3"):
            model_size = "small"
            warn(f"  Low RAM → using WhisperX model '{model_size}'")

        debug(f"  Loading WhisperX '{model_size}' on {device} ({compute_type}) …")
        model = whisperx.load_model(
            model_size, device, compute_type=compute_type,
            language=language if language else None,
        )

        audio = whisperx.load_audio(str(vocals_path))
        result = model.transcribe(audio, batch_size=batch_size, language=language or None)

        # Tune alignment for music style
        # More relaxed settings for electronic / ambient / reverb-heavy genres
        align_kwargs = {}
        if any(g in genre.lower() for g in ("electronic", "dance", "ambient", "chill", "shoegaze")):
            debug("  Tuning alignment for electronic / ambient style (more tolerant)")
            # WhisperX doesn't expose every internal tolerance, but we can
            # influence behaviour via model choice and post-filtering later
            align_kwargs["return_char_alignments"] = False
        else:
            debug("  Using standard alignment settings")

        debug("  Running forced alignment …")
        model_a, metadata = whisperx.load_align_model(
            language_code=result.get("language", language or "en"),
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                if w.get("start") is None or w.get("end") is None:
                    continue
                text = (w.get("word") or "").strip()
                if not text:
                    continue
                words.append({
                    "word": text,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })

        used_backend = "whisperx"
        del model, model_a
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        warn(f"  WhisperX failed ({e}). Falling back to faster-whisper …")

    # ---------- 2. Fallback: faster-whisper ----------
    if not words:
        try:
            from faster_whisper import WhisperModel

            debug(f"  Loading faster-whisper '{model_size}' (int8) …")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments, _ = model.transcribe(
                str(vocals_path),
                word_timestamps=True,
                language=language if language else None,
                vad_filter=True,
            )
            for segment in segments:
                if segment.words:
                    for w in segment.words:
                        words.append({
                            "word": w.word.strip(),
                            "start": float(w.start),
                            "end": float(w.end),
                        })
            used_backend = "faster-whisper"
        except Exception as e:
            error(f"  faster-whisper also failed: {e}")
            return []

    debug(f"  Transcription backend used: {used_backend}  ({len(words)} words)")

    # ---------- 3. Prefer user-supplied lyrics when it makes sense ----------
    if user_lyrics and words:
        user_words = [w.strip() for w in user_lyrics.split() if w.strip()]
        asr_text = " ".join(w["word"] for w in words).lower()
        user_text = " ".join(user_words).lower()

        # Simple similarity check (length + rough token overlap)
        len_ratio = len(user_words) / max(len(words), 1)
        if 0.6 < len_ratio < 1.6:
            # Prefer user lyrics text while keeping ASR timings
            # (best practical compromise without a full forced-alignment pipeline)
            debug("  User lyrics look compatible → using them for word labels")
            # Align lengths as best we can
            final = []
            n = min(len(words), len(user_words))
            for i in range(n):
                final.append({
                    "word": user_words[i],
                    "start": words[i]["start"],
                    "end": words[i]["end"],
                })
            # If user lyrics are longer, append remaining with estimated spacing
            if len(user_words) > n and words:
                last_end = words[-1]["end"]
                avg_dur = (words[-1]["end"] - words[0]["start"]) / max(len(words), 1)
                for i, w in enumerate(user_words[n:]):
                    start = last_end + i * avg_dur * 0.8
                    final.append({"word": w, "start": start, "end": start + avg_dur * 0.6})
            words = final
        else:
            warn("  User lyrics length differs significantly from ASR – keeping ASR words")

    return words

def OLD_transcribe_words(
    vocals_path: Path,
    language: str = "en",
    model_size: str = "medium",
) -> List[Dict[str, Any]]:
    """Return list of {word, start, end} using faster-whisper."""
    """Word-level transcription with faster-whisper (int8 on CPU)."""
    from faster_whisper import WhisperModel
    debug(f"  Loading Whisper '{model_size}' (int8) …")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
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

def snap_words_to_energy(
    words: List[Dict[str, Any]],
    vocals_path: Path,
    sr: int = 22050,
    search_window: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Post-process word timings: snap start/end to nearby local energy peaks
    in the vocals stem. This noticeably improves alignment on music.
    """
    if not words:
        return words

    import librosa
    import numpy as np
    from scipy.signal import find_peaks

    debug("  Snapping word boundaries to vocal energy peaks …")
    y, sr = load_audio_safe(vocals_path, sr=sr)
    hop = 256
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # Find local peaks in energy
    peaks, _ = find_peaks(rms, height=np.percentile(rms, 40), distance=2)
    peak_times = times[peaks]

    if len(peak_times) == 0:
        return words

    snapped = []
    for w in words:
        start, end = w["start"], w["end"]

        # Snap start to nearest peak within search_window
        candidates = peak_times[(peak_times >= start - search_window) &
                                (peak_times <= start + search_window)]
        if len(candidates) > 0:
            start = float(candidates[np.argmin(np.abs(candidates - start))])

        # Snap end (prefer a peak after the new start)
        candidates = peak_times[(peak_times >= start + 0.05) &
                                (peak_times <= end + search_window)]
        if len(candidates) > 0:
            end = float(candidates[np.argmin(np.abs(candidates - end))])
        else:
            end = max(end, start + 0.08)  # guarantee minimum duration

        if end <= start:
            end = start + 0.10

        snapped.append({
            "word": w["word"],
            "start": start,
            "end": end,
        })

    return snapped

def merge_vocal_regions(
    words: List[Dict[str, Any]],
    gap_threshold: float = 0.7,
) -> List[Tuple[float, float]]:
    """
    Merge word intervals into contiguous vocal-activity regions.
    A new region starts when the gap to the previous word exceeds gap_threshold.
    """
    if not words:
        return []

    # sort just in case
    words = sorted(words, key=lambda w: w["start"])
    regions = []
    cur_start, cur_end = words[0]["start"], words[0]["end"]
    for w in words[1:]:
        if w["start"] - cur_end <= gap_threshold:
            cur_end = max(cur_end, w["end"])
        else:
            regions.append((cur_start, cur_end))
            cur_start, cur_end = w["start"], w["end"]
    regions.append((cur_start, cur_end))
    return regions

# ---------------------------------------------------------------------------
# Structure + novelties
# ---------------------------------------------------------------------------

def analyze_structure_and_novelties(
    audio_path: Path,
    sr: int = 44100,
    hop_length: int = 512,
) -> List[Tuple[float, float, str]]:
    """
    Detect major sections (intro/verse/chorus/bridge/outro) and
    other novelty events (volume, timbre, texture changes).
    Returns list of (start, end, label) suitable for a features track.
    """
    import librosa
    import numpy as np
    from scipy.signal import find_peaks
    from sklearn.cluster import AgglomerativeClustering  # usually available via librosa deps

    debug("  Computing features for structure / novelty …")
    y, sr = load_audio_safe(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    # Adaptive hop for low RAM
    free = available_ram_gb()
    if free < 5.0:
        hop_length = max(hop_length, 1024)

    # Core features
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

    # Novelty / boundary strength (combine several cues)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Spectral novelty via MFCC delta
    mfcc_delta = librosa.feature.delta(mfcc)
    novelty = np.mean(np.abs(mfcc_delta), axis=0)
    novelty = librosa.util.normalize(novelty + 0.3 * librosa.util.normalize(onset_env))
    times = librosa.frames_to_time(np.arange(len(novelty)), sr=sr, hop_length=hop_length)

    # Find boundary candidates
    peaks, props = find_peaks(novelty, height=np.percentile(novelty, 70),
                              distance=int(1.5 * sr / hop_length))  # min ~1.5 s
    boundary_times = [0.0] + sorted(times[peaks].tolist()) + [duration]

    # Simple segment clustering on mean feature vectors
    n_segments = max(4, min(12, len(boundary_times) - 1))
    features, seg_times = [], []
    for i in range(len(boundary_times) - 1):
        t0, t1 = boundary_times[i], boundary_times[i + 1]
        f0 = librosa.time_to_frames(t0, sr=sr, hop_length=hop_length)
        f1 = min(librosa.time_to_frames(t1, sr=sr, hop_length=hop_length), mfcc.shape[1])
        if f1 <= f0:
            continue
        vec = np.concatenate([
            np.mean(mfcc[:, f0:f1], axis=1),
            np.mean(chroma[:, f0:f1], axis=1),
            [np.mean(rms[f0:f1]), np.mean(cent[f0:f1])]
        ])
        features.append(vec)
        seg_times.append((t0, t1))

    if len(features) < 2:
        return [(0.0, duration, "full")]

    features = np.array(features)
    # Normalise
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    n_clusters = min(5, len(features))
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(features)

    # Heuristic naming
#    results: List[Tuple[float, float, str]] = []
#    energy = [np.mean(rms[librosa.time_to_frames(t0, sr=sr, hop_length=hop_length):
#                          librosa.time_to_frames(t1, sr=sr, hop_length=hop_length)])
#              for t0, t1 in seg_times]
    energy = []
    for t0, t1 in seg_times:
        f0 = librosa.time_to_frames(t0, sr=sr, hop_length=hop_length)
        f1 = librosa.time_to_frames(t1, sr=sr, hop_length=hop_length)
        energy.append(np.mean(rms[f0:f1]))
    median_e = np.median(energy)

    results = []
    for i, ((t0, t1), lab, e) in enumerate(zip(seg_times, labels, energy)):
        if i == 0 and t1 < 25:
            name = "intro"
        elif i == len(seg_times) - 1 and (duration - t0) < 30:
            name = "outro"
        elif e > median_e * 1.15:
            name = "chorus"
        elif e < median_e * 0.85:
            name = "bridge" if i > 1 else "verse"
        else:
            name = "verse"
        results.append((t0, t1, name))

    # Additional novelty events (volume / timbre spikes that are not section boundaries)
    vol_delta = np.abs(np.diff(rms, prepend=rms[0]))
    timbre_delta = np.abs(np.diff(cent, prepend=cent[0]))
    for arr, label in [(vol_delta, "volume_change"), (timbre_delta, "timbre_shift")]:
        pks, _ = find_peaks(arr, height=np.percentile(arr, 92),
                            distance=int(2.0 * sr / hop_length))
        for p in pks:
            t = times[p]
            # skip if very close to an existing boundary
            if any(abs(t - b) < 1.0 for b in boundary_times):
                continue
            results.append((t, t + 0.15, label))

    # Sort by time
    results.sort(key=lambda x: x[0])
    return results

def OLD_load_lyrics_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    # Simple normalisation – keep it readable
    return " ".join(text.split())

# ---------------------------------------------------------------------------
# Genre + Emotion + Color Palette
# ---------------------------------------------------------------------------

def is_likely_music(y, sr, tempo: float, beat_count: int) -> bool:
    """Simple heuristic: regular beats + harmonic content + sustained energy."""
    import librosa
    import numpy as np
    if beat_count < 8 or tempo < 40 or tempo > 220:
        return False
    # Harmonic ratio proxy
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.mean(y_harm ** 2)
    perc_energy = np.mean(y_perc ** 2)
    harm_ratio = harm_energy / (harm_energy + perc_energy + 1e-8)
    # Spectral flatness (lower = more tonal)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    return harm_ratio > 0.25 and flatness < 0.4

def estimate_genre_and_emotion(
    audio_path: Path,
    tempo: float,
    words: Optional[List[Dict]] = None,
    sr: int = 44100,
) -> Dict[str, Any]:
    """
    Returns dict with:
      genre, confidence, emotion, energy, valence_proxy, palette (list of hex)
    """
    import librosa
    import numpy as np

    y, sr = load_audio_safe(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    # Feature extraction (lightweight)
    hop = 1024 if available_ram_gb() < 5 else 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    mean_rms = float(np.mean(rms))
    mean_cent = float(np.mean(cent))
    mean_bw = float(np.mean(bandwidth))
    mean_contrast = float(np.mean(contrast))
    mean_zcr = float(np.mean(zcr))
    percussiveness = float(np.mean(onset_env))

    # Simple major/minor proxy (higher chroma variance in 5ths etc. is rough)
    chroma_mean = np.mean(chroma, axis=1)
    # very rough valence: brighter + higher cent → higher valence
    valence = 0.5 + 0.3 * (mean_cent / 4000 - 0.5) + 0.2 * (mean_contrast / 30 - 0.5)
    valence = float(np.clip(valence, 0.0, 1.0))
    arousal = float(np.clip(0.4 * (tempo / 140) + 0.4 * (mean_rms * 10) + 0.2 * (percussiveness / 5), 0, 1))

    # Lyric keyword hints (very lightweight)
    lyric_boost = {}
    if words:
        text = " ".join(w["word"].lower() for w in words)
        keywords = {
            "hip-hop": ["rap", "beat", "flow", "mic", "rhyme", "hood", "street"],
            "rock": ["guitar", "rock", "scream", "fire", "night", "road"],
            "pop": ["love", "baby", "heart", "dance", "tonight", "feel"],
            "electronic": ["bass", "drop", "synth", "club", "rave", "pulse"],
            "metal": ["scream", "dark", "blood", "fire", "death", "shadow"],
            "r&b": ["love", "baby", "soul", "night", "feel", "touch"],
            "country": ["road", "truck", "beer", "home", "heart", "whiskey"],
            "jazz": ["blue", "night", "soul", "swing", "moon"],
        }
        for g, kws in keywords.items():
            score = sum(1 for k in kws if k in text)
            if score:
                lyric_boost[g] = score

    # Rule-based genre scoring
    scores = {
        "electronic / dance": 0.0,
        "pop": 0.0,
        "rock": 0.0,
        "hip-hop / rap": 0.0,
        "r&b / soul": 0.0,
        "metal": 0.0,
        "indie / alternative": 0.0,
        "ambient / chill": 0.0,
        "jazz": 0.0,
        "classical / orchestral": 0.0,
        "country / folk": 0.0,
    }

    # Tempo + energy rules
    if 120 <= tempo <= 140 and mean_rms > 0.08 and percussiveness > 3:
        scores["electronic / dance"] += 2.5
        scores["pop"] += 1.5
    if 90 <= tempo <= 110 and mean_cent < 2500:
        scores["hip-hop / rap"] += 2.0
        scores["r&b / soul"] += 1.0
    if 110 <= tempo <= 140 and mean_contrast > 20 and mean_bw > 2000:
        scores["rock"] += 2.0
        scores["indie / alternative"] += 1.0
    if tempo > 140 and mean_cent > 3000 and mean_zcr > 0.1:
        scores["metal"] += 2.5
        scores["rock"] += 1.0
    if tempo < 90 and mean_rms < 0.06 and mean_cent < 2000:
        scores["ambient / chill"] += 2.5
        scores["jazz"] += 1.0
    if 70 <= tempo <= 100 and mean_contrast < 18:
        scores["r&b / soul"] += 1.8
        scores["pop"] += 1.0
    if mean_cent > 2800 and valence > 0.6 and 100 <= tempo <= 130:
        scores["pop"] += 2.0
    if mean_bw < 1800 and mean_zcr < 0.05 and tempo < 100:
        scores["classical / orchestral"] += 1.5
        scores["jazz"] += 1.0
    if "guitar" in (lyric_boost or {}) or (mean_contrast > 22 and 100 < tempo < 140):
        scores["country / folk"] += 1.2
        scores["rock"] += 0.8

    # Apply lyric boosts
    for g, sc in lyric_boost.items():
        if g in scores:
            scores[g] += sc * 0.8
        elif g == "hip-hop":
            scores["hip-hop / rap"] += sc * 0.8
        elif g == "electronic":
            scores["electronic / dance"] += sc * 0.8

    # Pick best
    genre = max(scores, key=scores.get)
    conf = scores[genre] / (sum(scores.values()) + 1e-6)
    conf = float(np.clip(conf * 1.5, 0.15, 0.95))  # rough calibration

    # Emotion mapping
    if valence > 0.65 and arousal > 0.6:
        emotion = "joyful / energetic"
    elif valence > 0.6 and arousal < 0.45:
        emotion = "warm / content"
    elif valence < 0.4 and arousal > 0.55:
        emotion = "tense / aggressive"
    elif valence < 0.4 and arousal < 0.4:
        emotion = "melancholic / reflective"
    elif arousal > 0.7:
        emotion = "intense / driving"
    else:
        emotion = "neutral / balanced"

    # Color palette (primary = emotion)
    # Format: list of (hex, name)
    palettes = {
        "joyful / energetic": [
            ("#FF6B6B", "Coral Red"),      # primary – joy/energy
            ("#FFE66D", "Sunny Yellow"),
            ("#4ECDC4", "Aqua"),
            ("#1A1A2E", "Deep Navy"),
        ],
        "warm / content": [
            ("#FF9F43", "Warm Amber"),     # primary
            ("#FEC84B", "Golden"),
            ("#E17055", "Soft Terracotta"),
            ("#2D3436", "Charcoal"),
        ],
        "tense / aggressive": [
            ("#E74C3C", "Blood Red"),      # primary
            ("#2C3E50", "Dark Slate"),
            ("#F39C12", "Warning Orange"),
            ("#8E44AD", "Deep Purple"),
        ],
        "melancholic / reflective": [
            ("#5B6EE1", "Midnight Blue"),  # primary
            ("#A29BFE", "Soft Lavender"),
            ("#636E72", "Cool Grey"),
            ("#2D3436", "Almost Black"),
        ],
        "intense / driving": [
            ("#9B59B6", "Electric Purple"),# primary
            ("#E74C3C", "Hot Red"),
            ("#1ABC9C", "Teal Pulse"),
            ("#2C3E50", "Night"),
        ],
        "neutral / balanced": [
            ("#00B894", "Fresh Teal"),     # primary
            ("#FDCB6E", "Soft Gold"),
            ("#6C5CE7", "Muted Violet"),
            ("#2D3436", "Graphite"),
        ],
    }

    # Genre-tinted adjustments (optional extra color or swap)
    palette = palettes.get(emotion, palettes["neutral / balanced"])

    # Slight genre influence on secondary colors
    if "electronic" in genre or "dance" in genre:
        palette = [palette[0], ("#00F5FF", "Cyber Cyan"), palette[2], ("#0A0A0A", "Void Black")]
    elif "metal" in genre:
        palette = [palette[0], ("#C0392B", "Crimson"), ("#7F8C8D", "Steel"), ("#1C1C1C", "Black")]
    elif "hip-hop" in genre:
        palette = [palette[0], ("#F1C40F", "Gold Chain"), ("#8E44AD", "Royal Purple"), ("#2C3E50", "Urban Night")]
    elif "ambient" in genre or "chill" in genre:
        palette = [palette[0], ("#81ECEC", "Mist"), ("#74B9FF", "Sky"), ("#DFE6E9", "Cloud")]

    return {
        "is_music": True,          # caller already checked
        "genre": genre,
        "confidence": round(conf, 2),
        "emotion": emotion,
        "energy": round(arousal, 2),
        "valence": round(valence, 2),
        "tempo": round(tempo, 1),
        "palette": palette,        # list of (hex, name)
    }

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def handle_lyrics(config, lyrics_path, gen_words):
    if lyrics_path is None and config.get("lyrics_file"):
        lyrics_path = Path(config["lyrics_file"])
    if lyrics_path:
        lyrics_path = lyrics_path.resolve()
#        if not lyrics_path.is_file():
#            error(f"File not found: {relpath(lyrics_path, audio_path)}")
#            sys.exit(1)
    if gen_words and lyrics_path is None:
        if ask_bool(config, "has_lyrics", f"{ANSI['memo']}Have a lyrics .txt file?", False):
            lyrics_str = ask(config, "lyrics_file", f"{ANSI['memo']}Path to lyrics file")
            lyrics_path = Path(lyrics_str).expanduser().resolve()
            if not lyrics_path.is_file():
                warn(f"Lyrics file not found: {relpath(lyrics_path, audio_path)}")
                lyrics_path = None

    user_lyrics_text = None
    if lyrics_path and lyrics_path.is_file():
        user_lyrics_text = lyrics_path.read_text(encoding="utf-8").strip()
        user_lyrics_text = " ".join(user_lyrics_text.split())  # normalise whitespace
        config["lyrics_file"] = str(lyrics_path)
        info(f"  Using user-supplied lyrics ({len(user_lyrics_text.split())} words)")
    return user_lyrics_text

def handle_beats(config, gen_beats, gen_genre, prepared, audio_path, label_dir, work_dir, force):
    tempo = 120.0
    beat_times: List[float] = []

    # ----- Beats -----
    # Beats (needed early for genre detection)
    beats_file = label_dir / "beats.txt"
    if gen_beats or gen_genre:
        BEATS_VERSION = "1"  # Bump when step logic changes
        if is_up_to_date(beats_file, prepared, version=BEATS_VERSION, force=force):
            debug("  [cache] beats.txt is up-to-date → loading cached tempo")
            # Optional: you can also store tempo in the JSON config
            tempo = config.get("tempo", 120.0)
            beat_times = []          # we don't need the actual times if only genre is wanted
        else:
            debug(f"\n[2.5/{NSTEPS}] Detecting beats …")
            beat_times, tempo = detect_beats(prepared)
            debug(f"  Tempo ≈ {tempo:.1f} BPM")
            if gen_beats:
                write_audacity_labels(beat_times, label_dir / "beats.txt", basedir=work_dir)
                mark_cache(beats_file, BEATS_VERSION)
            config["tempo"] = tempo
            save_config(audio_path, config)
    return beat_times

# ----- Drum / general onsets -----
# Onsets
def handle_onsets(gen_drums, gen_onsets, stems, prepared, audio_path, label_dir, work_dir, force):
    ONSETS_VERSION = "1"  # Bump when step logic changes
    if gen_drums:
        drums_file = label_dir / "drums.txt"
        if is_up_to_date(drums_file, stems.get("drums", prepared), version=ONSETS_VERSION, force=force):
            debug("  [cache] drums.txt is up-to-date → skipping")
        else:
            debug(f"\n[3/{NSTEPS}] Detecting drum / percussive onsets …")
            drum_times = detect_onsets(stems.get("drums", prepared))
            write_audacity_labels(drum_times, label_dir / "drums.txt", basedir=work_dir)
            mark_cache(drums_file, ONSETS_VERSION)

    if gen_onsets:
        onsets_file = label_dir / "onsets.txt"
        if is_up_to_date(onsets_file, stems.get("other", prepared), version=ONSETS_VERSION, force=force):
            debug("  [cache] onsets.txt is up-to-date → skipping")
        else:
            debug(f"\n[4/{NSTEPS}] Detecting general / melodic onsets …")
            onset_times = detect_onsets(stems.get("other", prepared))
            write_audacity_labels(onset_times, label_dir / "onsets.txt", basedir=work_dir)
            mark_cache(onsets_file, ONSETS_VERSION)

# ------------------------------------------------------------------
# Genre detection (early so we can tune alignment)
# ------------------------------------------------------------------
def handle_genre(config, gen_genre, beat_times, prepared, work_dir, args):
    detected_genre = config.get("genre", "unknown")
    tempo = config.get("tempo", 120.0)
    if gen_genre:
        GENRE_VERSION = "1"  # Bump when step logic changes
        debug("\n[8/9] Genre & color palette analysis …")
        y_check, sr_check = load_audio_safe(prepared, sr=22050)
        force_music = args.force_music
        if force_music:
            info("  --force-music specified → treating as music")

        if is_likely_music(y_check, sr_check, tempo, len(beat_times)) or force_music:
            result = estimate_genre_and_emotion(prepared, tempo, words=None)  # words not ready yet
            detected_genre = result["genre"]
            info(f"\n  Detected as music")
            info(f"  Genre        : {result['genre']}  (confidence ≈ {result['confidence']:.0%})")
            info(f"  Emotion      : {result['emotion']}")
            info(f"  Energy/Valence: {result['energy']:.2f} / {result['valence']:.2f}")
            info(f"  Suggested palette (primary = main emotion):")
            for i, (hex_col, name) in enumerate(result["palette"]):
                marker = "← primary" if i == 0 else ""
                info(f"    {hex_col}  {name}  {marker}")

            config["genre"] = result["genre"]
            config["emotion"] = result["emotion"]
            config["palette"] = [{"hex": h, "name": n} for h, n in result["palette"]]
            save_config(audio_path, config)

            summary = work_dir / "summary.txt"
            with open(summary, "w", encoding="utf-8") as f:
                f.write(f"Genre: {result['genre']} ({result['confidence']:.0%})\n")
                f.write(f"Emotion: {result['emotion']}\n")
                f.write(f"Tempo: {result['tempo']} BPM\n")
                f.write("Palette:\n")
                for h, n in result["palette"]:
                    f.write(f"  {h}  {n}\n")
            good(f"  Summary written → {relpath(summary, work_dir)}")
        else:
            warn("  Audio does not appear to be music – skipping genre/palette.")
            warn("  Use --force-music to override.")

# ------------------------------------------------------------------
# Transcription + alignment + snapping
# ------------------------------------------------------------------
# ----- Transcription + words (most expensive) -----
def handle_transcription(config, gen_words, gen_vocals, stems, prepared, language, user_lyrics_text, whisper_model, label_dir, work_dir, force):
    detected_genre = config.get("genre", "unknown")
    words_file = label_dir / "words.txt"
    words_regions_file = label_dir / "words_regions.txt"
    transcript_file = work_dir / "transcript.txt"
    vocals_path = stems.get("vocals", prepared)
    words: List[Dict[str, Any]] = []
    if gen_words or gen_vocals:
        TRANSCRIBE_VERSION = "1"  # Bump when step logic changes
        cache_outputs = [words_file, words_regions_file, transcript_file]
        if is_up_to_date(cache_outputs, vocals_path, prepared,
                         version=TRANSCRIBE_VERSION, force=force):
            debug("  [cache] word timings & transcript are up-to-date → skipping transcription")
            # Optionally reload words from the label file if you need them later
            # (for vocal regions, etc.)
        else:
            debug("\n[6/9] Transcription / alignment / vocal regions …")
            words = transcribe_words(
                vocals_path,
                language=language,
                model_size=whisper_model,
                genre=detected_genre,
                user_lyrics=user_lyrics_text,
            )

            if words:
                words = snap_words_to_energy(words, vocals_path)

                if gen_words:
                    write_audacity_labels(
                        [w["start"] for w in words],
                        words_file,
                        labels=[w["word"] for w in words],
                        basedir=work_dir,
                    )
                    write_audacity_labels(
                        [w["start"] for w in words],
                        words_regions_file,
                        labels=[w["word"] for w in words],
                        as_regions=True,
                        durations=[w["end"] - w["start"] for w in words],
                        basedir=work_dir,
                    )
                    transcript_file.write_text(
                        " ".join(w["word"] for w in words), encoding="utf-8"
                    )
                    mark_cache(cache_outputs, TRANSCRIBE_VERSION)
                    info(f"  Transcript saved → {relpath(transcript_file, work_dir)}")

        # Vocal regions (can be rebuilt cheaply from words if needed)
        if gen_vocals:
            vocals_regions_file = label_dir / "vocals.txt"
            if words or not is_up_to_date(vocals_regions_file, words_file, version=TRANSCRIBE_VERSION, force=force):
                if not words and words_file.exists():
                    # very light reload if we skipped transcription
                    words = []  # you can parse the label file here if desired
                regions = merge_vocal_regions(words) if words else []
                if regions:
                    write_audacity_labels(
                        [r[0] for r in regions],
                        vocals_regions_file,
                        labels=["vocals"] * len(regions),
                        as_regions=True,
                        durations=[r[1] - r[0] for r in regions],
                        basedir=work_dir,
                    )
                    mark_cache(vocals_regions_file, TRANSCRIBE_VERSION)
                else:
                    warn("  No contiguous vocal regions found.")

# ------------------------------------------------------------------
# Structure + novelties
# ------------------------------------------------------------------
def handle_features(gen_features, label_dir):
    if gen_features:
        FEATURES_VERSION = "1"  # Bump when step logic changes
        features_file = label_dir / "features.txt"
        if is_up_to_date(features_file, prepared, version=FEATURES_VERSION, force=force):
            debug("  [cache] features.txt is up-to-date → skipping")
        else:
            debug(f"\n[7/{NSTEPS}] Analysing song structure and novelties …")
            features = analyze_structure_and_novelties(prepared)
            if features:
                starts = [f[0] for f in features]
                durs = [f[1] - f[0] for f in features]
                labels = [f[2] for f in features]
                write_audacity_labels(
                    starts,
                    label_dir / "features.txt",
                    labels=labels,
                    as_regions=True,
                    durations=durs,
                    basedir=work_dir,
                )
                mark_cache(features_file, FEATURES_VERSION)
            else:
                warn("  No structural features detected.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Audio timing + genre + palette generator")
    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument("--lyrics", type=Path, default=None,
                        help="Optional plain-text lyrics file")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Working directory (default: <audio-dir>/auto_seq)")
    parser.add_argument("--force-music", action="store_true",
                        help="Force treating the audio as music and run genre/palette analysis")
    parser.add_argument("--force", action="store_true",
                        help="Ignore all caches and recompute everything")
    args = parser.parse_args()

    audio_path = args.audio.resolve()
    if not audio_path.is_file():
        error(f"File not found: {audio_path}")
        sys.exit(1)

    info("=" * 64, srcline="")
    info(f"Audio Timing + Genre/Palette Generator  v{APP_VERSION}")
    info("=" * 64, srcline="")

    free = available_ram_gb()
    info(f"Estimated available RAM: {free:.1f} GB")
    if free < 4.5:
        warn("Low free RAM – using conservative settings.")

    # 0. Dependencies
    check_and_offer_install()
    ensure_ffmpeg()

    # 1. Config / user choices
    config = load_config(audio_path)
    work_dir = (args.work_dir or (audio_path.parent / "auto_seq")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    debug(f"Working directory: {relpath(work_dir, audio_path)}")

    # User choices
    do_separate = ask_bool(config, "separate",
                           f"{ANSI['memo']}Run Demucs source separation?", free > 6.0)
    language = ask(config, "language", f"{ANSI['memo']}Language code", "en")
    whisper_model = ask(config, "whisper_model",
                        f"{ANSI['memo']}Whisper model (tiny/base/small/medium/large-v3)",
                        "small" if free < 6 else "medium")
    gen_drums   = ask_bool(config, "gen_drums",   f"{ANSI['memo']}Drum onset track?", True)
    gen_onsets  = ask_bool(config, "gen_onsets",  f"{ANSI['memo']}General onset track?", True)
    gen_beats   = ask_bool(config, "gen_beats",   f"{ANSI['memo']}Beat track?", True)
    gen_words   = ask_bool(config, "gen_words",   f"{ANSI['memo']}Word track?", True)
    gen_vocals  = ask_bool(config, "gen_vocals",  f"{ANSI['memo']}Contiguous vocals regions?", True)
    gen_features= ask_bool(config, "gen_features",f"{ANSI['memo']}Structure + novelty features?", True)
    gen_genre   = ask_bool(config, "gen_genre",   f"{ANSI['memo']}Detect genre + color palette?", True)

    # Lyrics handling
    lyrics_path = args.lyrics
    user_lyrics_text = handle_lyrics(config, lyrics_path, gen_words)
    save_config(audio_path, config)

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    force = args.force
    if force:
        warn("  --force specified → all caches will be ignored")

    # 2. Prepare
    prepared = prepare_audio(audio_path, work_dir, force=force)

    # 3. Separation
    stems = separate_sources(prepared, work_dir, do_separate, force=force)

    label_dir = work_dir / "labels"
    label_dir.mkdir(exist_ok=True)

    # 4. Onsets & beats
    beat_times = handle_beats(config, gen_beats, gen_genre, prepared, audio_path, label_dir, work_dir, force)
    handle_onsets(gen_drums, gen_onsets, stems, prepared, audio_path, label_dir, work_dir, force)

    handle_genre(config, gen_genre, beat_times, prepared, work_dir, args)

    handle_transcription(config, gen_words, gen_vocals, stems, prepared, language, user_lyrics_text, whisper_model, label_dir, work_dir, force)

    handle_features(gen_features, label_dir)

    # Done
    memo(f"\n[9/{NSTEPS}] Finished.")
    info(f"\nLabel files → {relpath(label_dir, audio_path)}")
    info("Import via File → Import → Labels in Audacity (each file is separate track).")
    memo("\nRe-run on same audio file to reuse previous answers.")
    debug("elapsed:", elapsed())

if __name__ == "__main__":
    main()

#eof