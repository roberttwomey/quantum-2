#!/usr/bin/env python3
"""Voice chat assistant using Whisper STT, Ollama Gemma 3 Nano, and Piper TTS.

This script performs continuous voice activity detection (VAD) on microphone
audio, automatically segments speech, transcribes each utterance with Whisper,
sends the resulting text to an Ollama model (`gemma3n:e2b` by default), and
plays back the assistant response via Piper text-to-speech using the Python
`piper-tts` library.

Requirements:
    - ollama (Python package) with the `gemma3n:e2b` model pulled locally
    - faster-whisper
    - sounddevice
    - soundfile
    - numpy
    - piper-tts (Python package) and at least one Piper voice model file

Example usage:
    python run-experience.py --piper-voice piper/en_GB-alan-medium.onnx
    
    python run-experience.py --piper-voice piper/en_GB-alan-medium.onnx --show-levels

To test just ollama: 
ollama run gemma3n:e2b

Environment variables:
    OLLAMA_MODEL   override Ollama model name (default: gemma3n:e2b)
    WHISPER_MODEL  override Whisper model size (default: tiny)
    PIPER_VOICE    override Piper voice path if --piper-voice not provided
    INTERRUPTABLE  override interruptable behavior (default: true)
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List
import wave

import numpy as np
import ollama
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import torch
import dotenv

from piper import PiperVoice
try:  # Optional type that some versions expose
    from piper import AudioChunk  # type: ignore
except ImportError:  # pragma: no cover - older library versions
    AudioChunk = None



dotenv.load_dotenv()

def fix_user_paths() -> None:
    """
    If /home/cohab does not exist, replace /home/cohab prefixes in environment variables
    with the current user's home directory.
    """
    # Check if we are likely on a different machine (i.e., /home/cohab missing)
    # or just want to be safe and use the current user's home.
    cohab_home = Path("/home/cohab")
    if cohab_home.exists():
        return

    current_home = Path.home()
    print(f"Notice: {cohab_home} not found. Remapping paths to {current_home}...", file=sys.stderr)

    for key, value in os.environ.items():
        if value and "/home/cohab" in value:
            new_value = value.replace("/home/cohab", str(current_home))
            os.environ[key] = new_value
            # print(f"  Remapped {key}: {value} -> {new_value}", file=sys.stderr)

fix_user_paths()


DEFAULT_SYSTEM_PROMPT = (
    os.environ.get(
        "SYSTEM_PROMPT",
        "You are an interactive quantum computing system, Q, conducting a dialog with two human participants, A and B. Take them through a series of improvisational exercises that unfold key quantum phenomena.",
    )
)

DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3n:e2b")
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
DEFAULT_SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
DEFAULT_PLAYBACK_SPEED = float(os.environ.get("PLAYBACK_SPEED", "1.0"))
DEFAULT_INPUT_DEVICE_KEYWORD = os.environ.get(
    "INPUT_DEVICE_KEYWORD", "OpenRun Pro 2 by Shokz"
)
DEFAULT_ACTIVATION_THRESHOLD = float(os.environ.get("ACTIVATION_THRESHOLD", "0.03"))
DEFAULT_SILENCE_THRESHOLD = float(os.environ.get("SILENCE_THRESHOLD", "0.015"))
DEFAULT_SILENCE_DURATION = float(os.environ.get("SILENCE_DURATION", "0.8"))
DEFAULT_MIN_PHRASE_SECONDS = float(os.environ.get("MIN_PHRASE_SECONDS", "0.5"))
DEFAULT_BLOCK_DURATION = float(os.environ.get("BLOCK_DURATION", "0.2"))
DEFAULT_INTERRUPTABLE_ENV = os.environ.get("INTERRUPTABLE", "true").lower()
DEFAULT_INTERRUPTABLE = DEFAULT_INTERRUPTABLE_ENV in ("true", "1", "yes", "on")
DEFAULT_FLUSH_ON_INTERRUPT_ENV = os.environ.get("FLUSH_ON_INTERRUPT", "false").lower()
DEFAULT_FLUSH_ON_INTERRUPT = DEFAULT_FLUSH_ON_INTERRUPT_ENV in ("true", "1", "yes", "on")
LOG_ROOT = Path(os.environ.get("LOG_ROOT", str(Path(__file__).parent / "logs"))).expanduser()
DEFAULT_HISTORY_TRUNCATION_LIMIT = int(os.environ.get("HISTORY_TRUNCATION_LIMIT", "11"))
DEFAULT_OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.7"))
DEFAULT_OLLAMA_TOP_P = float(os.environ.get("OLLAMA_TOP_P", "0.9"))
DEFAULT_OLLAMA_TOP_K = int(os.environ.get("OLLAMA_TOP_K", "40"))
DEFAULT_OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "100"))
DEFAULT_OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "2048"))
LAST_HEADSET_MAC = os.environ.get("LAST_HEADSET_MAC")
LAST_HEADSET_NAME = os.environ.get("LAST_HEADSET_NAME")

@dataclass
class ConversationConfig:
    """Runtime configuration for the voice chat assistant."""

    ollama_model: str = DEFAULT_OLLAMA_MODEL
    whisper_model: str = DEFAULT_WHISPER_MODEL
    whisper_compute_type: str = "int8"  # optimized for Jetson
    piper_voice: Path | None = None
    piper_config: Path | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    sample_rate: int = DEFAULT_SAMPLE_RATE
    max_record_seconds: int = 20
    piper_length_scale: float | None = None
    piper_noise_scale: float | None = None
    piper_noise_w: float | None = None
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD
    silence_duration: float = DEFAULT_SILENCE_DURATION
    min_phrase_seconds: float = DEFAULT_MIN_PHRASE_SECONDS
    block_duration: float = DEFAULT_BLOCK_DURATION
    show_levels: bool = True
    input_device_keyword: str | None = DEFAULT_INPUT_DEVICE_KEYWORD
    input_device_index: int | None = None
    interruptable: bool = DEFAULT_INTERRUPTABLE
    flush_on_interrupt: bool = DEFAULT_FLUSH_ON_INTERRUPT
    history_truncation_limit: int = DEFAULT_HISTORY_TRUNCATION_LIMIT
    ollama_temperature: float = DEFAULT_OLLAMA_TEMPERATURE
    ollama_top_p: float = DEFAULT_OLLAMA_TOP_P
    ollama_top_k: int = DEFAULT_OLLAMA_TOP_K
    ollama_num_predict: int = DEFAULT_OLLAMA_NUM_PREDICT
    ollama_num_ctx: int = DEFAULT_OLLAMA_NUM_CTX


def parse_args() -> ConversationConfig:
    parser = argparse.ArgumentParser(description="Interactive voice chat assistant")
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name to use (default: %(default)s)",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help="Whisper model size to load (default: %(default)s)",
    )
    parser.add_argument(
        "--whisper-compute-type",
        default="int8",
        help="Quantization type for Whisper (default: int8, options: float16, int8_float16, int8)",
    )
    parser.add_argument(
        "--piper-voice",
        default=os.environ.get("PIPER_VOICE"),
        type=Path,
        help="Path to Piper voice model (*.onnx) (default: env PIPER_VOICE)",
    )
    parser.add_argument(
        "--piper-config",
        type=Path,
        help="Optional path to Piper voice config (*.json); defaults to <voice>.json",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt sent with each conversation",
    )
    parser.add_argument(
        "--max-record-seconds",
        type=int,
        default=20,
        help="Maximum seconds to record per turn (default: %(default)s)",
    )
    parser.add_argument(
        "--piper-length-scale",
        type=float,
        help="Override Piper config length_scale (lower=faster)",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=DEFAULT_PLAYBACK_SPEED,
        help="Audio playback speed multiplier (default: %(default)s). Overrides length_scale if length_scale is not set.",
    )
    parser.add_argument(
        "--piper-noise-scale",
        type=float,
        help="Override Piper config noise_scale",
    )
    parser.add_argument(
        "--piper-noise-w",
        type=float,
        help="Override Piper config noise_w",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.03,
        help="RMS amplitude that starts a speech segment (default: %(default)s)",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.015,
        help="RMS amplitude below which audio counts as silence (default: %(default)s)",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.8,
        help="Seconds of silence that end a speech segment (default: %(default)s)",
    )
    parser.add_argument(
        "--min-phrase-seconds",
        type=float,
        default=0.5,
        help="Discard segments shorter than this many seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--block-duration",
        type=float,
        default=0.2,
        help="Processing block size in seconds for VAD (default: %(default)s)",
    )
    parser.add_argument(
        "--show-levels",
        action="store_true",
        default=True,
        help="Print live RMS level meter to stderr (default: enabled)",
    )
    parser.add_argument(
        "--no-show-levels",
        dest="show_levels",
        action="store_false",
        help="Disable live RMS level meter",
    )
    parser.add_argument(
        "--input-device-keyword",
        default=DEFAULT_INPUT_DEVICE_KEYWORD,
        help="Substring to match desired input device (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate for recording and playback (default: %(default)s)",
    )
    parser.add_argument(
        "--no-interruptable",
        action="store_true",
        help="Disable interruptable behavior for Ollama queries and audio playback (default: from INTERRUPTABLE env or enabled)",
    )
    args = parser.parse_args()

    if args.piper_voice is None:
        parser.error("Piper voice model must be provided via --piper-voice or PIPER_VOICE")
    if not args.piper_voice.exists():
        parser.error(f"Piper voice model not found: {args.piper_voice}")

    input_keyword = args.input_device_keyword.strip() if args.input_device_keyword else None
    if input_keyword == "":
        input_keyword = None

    # Calculate length_scale from playback_speed if not explicitly provided
    length_scale = args.piper_length_scale
    if length_scale is None:
        # length_scale of 1.0 is normal speed. Lower is faster.
        # speed = 1.25 -> length_scale = 1/1.25 = 0.8
        speed = args.playback_speed
        if speed <= 0:
            speed = 1.0
        length_scale = 1.0 / speed

    return ConversationConfig(
        ollama_model=args.ollama_model,
        whisper_model=args.whisper_model,
        whisper_compute_type=args.whisper_compute_type,
        piper_voice=args.piper_voice,
        piper_config=args.piper_config,
        system_prompt=args.system_prompt,
        sample_rate=args.sample_rate,
        max_record_seconds=args.max_record_seconds,
        piper_length_scale=length_scale,
        piper_noise_scale=args.piper_noise_scale,
        piper_noise_w=args.piper_noise_w,
        activation_threshold=args.activation_threshold,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        min_phrase_seconds=args.min_phrase_seconds,
        block_duration=args.block_duration,
        show_levels=args.show_levels,
        input_device_keyword=input_keyword,
        interruptable=False if args.no_interruptable else DEFAULT_INTERRUPTABLE,
        history_truncation_limit=DEFAULT_HISTORY_TRUNCATION_LIMIT,
        ollama_temperature=DEFAULT_OLLAMA_TEMPERATURE,
        ollama_top_p=DEFAULT_OLLAMA_TOP_P,
        ollama_top_k=DEFAULT_OLLAMA_TOP_K,
        ollama_num_predict=DEFAULT_OLLAMA_NUM_PREDICT,
        ollama_num_ctx=DEFAULT_OLLAMA_NUM_CTX,
    )


def load_whisper_model(name: str, compute_type: str = "int8") -> WhisperModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Faster Whisper model '{name}' on {device} ({compute_type})…", file=sys.stderr)
    return WhisperModel(name, device=device, compute_type=compute_type)


def resolve_piper_config_path(model_path: Path, config_path: Path | None) -> Path:
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Piper config not found: {config_path}")
        return config_path

    candidate = model_path.with_suffix(model_path.suffix + ".json")
    if candidate.exists():
        return candidate

    alt = model_path.with_suffix(".json")
    if alt.exists():
        return alt

    raise FileNotFoundError(
        "Could not infer Piper config JSON. Provide --piper-config explicitly."
    )


def load_piper_voice(
    model_path: Path,
    config_path: Path | None,
    *,
    length_scale: float | None = None,
    noise_scale: float | None = None,
    noise_w: float | None = None,
) -> PiperVoice:
    resolved = resolve_piper_config_path(model_path, config_path)
    with open(resolved, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    overrides = {
        "length_scale": length_scale,
        "noise_scale": noise_scale,
        "noise_w": noise_w,
    }

    applied = {k: v for k, v in overrides.items() if v is not None}
    tmp_path: Path | None = None

    if applied:
        config_data.update(applied)
        # Some Piper voices (e.g. aru) have params inside an 'inference' block
        if "inference" in config_data:
             config_data["inference"].update(applied)

        tmp_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        tmp_path = Path(tmp_file.name)
        json.dump(config_data, tmp_file)
        tmp_file.flush()
        tmp_file.close()
        config_to_use = tmp_path
        print(
            "Loading Piper voice '{}' with overrides {}".format(
                model_path.name,
                ", ".join(f"{k}={v}" for k, v in applied.items()),
            ),
            file=sys.stderr,
        )
    else:
        config_to_use = resolved
        config_to_use = resolved
        # print(
        #     f"Loading Piper voice '{model_path.name}' with config '{resolved.name}'…",
        #     file=sys.stderr,
        # )

    try:
        voice = PiperVoice.load(str(model_path), config_path=str(config_to_use))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return voice


def ensure_log_dir() -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    return LOG_ROOT


def append_log_line(log_path: Path, payload: dict[str, Any]) -> None:
    record = {"timestamp": datetime.now().isoformat(), **payload}
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def meter_break(show_levels: bool) -> None:
    if show_levels:
        sys.stderr.write("\n") #\n
        sys.stderr.flush()


def check_bluetooth_connection_status(mac: str) -> bool:
    """Check if a Bluetooth device is connected via bluetoothctl."""
    try:
        result = subprocess.run(
            ["bluetoothctl", "info", mac],
            capture_output=True,
            text=True,
            check=False,
        )
        return "Connected: yes" in result.stdout
    except Exception:
        return False


def find_pulseaudio_card_by_mac(mac: str, max_retries: int = 5, retry_delay: float = 1.0) -> str | None:
    """Find PulseAudio card by MAC address, with retries."""
    for attempt in range(max_retries):
        pactl_result = subprocess.run(
            ["pactl", "list", "cards", "short"],
            capture_output=True,
            text=True,
            check=False,
        )
        
        for line in pactl_result.stdout.splitlines():
            if mac.lower() in line.lower():
                card_id = line.split()[0]
                return card_id
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    return None


def try_auto_connect_headset(mac: str, name: str) -> bool:
    """Attempt to auto-connect to a saved headset."""
    if not mac or not name:
        return False
    
    print(f"Attempting to auto-connect to last headset: {name} ({mac})", file=sys.stderr)
    
    try:
        # Check if already connected
        if check_bluetooth_connection_status(mac):
            print(f"Device {name} ({mac}) is already connected.", file=sys.stderr)
            # Try to find and configure PulseAudio card
            card_id = find_pulseaudio_card_by_mac(mac, max_retries=3, retry_delay=0.5)
            if card_id:
                subprocess.run(
                    ["pactl", "set-card-profile", card_id, "headset-head-unit"],
                    capture_output=True,
                    check=False,
                )
                print(f"Auto-connect successful! {name} is connected and configured.", file=sys.stderr)
                return True
            else:
                print(f"Device is connected via Bluetooth, but PulseAudio card not yet available.", file=sys.stderr)
                print(f"This is normal - the device should work as the system default audio device.", file=sys.stderr)
                return True
        
        # Initialize Bluetooth
        subprocess.run(["bluetoothctl", "power", "on"], capture_output=True, check=False)
        subprocess.run(["bluetoothctl", "agent", "on"], capture_output=True, check=False)
        subprocess.run(["bluetoothctl", "default-agent"], capture_output=True, check=False)
        
        # Trust and connect
        bluetoothctl_input = f"trust {mac}\nconnect {mac}\n"
        result = subprocess.run(
            ["bluetoothctl"],
            input=bluetoothctl_input.encode(),
            capture_output=True,
            check=False,
            timeout=10,
        )
        
        # Wait for connection to establish and check status
        for attempt in range(5):
            time.sleep(1)
            if check_bluetooth_connection_status(mac):
                break
        else:
            print(f"Auto-connect failed: Could not establish Bluetooth connection to {name} ({mac}).", file=sys.stderr)
            return False
        
        # Try to find and configure PulseAudio card (with retries)
        card_id = find_pulseaudio_card_by_mac(mac, max_retries=5, retry_delay=1.0)
        if card_id:
            subprocess.run(
                ["pactl", "set-card-profile", card_id, "headset-head-unit"],
                capture_output=True,
                check=False,
            )
            print(f"Auto-connect successful! Connected {name} ({mac}) in headset (HFP/HSP) mode.", file=sys.stderr)
            return True
        else:
            print(f"Bluetooth connection established to {name} ({mac}), but PulseAudio card not yet available.", file=sys.stderr)
            print(f"This is normal - the device should work as the system default audio device.", file=sys.stderr)
            return True
        
    except Exception as exc:
        print(f"Auto-connect error: {exc}", file=sys.stderr)
        return False


def scan_bluetooth_devices() -> list[tuple[str, str]]:
    """Scan for Bluetooth devices and return list of (mac, name) tuples."""
    devices: list[tuple[str, str]] = []
    
    try:
        # Initialize Bluetooth
        subprocess.run(["bluetoothctl", "power", "on"], capture_output=True, check=False)
        subprocess.run(["bluetoothctl", "agent", "on"], capture_output=True, check=False)
        subprocess.run(["bluetoothctl", "default-agent"], capture_output=True, check=False)
        
        # Start scanning
        scan_process = subprocess.Popen(
            ["bluetoothctl", "scan", "on"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for devices to be discovered
        time.sleep(5)
        
        # Stop scanning
        scan_process.terminate()
        subprocess.run(["bluetoothctl", "scan", "off"], capture_output=True, check=False)
        
        # Get list of devices
        result = subprocess.run(
            ["bluetoothctl", "devices"],
            capture_output=True,
            text=True,
            check=False,
        )
        
        # Filter for audio devices first
        audio_devices = []
        all_devices = []
        
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split(None, 2)
            if len(parts) >= 3:
                mac = parts[1]
                name = parts[2]
                device_tuple = (mac, name)
                all_devices.append(device_tuple)
                # Check if it's an audio device
                name_lower = name.lower()
                if any(keyword in name_lower for keyword in ["headset", "audio", "speaker", "earbuds", "airpods"]):
                    audio_devices.append(device_tuple)
        
        # Return audio devices if found, otherwise all devices
        return audio_devices if audio_devices else all_devices
    except Exception as exc:
        print(f"Bluetooth scan error: {exc}", file=sys.stderr)
        return []


def connect_to_headset(mac: str, name: str) -> bool:
    """Connect to a specific Bluetooth headset."""
    try:
        print(f"Connecting to: {name} ({mac})", file=sys.stderr)
        
        # Check if already connected
        if check_bluetooth_connection_status(mac):
            print(f"Device {name} ({mac}) is already connected via Bluetooth.", file=sys.stderr)
        else:
            # Trust and connect
            bluetoothctl_input = f"trust {mac}\nconnect {mac}\n"
            subprocess.run(
                ["bluetoothctl"],
                input=bluetoothctl_input.encode(),
                capture_output=True,
                check=False,
                timeout=10,
            )
            
            # Wait for connection to establish and verify
            for attempt in range(5):
                time.sleep(1)
                if check_bluetooth_connection_status(mac):
                    break
            else:
                print(f"Warning: Could not verify Bluetooth connection to {name} ({mac}).", file=sys.stderr)
        
        # Try to find and configure PulseAudio card (with retries)
        card_id = find_pulseaudio_card_by_mac(mac, max_retries=5, retry_delay=1.0)
        if card_id:
            subprocess.run(
                ["pactl", "set-card-profile", card_id, "headset-head-unit"],
                capture_output=True,
                check=False,
            )
            print(f"Connected {name} ({mac}) in headset (HFP/HSP) mode.", file=sys.stderr)
        else:
            print(f"Bluetooth connection established to {name} ({mac}), but PulseAudio card not yet available.", file=sys.stderr)
            print(f"This is normal - the device should work as the system default audio device.", file=sys.stderr)
        
        # Save to environment (will be saved to .env by caller)
        os.environ["LAST_HEADSET_MAC"] = mac
        os.environ["LAST_HEADSET_NAME"] = name
        return True
    except Exception as exc:
        print(f"Connection error: {exc}", file=sys.stderr)
        return False


def ensure_headset_connected() -> None:
    """Ensure a Bluetooth headset is connected, auto-connecting or prompting user if needed."""
    # Try auto-connect first
    if LAST_HEADSET_MAC and LAST_HEADSET_NAME:
        if try_auto_connect_headset(LAST_HEADSET_MAC, LAST_HEADSET_NAME):
            return
        # Auto-connect failed, but device might still be available - continue to scan
        print("Auto-connect did not succeed. Scanning for available devices...", file=sys.stderr)
        print("", file=sys.stderr)
    
    # Scan for devices
    print("Scanning for Bluetooth devices...", file=sys.stderr)
    devices = scan_bluetooth_devices()
    
    if not devices:
        print("No Bluetooth devices found.", file=sys.stderr)
        return
    
    # Display menu
    print("", file=sys.stderr)
    print("Available Bluetooth devices:", file=sys.stderr)
    print("=" * 28, file=sys.stderr)
    for idx, (mac, name) in enumerate(devices, 1):
        print(f"{idx}) {name} ({mac})", file=sys.stderr)
    
    # Get user selection
    print("", file=sys.stderr)
    try:
        selection = input("Select a device (1-{}) or 'q' to quit: ".format(len(devices)))
        if selection.lower() == 'q':
            print("Cancelled.", file=sys.stderr)
            return
        
        idx = int(selection)
        if idx < 1 or idx > len(devices):
            print("Invalid selection.", file=sys.stderr)
            return
        
        mac, name = devices[idx - 1]
        if connect_to_headset(mac, name):
            # Save to .env file (look for it in current dir or parent dirs, like dotenv does)
            env_path = None
            current = Path.cwd()
            for parent in [current] + list(current.parents):
                candidate = parent / ".env"
                if candidate.exists():
                    env_path = candidate
                    break
            
            # If not found, use .env in current directory
            if env_path is None:
                env_path = Path(".env")
            
            # Read existing .env or create new content
            if env_path.exists():
                with open(env_path, "r") as f:
                    content = f.read()
                lines = content.splitlines()
            else:
                lines = []
            
            # Update or add headset info
            updated_mac = False
            updated_name = False
            for i, line in enumerate(lines):
                if line.startswith("LAST_HEADSET_MAC="):
                    lines[i] = f'LAST_HEADSET_MAC={mac}'
                    updated_mac = True
                elif line.startswith("LAST_HEADSET_NAME="):
                    lines[i] = f'LAST_HEADSET_NAME="{name}"'
                    updated_name = True
            
            if not updated_mac:
                lines.append(f"LAST_HEADSET_MAC={mac}")
            if not updated_name:
                lines.append(f'LAST_HEADSET_NAME="{name}"')
            
            with open(env_path, "w") as f:
                f.write("\n".join(lines) + "\n")
    except (ValueError, KeyboardInterrupt, EOFError):
        print("Cancelled or invalid input.", file=sys.stderr)


def find_input_device(keyword: str, min_channels: int = 1) -> int | None:
    keyword_lower = keyword.lower()
    for idx, device in enumerate(sd.query_devices()):
        name = device.get("name", "")
        if keyword_lower in name.lower() and device.get("max_input_channels", 0) >= min_channels:
            return idx
    return None


def rms_amplitude(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block))))


def phrase_stream(
    config: ConversationConfig, 
    stop_event: threading.Event | None = None,
    on_voice_activity: Callable[[], None] | None = None,
) -> Iterable[np.ndarray]:
    """Yield successive speech segments detected from the microphone.
    
    Args:
        config: Conversation configuration
        stop_event: Event to stop the stream
        on_voice_activity: Callback called immediately when voice activity is detected
    """

    channels = 1
    block_size = max(1, int(config.sample_rate * config.block_duration))
    silence_blocks_required = max(1, int(config.silence_duration / config.block_duration))
    max_blocks = max(1, int(config.max_record_seconds / config.block_duration))
    min_blocks = max(1, int(config.min_phrase_seconds / config.block_duration))

    q: queue.Queue[np.ndarray] = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        # if status:
        #     print(f"[vad] {status}", file=sys.stderr)
        q.put(indata.copy())

    print("Listening continuously… (Ctrl+C to exit)")
    with sd.InputStream(
        samplerate=config.sample_rate,
        channels=channels,
        dtype="float32",
        blocksize=block_size,
        callback=audio_callback,
        device=config.input_device_index,
    ):
        recording = False
        silence_blocks = 0
        collected: List[np.ndarray] = []
        block_counter = 0

        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                block = q.get(timeout=0.1)
            except queue.Empty:
                continue
            block_counter += 1 if recording else 0
            amp = rms_amplitude(block)

            if config.show_levels:
                meter_width = 40
                normalized = min(1.0, amp / max(config.activation_threshold, 1e-6))
                filled = int(normalized * meter_width)
                bar = "#" * filled + "-" * (meter_width - filled)
                suffix = "REC"
                if recording and amp >= config.activation_threshold:
                    suffix = "REC (*)"
                sys.stderr.write(
                    f"\rLevel {amp:0.3f} |{bar}| {suffix}"
                )
                sys.stderr.flush()

            if not recording:
                if amp >= config.activation_threshold:
                    # Voice activity detected - pause immediately
                    if on_voice_activity:
                        on_voice_activity()
                    recording = True
                    collected = [block]
                    silence_blocks = 0
                    block_counter = 1
            else:
                collected.append(block)
                if amp < config.silence_threshold:
                    silence_blocks += 1
                else:
                    silence_blocks = 0

                if silence_blocks >= silence_blocks_required or block_counter >= max_blocks:
                    duration = len(collected) * config.block_duration
                    recording = False
                    silence_blocks = 0
                    block_counter = 0

                    if len(collected) < min_blocks:
                        print("Discarded short segment.", file=sys.stderr)
                        collected = []
                        meter_break(config.show_levels)
                        continue

                    audio = np.concatenate(collected, axis=0)
                    collected = []
                    meter_break(config.show_levels)
                    yield audio



def transcribe_audio(model: WhisperModel, audio_path: Path, show_levels: bool) -> str:
    print("Transcribing with Faster Whisper…", file=sys.stderr)
    start_time = time.perf_counter()
    
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=1,  # Greedy decoding for speed
        temperature=0,
    )
    
    # faster-whisper returns a generator, so we must iterate to get results
    text_segments = []
    for segment in segments:
        text_segments.append(segment.text)
        
    text = " ".join(text_segments).strip()
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    # print(f"Whisper transcription completed in {duration:.2f} seconds", file=sys.stderr)
    meter_break(show_levels)
    print(f"You said: {text}")
    return text


class SentenceAccumulator:
    """Accumulates text chunks and yields complete sentences."""
    def __init__(self):
        self.buffer = ""
        # Simple sentence endings. Can be improved with nltk/spacy if needed,
        # but kept simple for speed and dependency minimization.
        self.endings = {'.', '!', '?', ':'}
        
    def add(self, text: str) -> Iterable[str]:
        self.buffer += text
        while True:
            # Find the first sentence ending
            earliest_end = -1
            best_mark = None
            
            for mark in self.endings:
                idx = self.buffer.find(mark)
                if idx != -1:
                    if earliest_end == -1 or idx < earliest_end:
                        earliest_end = idx
                        best_mark = mark
            
            if earliest_end == -1:
                break
                
            # We found a sentence end. 
            # Check if it looks like an abbreviation (e.g. "Mr.", "1.5")
            # This is a basic heuristic.
            candidate = self.buffer[:earliest_end+1]
            remainder = self.buffer[earliest_end+1:]
            
            # Very basic abbreviation check: if the "sentence" is too short (<=3 chars) 
            # and ends in dot, treat it as part of next sentence (e.g. "Mr.")
            # unless it's just "No." or "Ok."
            if best_mark == '.' and len(candidate.strip()) <= 3 and candidate.strip().lower() not in ["no.", "ok.", "hi."]:
                 # It might be an abbreviation, simplified behavior: just wait for more context or next splitter
                 # But sticking to simple split for now to ensure low latency.
                 # To do this properly requires lookahead.
                 pass

            yield candidate.strip()
            self.buffer = remainder

    def flush(self) -> Iterable[str]:
        if self.buffer.strip():
            yield self.buffer.strip()
        self.buffer = ""


def query_ollama_streaming(
    model_name: str,
    messages: list[dict[str, str]],
    interruptable: bool = True,
    stop_event: threading.Event | None = None,
    options: dict[str, Any] | None = None,
) -> Iterable[str]:
    """
    Yields complete sentences from Ollama.
    """
    client = ollama.Client()
    
    # Extract just the new user message for logging
    user_content = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "...")
    print(f"Querying Ollama '{model_name}': {user_content[:60]}...", file=sys.stderr)

    try:
        stream = client.chat(
            model=model_name,
            messages=messages,
            stream=True,
            keep_alive=-1,
            options=options or {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 100,
                "num_ctx": 2048,
            },
        )
    except Exception as e:
        print(f"Ollama error: {e}", file=sys.stderr)
        return

    accumulator = SentenceAccumulator()

    for chunk in stream:
        if stop_event and stop_event.is_set():
            # print("Ollama query interrupted.", file=sys.stderr)
            return

        content = ""
        # Extract content from various chunk formats
        if isinstance(chunk, dict):
            content = chunk.get("message", {}).get("content", "")
        else:
             # Object access
            msg = getattr(chunk, "message", None)
            if msg:
                content = getattr(msg, "content", "")
        
        if content:
            for sentence in accumulator.add(content):
                yield sentence
    
    for sentence in accumulator.flush():
        yield sentence


def synthesize_with_piper(
    voice: PiperVoice, text: str, output_wav: Path
) -> None:
    print("Synthesizing speech with Piper…", file=sys.stderr)
    audio_iter = voice.synthesize(text)
    base_sample_rate = int(
        getattr(voice, "sample_rate", getattr(getattr(voice, "config", {}), "sample_rate", DEFAULT_SAMPLE_RATE))
    )

    def extract_audio_field(obj: Any) -> Any | None:
        field_candidates = (
            "audio",
            "_audio",
            "buffer",
            "data",
            "pcm",
            "samples",
            "wave",
            "waveform",
            "frames",
            "chunk",
            "audio_int16_bytes",
            "audio_int16_array",
            "audio_float_array",
            "_audio_int16_bytes",
            "_audio_int16_array",
        )
        for attr in field_candidates:
            value = getattr(obj, attr, None)
            if value is not None:
                return value
        return None

    def to_bytes_and_rate(chunk: Any) -> tuple[bytes, int | None]:
        current_rate: int | None = None
        data: Any = chunk

        if AudioChunk is not None and isinstance(chunk, AudioChunk):
            maybe = extract_audio_field(chunk)
            if maybe is not None:
                data = maybe
            current_rate = getattr(chunk, "sample_rate", None)
        elif isinstance(chunk, dict):
            if "audio" in chunk:
                data = chunk["audio"]
            else:
                for key in ("buffer", "data", "pcm", "samples"):
                    if key in chunk:
                        data = chunk[key]
                        break
            current_rate = chunk.get("sample_rate")
        else:
            maybe = extract_audio_field(chunk)
            if maybe is not None:
                data = maybe
                current_rate = getattr(chunk, "sample_rate", None)

        if isinstance(data, np.ndarray):
            return data.astype(np.int16).tobytes(), current_rate
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data), current_rate
        if isinstance(data, (tuple, list)) and data:
            first = data[0]
            if isinstance(first, np.ndarray):
                return first.astype(np.int16).tobytes(), current_rate
            if isinstance(first, (bytes, bytearray, memoryview)):
                return bytes(first), current_rate
        if data is chunk and hasattr(chunk, "__iter__") and not isinstance(
            chunk, (str, bytes, bytearray, memoryview)
        ):
            try:
                arr = np.fromiter(chunk, dtype=np.int16)
                return arr.tobytes(), current_rate
            except TypeError:
                pass

        # Fall back to generic bytes conversion if possible
        try:
            return bytes(data), current_rate
        except Exception as exc:
            raise TypeError(
                f"Unsupported Piper chunk type: {type(chunk)!r} (available attrs: {dir(chunk)})"
            ) from exc

    with wave.open(str(output_wav), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(base_sample_rate)

        for chunk in audio_iter:
            data, maybe_rate = to_bytes_and_rate(chunk)
            if maybe_rate and maybe_rate != base_sample_rate:
                wav_file.setframerate(maybe_rate)
            wav_file.writeframes(data)


class TTSWorker:
    """
    Handles background TTS synthesis and audio buffering.
    Input: Text sentences
    Output: Audio chunks in a thread-safe queue for the player
    """
    def __init__(self, voice: PiperVoice, sample_rate: int):
        self.voice = voice
        self.sample_rate = sample_rate
        self.input_queue: queue.Queue[str | None] = queue.Queue()
        self.audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()  # For pausing synthesis
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.started = False

    def start(self):
        if not self.started:
            self.stop_event.clear()
            self.pause_event.clear()
            self.thread.start()
            self.started = True

    def stop(self):
        self.stop_event.set()
        self.pause_event.clear()  # Unpause to allow thread to exit
        # Drain queues to unblock
        while not self.input_queue.empty():
            try: self.input_queue.get_nowait()
            except queue.Empty: pass
        self.input_queue.put(None) # Sentinel

    def pause(self):
        """Pause TTS synthesis (stops processing new text, but keeps worker alive)"""
        self.pause_event.set()

    def resume(self):
        """Resume TTS synthesis"""
        self.pause_event.clear()

    def flush(self):
        """Flush all queued audio chunks and pending text"""
        # Clear audio queue
        while not self.audio_queue.empty():
            try: 
                self.audio_queue.get_nowait()
            except queue.Empty: 
                pass
        # Clear input queue
        while not self.input_queue.empty():
            try: 
                self.input_queue.get_nowait()
            except queue.Empty: 
                pass

    def put_text(self, text: str):
        self.input_queue.put(text)

    def _worker_loop(self):
        while not self.stop_event.is_set():
            # Check if paused
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue
                
            try:
                text = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if text is None:
                break

            # Synthesize
            try:
                # print(f"[TTS] Synthesizing: {text[:30]}...", file=sys.stderr)
                stream = self.voice.synthesize(text)
                for chunk in stream:
                    if self.stop_event.is_set():
                        break
                    # Check if paused during synthesis
                    if self.pause_event.is_set():
                        break
                    
                    audio_array = None
                    
                    # 1. Try to get float array directly (most efficient)
                    if hasattr(chunk, "audio_float_array") and chunk.audio_float_array is not None:
                        audio_array = chunk.audio_float_array.astype(np.float32)
                    
                    # 2. Try to get bytes
                    elif hasattr(chunk, "audio_int16_bytes") and chunk.audio_int16_bytes is not None:
                         audio_data = chunk.audio_int16_bytes
                         audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    elif hasattr(chunk, "bytes") and chunk.bytes is not None:
                         audio_data = chunk.bytes
                         audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                     # 3. Fallback to generic bytes conversion
                    else:
                        try:
                            audio_data = bytes(chunk)
                            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        except Exception:
                            pass

                    if audio_array is None:
                         print(f"TTS Warning: Could not extract audio from {type(chunk)}: {dir(chunk)}", file=sys.stderr)
                         continue

                    self.audio_queue.put(audio_array)
            except Exception as e:
                print(f"TTS Error: {e}", file=sys.stderr)
        
        self.audio_queue.put(None) # End of audio stream


def play_audio_stream(
    audio_queue: queue.Queue[np.ndarray | None],
    sample_rate: int,
    interrupt_event: threading.Event,
    interruptable: bool = True,
    save_path: Path | None = None,
) -> None:
    # print(f"Starting audio playback stream at {sample_rate}Hz...", file=sys.stderr)
    try:
        # block_size = max(1024, sample_rate // 10) # 100ms latency
        # Smaller block size for lower latency?
        block_size = 1024 

        wav_file = None
        if save_path:
            wav_file = wave.open(str(save_path), "wb")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2) # 16-bit PCM
            wav_file.setframerate(sample_rate)

        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
        ) as stream:
             while True:
                if interruptable and interrupt_event.is_set():
                    # Gracefully stop playback when interrupted
                    # print("Playback interrupted by event.", file=sys.stderr)
                    break

                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if chunk is None:
                    # End of stream
                    break

                # Write chunk to stream
                # sd.OutputStream.write expects (frames, channels)
                if chunk.ndim == 1:
                    chunk = chunk[:, np.newaxis]
                
                stream.write(chunk)

                if wav_file:
                    # Convert float32 back to int16 for wav
                    # formatting: clip to [-1, 1], scale to 32767
                    clipped = np.clip(chunk, -1.0, 1.0)
                    int16_data = (clipped * 32767).astype(np.int16)
                    wav_file.writeframes(int16_data.tobytes())
        
        if wav_file:
            wav_file.close()
                
    except Exception as e:
        print(f"Playback Error: {e}", file=sys.stderr)


def play_audio(audio_path: Path, interrupt_event: threading.Event, interruptable: bool = True) -> bool:
    data, samplerate = sf.read(audio_path, dtype="float32")
    if data.ndim == 1:
        data = data[:, np.newaxis]
    frames_total = data.shape[0]
    channels = data.shape[1]
    block = max(1024, samplerate // 10)

    interrupt_event.clear()

    with sd.OutputStream(
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
    ) as stream:
        cursor = 0
        while cursor < frames_total:
            if interruptable and interrupt_event.is_set():
                stream.abort()
                stream.stop()
                return False

            end = min(cursor + block, frames_total)
            chunk = data[cursor:end]
            stream.write(chunk)
            cursor = end

    return True


def is_reset_command(text: str) -> bool:
    """Check if the transcribed text is a command to reset the conversation."""
    text_lower = text.lower().strip()
    reset_phrases = [
        "start over",
        "lets start over",
        "let's start over",
        "let us start over",
        "reset",
        "clear",
        "new conversation",
        "begin again",
        "start fresh",
        "restart",
        "forget everything",
        "forget that",
    ]
    return any(phrase in text_lower for phrase in reset_phrases)


def build_initial_messages(system_prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}]


def run_conversation(config: ConversationConfig) -> None:
    whisper_model = load_whisper_model(config.whisper_model, config.whisper_compute_type)
    messages = build_initial_messages(config.system_prompt)
    assert config.piper_voice is not None
    piper_voice = load_piper_voice(
        config.piper_voice,
        config.piper_config,
        length_scale=config.piper_length_scale,
        noise_scale=config.piper_noise_scale,
        noise_w=config.piper_noise_w,
    )

    # Create session directory
    log_dir = ensure_log_dir()
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = log_dir / f"session-{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = session_dir / "session.jsonl"
    
    append_log_line(
        log_file,
        {
            "type": "session_start",
            "session_id": session_id,
            "config": asdict(config),
            "env_overrides": {
                k: v for k, v in os.environ.items() if k in [
                    "SYSTEM_PROMPT", "OLLAMA_MODEL", "WHISPER_MODEL", "PIPER_VOICE",
                    "SAMPLE_RATE", "INPUT_DEVICE_KEYWORD", "ACTIVATION_THRESHOLD",
                    "SILENCE_THRESHOLD", "SILENCE_DURATION", "MIN_PHRASE_SECONDS",
                    "BLOCK_DURATION", "INTERRUPTABLE", "FLUSH_ON_INTERRUPT",
                    "LOG_ROOT", "HISTORY_TRUNCATION_LIMIT", "OLLAMA_TEMPERATURE",
                    "OLLAMA_TOP_P", "OLLAMA_TOP_K", "OLLAMA_NUM_PREDICT", "OLLAMA_NUM_CTX",
                    "PLAYBACK_SPEED"
                ]
            }
        },
    )

    # Ensure Bluetooth headset is connected
    ensure_headset_connected()
    # Reload environment variables in case headset info was updated
    dotenv.load_dotenv(override=True)
    # Update module-level variables after reload
    global LAST_HEADSET_MAC, LAST_HEADSET_NAME
    LAST_HEADSET_MAC = os.environ.get("LAST_HEADSET_MAC")
    LAST_HEADSET_NAME = os.environ.get("LAST_HEADSET_NAME")
    # Update the input device keyword if we have a saved headset name
    if LAST_HEADSET_NAME and not config.input_device_keyword:
        config.input_device_keyword = LAST_HEADSET_NAME

    if config.input_device_keyword:
        device_index = find_input_device(config.input_device_keyword)
        if device_index is not None:
            config.input_device_index = device_index
            dev_info = sd.query_devices(device_index)
            print(
                f"Using input device #{device_index}: {dev_info['name']}",
                file=sys.stderr,
            )
        else:
            # Check if we just connected a Bluetooth device
            bluetooth_connected = False
            if LAST_HEADSET_MAC:
                bluetooth_connected = check_bluetooth_connection_status(LAST_HEADSET_MAC)
            
            # Get the default input device to see what we're actually using
            try:
                default_device = sd.query_devices(kind='input')
                default_name = default_device.get('name', 'unknown') if default_device else 'unknown'
                
                if bluetooth_connected:
                    print(
                        f"Note: Bluetooth device '{config.input_device_keyword}' is connected, "
                        f"but not found by exact name match in audio devices.",
                        file=sys.stderr,
                    )
                    print(
                        f"Using system default input device: '{default_name}' "
                        f"(this is likely the Bluetooth headset).",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Note: no input device found matching '{config.input_device_keyword}'. "
                        f"Falling back to system default: '{default_name}'.",
                        file=sys.stderr,
                    )
            except Exception:
                if bluetooth_connected:
                    print(
                        f"Note: Bluetooth device '{config.input_device_keyword}' is connected, "
                        f"but not found by exact name match. Using system default audio device.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Note: no input device found matching '{config.input_device_keyword}'. "
                        "Falling back to system default.",
                        file=sys.stderr,
                    )

    stop_event = threading.Event()
    segment_queue: queue.Queue[np.ndarray] = queue.Queue()
    pending_segments: list[np.ndarray] = []
    playback_interrupt = threading.Event()
    pending_concatenation = ""
    
    # Shared references to current TTS worker and playback thread for interruption
    current_tts_worker: TTSWorker | None = None
    current_playback_thread: threading.Thread | None = None
    current_abort_event: threading.Event | None = None

    def on_voice_activity_detected():
        """Called immediately when voice activity is detected (before phrase is complete)"""
        if config.interruptable:
            # Immediately pause TTS synthesis and flush audio queue
            if current_tts_worker is not None:
                current_tts_worker.pause()
                current_tts_worker.flush()  # Flush queued audio chunks
            playback_interrupt.set()
            # Also abort current LLM generation if in progress
            if current_abort_event is not None:
                current_abort_event.set()

    def producer() -> None:
        try:
            for segment in phrase_stream(config, stop_event=stop_event, on_voice_activity=on_voice_activity_detected):
                segment_queue.put(segment)
                # Note: pausing already happened in on_voice_activity_detected when voice was first detected
        except Exception as exc:
            print(f"Phrase producer error: {exc}", file=sys.stderr)

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    # Use the session directory for audio files instead of a temp dir
    try:
        # Initialize TTS Worker
        tts_worker = TTSWorker(piper_voice, config.sample_rate)
        tts_worker.start()

        # Announce readiness
        # print("Ready to chat...", file=sys.stderr)
        
        # startup_text = "I am connected and ready to chat."
        print("Let's begin...", file=sys.stderr)
        startup_text = "Let's begin."
        # if LAST_HEADSET_NAME:
        #      # Clean up name for TTS? "OpenRun Pro 2 by Shokz" is fine.
        #      startup_text = f"I am connected to the {LAST_HEADSET_NAME} and ready to chat."
        
        startup_audio = session_dir / "startup.wav"
        synthesize_with_piper(
            piper_voice,
            startup_text,
            startup_audio,
        )
        # Clear interrupt before playing startup sound
        playback_interrupt.clear()
        play_audio(startup_audio, playback_interrupt, interruptable=False)

        # Generate and play the first assistant message to start the conversation
        print("Generating initial greeting...", file=sys.stderr)
        
        # Create a TTS worker for the initial message
        initial_tts_worker = TTSWorker(piper_voice, config.sample_rate)
        initial_tts_worker.start()
        current_tts_worker = initial_tts_worker
        
        abort_event = threading.Event()
        current_abort_event = abort_event
        
        # Start playback thread for initial message
        tts_sample_rate = initial_tts_worker.voice.config.sample_rate if initial_tts_worker.voice else config.sample_rate
        playback_interrupt.clear()
        initial_response_audio_path = session_dir / "turn-000-initial-response.wav"
        
        initial_playback_thread = threading.Thread(
            target=play_audio_stream,
            args=(initial_tts_worker.audio_queue, tts_sample_rate, playback_interrupt, config.interruptable, initial_response_audio_path),
            daemon=True
        )
        current_playback_thread = initial_playback_thread
        initial_playback_thread.start()
        
        # Generate initial assistant message
        full_initial_text = ""
        prev_sentence_time = time.perf_counter()
        
        for sentence in query_ollama_streaming(
            config.ollama_model,
            messages,
            interruptable=config.interruptable,
            stop_event=abort_event,
            options={
                "temperature": config.ollama_temperature,
                "top_p": config.ollama_top_p,
                "top_k": config.ollama_top_k,
                "num_predict": config.ollama_num_predict,
                "num_ctx": config.ollama_num_ctx,
            }
        ):
            current_time = time.perf_counter()
            elapsed = current_time - prev_sentence_time
            prev_sentence_time = current_time
            
            print(f"\nAssistant: {sentence} ({elapsed:.2f}s)", flush=True)
            full_initial_text += sentence + " "
            
            # Clean the sentence for TTS
            cleaned_sentence = re.sub(r'[\*#_`~]', '', sentence)
            cleaned_sentence = re.sub(r'\([^\)]*\)', '', cleaned_sentence)
            cleaned_sentence = re.sub(r'\[[^\]]*\]', '', cleaned_sentence)
            cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()
            
            if cleaned_sentence:
                initial_tts_worker.put_text(cleaned_sentence)
        
        initial_tts_worker.put_text(None)  # End of input
        print()  # Newline after response
        
        # Check if generation was interrupted
        initial_interrupted = abort_event.is_set() if abort_event else False
        
        # Wait for playback to finish
        while initial_playback_thread.is_alive():
            if config.interruptable and playback_interrupt.is_set():
                initial_tts_worker.flush()
                initial_tts_worker.stop()
                initial_interrupted = True
                break
            initial_playback_thread.join(timeout=0.1)
        
        # Only add the initial assistant message if it wasn't interrupted
        if not initial_interrupted and full_initial_text.strip():
            messages.append({"role": "assistant", "content": full_initial_text.strip()})
            
            append_log_line(
                log_file,
                {
                    "type": "assistant",
                    "turn": 0,
                    "text": full_initial_text.strip(),
                    "audio_path": str(initial_response_audio_path),
                },
            )
        elif initial_interrupted:
            append_log_line(
                log_file,
                {
                    "type": "assistant_cancelled",
                    "turn": 0,
                },
            )
        
        initial_tts_worker.stop()
        current_tts_worker = None
        current_playback_thread = None
        current_abort_event = None

        turn = 1
        while True:
            try:
                if pending_segments:
                    phrase = pending_segments.pop(0)
                else:
                    phrase = segment_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            raw_audio = session_dir / f"turn-{turn:03d}-input.wav"
            sf.write(raw_audio, phrase, config.sample_rate)

            # If this segment interrupted a previous turn, ensure previous TTS is fully stopped
            if current_tts_worker is not None:
                current_tts_worker.stop()
                current_tts_worker = None
            if current_playback_thread is not None and current_playback_thread.is_alive():
                playback_interrupt.set()
                current_playback_thread.join(timeout=0.5)
                current_playback_thread = None
            if current_abort_event is not None:
                current_abort_event.set()
                current_abort_event = None

            user_text = transcribe_audio(whisper_model, raw_audio, config.show_levels)
            if not user_text:
                print("Did not catch that. Let's try again.")
                # Clear interrupt flag since this didn't result in a query
                playback_interrupt.clear()
                continue

            if pending_concatenation:
                if config.flush_on_interrupt:
                    print(f"Flushing previous input: '{pending_concatenation}' (flush on interrupt enabled)", file=sys.stderr)
                    pending_concatenation = ""
                else:
                    print(f"Concatenating previous input: '{pending_concatenation}' + '{user_text}'", file=sys.stderr)
                    user_text = f"{pending_concatenation} {user_text}"
                    pending_concatenation = ""

            # Check for reset command
            if is_reset_command(user_text):
                messages = build_initial_messages(config.system_prompt)
                print("Conversation reset. Starting fresh.", file=sys.stderr)
                append_log_line(
                    log_file,
                    {
                        "type": "reset",
                        "turn": turn,
                        "text": user_text,
                        "audio_path": str(raw_audio),
                    },
                )
                # Synthesize and play reset confirmation message
                reset_audio = session_dir / f"turn-{turn:03d}-reset.wav"
                synthesize_with_piper(
                    piper_voice,
                    "Ok, starting over",
                    reset_audio,
                )
                play_audio(reset_audio, playback_interrupt, interruptable=config.interruptable)
                turn += 1
                continue

            messages.append({"role": "user", "content": user_text})
            
            # Truncate history: Keep generic system prompt + last N messages
            # This prevents the context from growing indefinitely and slowing down prefill.
            limit = config.history_truncation_limit
            if len(messages) > limit:
                # Keep system prompt (index 0) and the last (limit - 1) messages
                # We subtract 1 to account for the system prompt
                num_to_keep = limit - 1
                messages = [messages[0]] + messages[-num_to_keep:]

            append_log_line(
                log_file,
                {
                    "type": "user",
                    "turn": turn,
                    "text": user_text,
                    "audio_path": str(raw_audio),
                },
            )

            abort_event = threading.Event()
            
            # --- New Streaming Implementation ---
            
            # 1. Start TTS Worker
            tts_worker = TTSWorker(piper_voice, config.sample_rate)
            tts_worker.start()
            
            # Store references for interruption handling
            current_tts_worker = tts_worker
            current_abort_event = abort_event
            
            # 2. Check interruption function
            def check_interrupt():
                if not config.interruptable:
                    return False
                try:
                    new_segment = segment_queue.get_nowait()
                    pending_segments.append(new_segment)
                    # Pause and flush TTS (already done by producer, but ensure it here too)
                    tts_worker.pause()
                    tts_worker.flush()
                    playback_interrupt.set()
                    abort_event.set()
                    tts_worker.stop()
                    return True
                except queue.Empty:
                    return False

            # 3. Stream from Ollama -> TTS Worker
            full_assistant_text = ""
            interrupted = False
            
            # Start a thread to monitor interruptions during LLM generation?
            # Or just check periodically in the generator loop? 
            # The generator loop runs in main thread, so we can check there.
            
            # Start Playback Thread immediately
            # Use the TTS voice's sample rate for playback
            tts_sample_rate = tts_worker.voice.config.sample_rate if tts_worker.voice else config.sample_rate
            
            # Clear any stale interrupt from the user's input
            playback_interrupt.clear()
            
            response_audio_path = session_dir / f"turn-{turn:03d}-response.wav"

            playback_thread = threading.Thread(
                target=play_audio_stream,
                args=(tts_worker.audio_queue, tts_sample_rate, playback_interrupt, config.interruptable, response_audio_path),
                daemon=True
            )
            current_playback_thread = playback_thread
            playback_thread.start()

            # Track timing for each sentence
            prev_sentence_time = time.perf_counter()
            
            for sentence in query_ollama_streaming(
                config.ollama_model, 
                messages,
                interruptable=config.interruptable,
                stop_event=abort_event,
                options={
                    "temperature": config.ollama_temperature,
                    "top_p": config.ollama_top_p,
                    "top_k": config.ollama_top_k,
                    "num_predict": config.ollama_num_predict,
                    "num_ctx": config.ollama_num_ctx,
                }
            ):
                if check_interrupt():
                    interrupted = True
                    break
                
                # Calculate time elapsed since previous sentence
                current_time = time.perf_counter()
                elapsed = current_time - prev_sentence_time
                prev_sentence_time = current_time
                
                # Print with timing information
                print(f"\nAssistant: {sentence} ({elapsed:.2f}s)", flush=True)
                full_assistant_text += sentence + " "
                
                # --- CLEANING ---
                # Remove asterisks, headers, bullets, parenthesis/brackets blocks (e.g. [laughs])
                cleaned_sentence = re.sub(r'[\*#_`~]', '', sentence)                   # Remove markdown chars
                cleaned_sentence = re.sub(r'\([^\)]*\)', '', cleaned_sentence)        # Remove (content)
                cleaned_sentence = re.sub(r'\[[^\]]*\]', '', cleaned_sentence)        # Remove [content]
                cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()     # Normalize whitespace

                if cleaned_sentence:
                     tts_worker.put_text(cleaned_sentence)
                
            tts_worker.put_text(None) # End of input
            print() # Newline after response
            
            if interrupted:
                # New audio input detected that resulted in interruption
                # Flush TTS queue completely and cancel everything
                tts_worker.flush()
                tts_worker.stop()
                # print("Interrupted during generation.", file=sys.stderr)
                append_log_line(log_file, {"type": "assistant_cancelled", "turn": turn})
                 # Rollback
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
                if messages and messages[-1]["role"] == "assistant": # Should not happen yet
                    messages.pop()
                pending_concatenation = user_text
                
                # Wait for playback thread to die (it should see interrupt event)
                playback_thread.join(timeout=1.0)
                # Clear references
                current_tts_worker = None
                current_playback_thread = None
                current_abort_event = None
                turn += 1
                continue

            # Wait for playback to finish naturally
            
            while playback_thread.is_alive():
                if config.interruptable and playback_interrupt.is_set():
                    interrupted = True
                    # Flush TTS queue when interrupted during playback
                    tts_worker.flush()
                    tts_worker.stop()
                    break
                playback_thread.join(timeout=0.1)

            if interrupted:
                 # Interruption during tail playback
                 # Treat as interruption: rollback
                 if messages and messages[-1]["role"] == "user":
                    messages.pop()
                 pending_concatenation = user_text
                 # Clear references
                 current_tts_worker = None
                 current_playback_thread = None
                 current_abort_event = None
                 turn += 1
                 continue
            
            messages.append({"role": "assistant", "content": full_assistant_text.strip()})
            
            append_log_line(
                log_file,
                {
                    "type": "assistant",
                    "turn": turn,
                    "text": full_assistant_text.strip(),
                    "audio_path": str(response_audio_path),
                },
            )
            
            tts_worker.stop() # Cleanup
            # Clear references after successful completion
            current_tts_worker = None
            current_playback_thread = None
            current_abort_event = None

            # Save full response audio for logging (non-blocking or post-hoc?)
            # Re-synthesizing for logs is expensive. Ideally we'd capture the stream.
            # For now, let's skip re-synthesis to save time/resources on Jetson.
            
            turn += 1
    except KeyboardInterrupt:
        print("\nExiting conversation.")
    finally:
        stop_event.set()
        producer_thread.join(timeout=1.0)
        append_log_line(
            log_file,
            {"type": "session_end"},
        )


def main() -> None:
    config = parse_args()
    run_conversation(config)


if __name__ == "__main__":
    main()

