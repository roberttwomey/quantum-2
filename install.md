# Installation Guide

This guide will help you set up and run the voice chat experience.

## Prerequisites

- Python 3.10 or later
- Administrator access (for some installations)
- Internet connection
- Microphone and speakers/headphones

## Step 1: Install Python Dependencies

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 2: Install Ollama

Ollama is required to run the language model. Install it:

**macOS:**
```bash
brew install ollama
```

**Linux:**
Download and install from [https://ollama.ai](https://ollama.ai) or use your distribution's package manager.

**Windows:**
Download the installer from [https://ollama.ai](https://ollama.ai).

Start the Ollama service:

```bash
ollama serve
```

In a new terminal window, pull the required model:

```bash
ollama pull gemma3n:e2b
```

Verify the model is available:

```bash
ollama list
```

You should see `gemma3n:e2b` in the list.

**Note:** Keep the `ollama serve` process running, or set it up to run as a background service.

## Step 3: Install Audio Dependencies

**macOS:**
On macOS, audio support is typically built-in, but you may need to install PortAudio for `sounddevice`:

```bash
brew install portaudio
```

**Linux:**
Install PortAudio and related audio libraries:

```bash
# Ubuntu/Debian:
sudo apt-get install portaudio19-dev python3-pyaudio

# Fedora/RHEL:
sudo dnf install portaudio-devel python3-pyaudio
```

**Windows:**
Audio support should work automatically with `sounddevice`. If you encounter issues, you may need to install additional audio drivers.

## Step 4: Download Piper Voice Model

You need at least one Piper voice model file. Use the helper script:

```bash
python download_voice.py en_GB-alan-medium
```

Or manually download:

```bash
# Create the piper directory if it doesn't exist
mkdir -p piper
cd piper

# Download a voice model (example: English UK, Alan, medium quality)
curl -L -o en_GB-alan-medium.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx"
curl -L -o en_GB-alan-medium.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json"

cd ..
```

To see available voices:
```bash
python download_voice.py --list
```

To list voices for a specific language:
```bash
python download_voice.py --list-language en_GB
```

## Step 5: Configure Environment Variables

Create a `.env` file in the project root (if it doesn't exist):

```bash
touch .env
```

Edit the `.env` file and add the following variables:

```bash
# Ollama model name
BFF_OLLAMA_MODEL=gemma3n:e2b

# Whisper model size (tiny, base, small, medium, large)
BFF_WHISPER_MODEL=tiny

# Path to Piper voice model
BFF_PIPER_VOICE=piper/en_GB-alan-medium.onnx

# Optional: System prompt
BFF_SYSTEM_PROMPT="you are SNAPPER a robot dog. you do not say woof, whir, tail wag. answer in 2 sentences or less."

# Optional: Audio settings
BFF_SAMPLE_RATE=16000
BFF_ACTIVATION_THRESHOLD=0.03
BFF_SILENCE_THRESHOLD=0.015
BFF_SILENCE_DURATION=0.8

# Optional: Input device (leave empty to use system default)
# BFF_INPUT_DEVICE_KEYWORD=
```

Adjust the paths and settings according to your setup.

## Step 6: Grant Microphone Permissions

**macOS:**
1. Go to **System Settings** (or **System Preferences** on older macOS)
2. Navigate to **Privacy & Security** → **Microphone**
3. Enable microphone access for **Terminal** (if running from terminal) or **Python** (if running as a standalone app)

You may be prompted when you first run the script.

**Linux:**
Microphone access should work automatically. If you encounter issues, check PulseAudio/ALSA permissions.

**Windows:**
1. Go to **Settings** → **Privacy** → **Microphone**
2. Enable microphone access for your application

## Step 7: Test the Installation

Test that everything works:

```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Make sure Ollama is running (in another terminal)
# ollama serve

# Run the experience
python run-experience.py --piper-voice piper/en_GB-alan-medium.onnx
```

If you've set `BFF_PIPER_VOICE` in your `.env` file, you can run:

```bash
python run-experience.py
```

## Troubleshooting

### Audio Issues

**Problem:** No audio input/output detected

**Solutions:**
- Check microphone permissions in system settings
- List available audio devices:
  ```bash
  python3 -c "import sounddevice as sd; print(sd.query_devices())"
  ```
- Try specifying an input device explicitly:
  ```bash
  python run-experience.py --input-device-keyword "Built-in Microphone"
  ```

**Problem:** `sounddevice` can't find audio devices

**Solution:**
- Make sure PortAudio is installed (see Step 3)
- Reinstall sounddevice: `pip install --force-reinstall sounddevice`

### Ollama Issues

**Problem:** `ollama` command not found

**Solution:**
- Make sure Ollama is installed (see Step 2)
- Verify Ollama service is running: `ollama serve`

**Problem:** Model not found

**Solution:**
- Pull the model: `ollama pull gemma3n:e2b`
- Verify: `ollama list`
- Check the model name matches in your `.env` file

### Whisper Issues

**Problem:** Slow transcription or out of memory errors

**Solutions:**
- Use a smaller Whisper model (set `BFF_WHISPER_MODEL=tiny` in `.env`)
- On systems with GPU, faster-whisper should use it automatically
- For CPU-only systems, consider using `tiny` or `base` models

**Problem:** `faster-whisper` installation fails

**Solution:**
- On macOS: Make sure you have Xcode Command Line Tools: `xcode-select --install`
- Try installing with: `pip install --upgrade faster-whisper`

### Piper TTS Issues

**Problem:** Voice model not found

**Solution:**
- Verify the path in `.env` or command line argument
- Make sure both `.onnx` and `.json` files are present
- Check file permissions

**Problem:** Audio playback issues

**Solution:**
- Check system audio output settings
- Try a different sample rate: `BFF_SAMPLE_RATE=22050` in `.env`
- Verify audio output device: `python3 -c "import sounddevice as sd; print(sd.query_devices())"`

### Python/Package Issues

**Problem:** `ModuleNotFoundError`

**Solution:**
- Make sure virtual environment is activated
- Reinstall packages: `pip install -r requirements.txt`

**Problem:** Python version mismatch

**Solution:**
- Use Python 3.10 or 3.11 (recommended)
- Check version: `python3 --version`
- Create new venv with specific version: `python3.11 -m venv venv`

### Bluetooth Headset (Optional)

The script supports Bluetooth headsets:

**macOS:**
- Pair your headset through System Settings → Bluetooth
- The script should automatically detect it if it's set as the default input/output device
- You can specify it explicitly: `--input-device-keyword "Your Headset Name"`

**Linux:**
- Use `bluetoothctl` to pair your device
- The script will attempt to auto-connect if a device was previously connected
- Linux-specific Bluetooth commands (`bluetoothctl`, `pactl`) are used for connection management

**Windows:**
- Pair your headset through Settings → Bluetooth
- The script should detect it automatically

## Quick Start Summary

Once everything is set up, you can run the experience with:

```bash
# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Make sure Ollama is running (in another terminal)
# ollama serve

# Run the experience
python run-experience.py
```

Or with explicit voice model:

```bash
python run-experience.py --piper-voice piper/en_GB-alan-medium.onnx
```

## Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Faster Whisper Documentation](https://github.com/guillaumekln/faster-whisper)
- [Piper TTS Documentation](https://github.com/rhasspy/piper)
- [Piper Voices](https://huggingface.co/rhasspy/piper-voices)

## System Requirements

- **RAM:** At least 8GB (16GB recommended for larger Whisper models)
- **Storage:** ~5GB for models and dependencies
- **CPU:** Any modern processor
- **GPU:** Optional, but will be used automatically if available (CUDA for NVIDIA, Metal for Apple Silicon)

## Notes

- The script uses `faster-whisper` for better performance than `openai-whisper`
- On systems with GPU support, PyTorch and faster-whisper will use GPU acceleration automatically
- Bluetooth headset support varies by platform (Linux has the most features)
- The script creates log files in `~/bff/logs/` by default (configurable via `BFF_LOG_ROOT`)

