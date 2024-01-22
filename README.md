Voice Assistant for the Nvidia Jetson Orin Nano
===============================================

This is an all-in-one voice assistant for the Nvidia Jetson Orin Nano 8GB.
The components here where specifically chosen to fit in the Orin Nano's limited 8GB memory.

It Uses the following components:

1. ASR - Open AI's whisper (base)
2. Wakeword (Hotword detection) - OpenWakeWord
3. Large Language Model served by llama_cpp - (LLM) OpenOrca Mistral 7B quantized (Q2_K) in gguf format
4. Voice Synthesis - CoquiTTS Using the Tachotron 2 DDC model
5. Additional we are also using noisereduce as the denoiser to help with background noise

We make full use of cuda acceleration for all the models whenever possible to acheive the fastest response latency.

The code still needs some cleanup and there are hardcoded paths.


Jetson Requirements
===================

This app requires Jetpack 6.0 DP to be installed in your Jetson Orin Nano


Hardware
========

This is the hardware I used. Other devices are possible:

- Vention USB soundcard for sound output
- Logitech Blue Snowball Ice for the mic

Running in Docker
=================

Building...

```
docker build . -t latest
```

Running... (replace home directory references wih your project path)

```
docker run --runtime=nvidia --device /dev/snd --device /dev/i2c-7 --network=host -v=/home/joseph/workspace/voice-assistant/model:/model -v=/home/joseph/workspace/voice-assistant/voices:/usr/local/app/voices -v=/home/joseph/workspace/voice-assistant/start-voice-assistant.py:/usr/local/app/start-voice-assistant.py -v=/home/joseph/workspace/voice-assistant/coqui_tts.py:/usr/local/app/conqui_tts.py -v=/home/joseph/workspace/voice-assistant/local:/root/.local -v=/home/joseph/workspace/voice-assistant/cache:/root/.cache -t latest
```