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

- Vention USB soundcard for sound output (can be something else but should at least support 48000 sample rate)
- Logitech Blue Snowball Ice for the mic (can be something else)
- Nvidia Jetson Orin Nano 8GB Developer Board
- Yahboom Jetson-Cube Case (Optional - but the case and LED lights makes this prettier)

Running in Docker
=================

Docker is the preferred way to run this.

Download LLM related models from Huggingface.

For this voice assistant OpenOrca Mistral 7B in GGUF format was used (mistral-7b-openorca.Q2_K.gguf). The specific model can be changed, though you have to be mindful of the model size. Place it under the /model directory (or wherever you mapped this location to using Docker). If you do change the model consider changing the prompting as well in the format_mistral_orca function.

Building...

clone the project

```
git clone git@github.com:jedld/jetson-voice-assistant.git
cd jetson-voice-assistant
docker build . -t latest
```

Running... (replace home directory references wih your project path)

```
docker run --runtime=nvidia --device /dev/snd --device /dev/i2c-7 --network=host -v=./model:/model -v=./voices:/usr/local/app/voices. -v=./local:/root/.local -v=./cache:/root/.cache -t voice-assistant:latest
```

The only thing you need to put a file in is in /model where you have to place the downloaded GGUF Mistral 7B model into.

Docker Environment Variables
============================

| Env Var Name                | Default Value                            | Description                                                                                          |
|-----------------------------|------------------------------------------|------------------------------------------------------------------------------------------------------|
| LLM_MODEL                   | mistral-7b-openorca.Q2_K.gguf            | The filename of the model. Possible to use others but haven't tried.                                 |
| WHISPER_MODEL               | base                                     | OpenAI whisper model name. See Open AI's whisper for details. Automatically downloaded during first boot. |
| VOCALIZER_CLASS             | CoquiTTS                                 | Default to CoquiTTS if using Coqui's framework for text to speech.                                    |
| INPUT_DEVICE                |                                          | Name of the Input device. Can be checked using `aplay -l`.                                            |
| OUTPUT_DEVICE               |                                          | Name of the Output device. Can be checked using `aplay -l`.                                           |
| OUTPUT_DEVICE_SAMPLE_RATE   | 48000                                    | Sample rate of the output device. Default is 48000.                                                  |
| CONTEXT_LENGTH              | 340                                      | LLM context length. Increasing this may cause Jetson to crash due to OOM.                             |

License
=======

Main python files here are covered by the MIT License. Python libraries are covered by their own licenses.




