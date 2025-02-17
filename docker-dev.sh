#!/bin/sh

docker run --runtime=nvidia --device /dev/snd --device /dev/i2c-7 --network=host -v=./model:/model -v=./voices:/usr/local/app/voices -v=./start-voice-assistant.py:/usr/local/app/start-voice-assistant.py -v=./coqui_tts.py:/usr/local/app/conqui_tts.py -v=./local:/root/.local -v=./cache:/root/.cache -it voice-assistant:latest /bin/bash

