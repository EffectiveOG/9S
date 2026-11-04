#!/usr/bin/env python3
"""List the microphones and speakers Jarvis can use.

Set AUDIO_INPUT_DEVICE / AUDIO_OUTPUT_DEVICE in .env to one of the indexes or
to part of a name shown here. Blank means the macOS default device.
"""

import pyaudio


def main():
    audio = pyaudio.PyAudio()
    try:
        default_in = audio.get_default_input_device_info().get("index")
    except OSError:
        default_in = None
    try:
        default_out = audio.get_default_output_device_info().get("index")
    except OSError:
        default_out = None

    for title, key, default in (("Microphones", "maxInputChannels", default_in),
                                ("Speakers", "maxOutputChannels", default_out)):
        print(f"{title}:")
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if info.get(key, 0) > 0:
                marker = "  <- default" if index == default else ""
                print(f"  {index:>3}  {info['name']}{marker}")
        print()
    audio.terminate()


if __name__ == "__main__":
    main()
