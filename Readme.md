## AI Music Generation

### Using Stable diffusion we can generate spectrogram images which we then can use to create music!
In this project, we leverage Stable Diffusion, a cutting-edge technique in artificial intelligence, to generate spectrogram images from audio data and vice versa.

Spectrogram Image Generation:
    Using the spectrogram images derived from Stable Diffusion, we obtain visual representations that encapsulate the frequency content and amplitude variations of audio tracks. These images are essential for the subsequent stages of music synthesis.

Music Synthesis:
    The spectrogram images are processed and interpreted to synthesize music compositions. By leveraging deep learning frameworks and audio processing libraries, we transform these visual representations back into audible forms, effectively translating complex spectrogram data into cool music!

Models and Training:
By using Dreambooth we can train our own models specifically for spectrogram generation. This require a dataset of spectrogram images, which can be created also with this project using audio_to_spectogram.py.

Here is a good tutorial from Stable diffusion Dreambooth training:
https://stable-diffusion-art.com/dreambooth/#What_you_need_to_train_Dreambooth

To obtain music for training we can for example use jamendo.com(no affiliation) which provides royalty free music. They also displays BPM for their songs as we want to train on songs that are similar in nature.


### Descriptions:
In the project you can also find some sample data from a test run and thus can be greatly improved.

#### audio.py  
This script processes spectrogram images generated in Stable diffusion to generate corresponding MP3 audio files.

We can chnage some values to increase the tempo and feel of the audio:
    clip_duration_ms = 40000
    bins_per_image = image.height
    n_mels = image.height
    window_duration_ms = 100
    padded_duration_ms = 400
    step_size_ms = 10

This script is based on this repo: https://github.com/chavinlo/riffusion-manipulation/blob/master/img2audio.py


#### audio_to_spectrogram.py
Creates a spectrogram from an audio file. This is helpful if we already have an audio file that we for example want to tweak. We can for example input these spectrograms into stable diffusion to get a different sound.

This code is also based on: https://github.com/chavinlo/riffusion-manipulation/blob/master/file2img.py

#### crossfade_audio_files.py
This script blends two audio files (1.wav and 2.wav) to create a seamless transition using beat alignment.

#### interpolate_and_blend
This script blends pairs of images (RGB.png) by splitting them into frequency bands and applying a smooth transition between them. We also create intermediate spectrograms(interpolation) and blend them to try to create a more gradual transition and longer audio. (Changing the values are strongly encouraged to get better transitions).

#### audio_join_files.py
This script processes a folder of images (*.png, *.jpg, *.jpeg, *.bmp) containing spectrogram representations of audio and concatenates them into a single audio file (joined_audio.mp3).  
-> This is useful for creating a single audiofile from the interpolated images

#### concat_spectrograms.py
This script concatenates multiple images horizontally into a single combined image and saves it to an output folder. This is useful if we want do do img2img generation in Stable diffusion to create variations of our music. My graphics card(8gb) limit me to 512x4096 but it is possible to generate even longer music using this approach. This method also generates seamless transitions by default.

#### img2rgb.py
Simple script to convert images to RGB if they aren't already. Stable diffusion generates RGB images by default.

