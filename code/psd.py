import librosa
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load('data/groov3.mp3')
S = np.abs(librosa.stft(y))

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                       ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

fig.savefig('images/psd.png')