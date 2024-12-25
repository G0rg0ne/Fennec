"""
This is a boilerplate pipeline 'mfcc'
generated using Kedro 0.19.10
"""
import numpy as np
import matplotlib.pyplot as plt

def monitoring_signal(input):
    fig = None
    audio_figs = {}
    for id, audio in input.items():
        fs, y = audio()
        y = 1.0 * y
        t = np.linspace(0, y.shape[0] / fs, y.shape[0])

        # visualize signal
        fig = plt.figure(figsize=(6, 4))  # put the plot in a variable called fig
        plt.title("sound audio wav")
        plt.plot(t, y)
        plt.show(block=False)
        audio_figs[id] = fig
    return audio_figs