import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
rng = np.random.default_rng(1)
K = 6
freqs = (2 * np.pi / 3) * rng.random(np.floor(K / 2).astype(int))
freqs = np.concatenate((freqs, -freqs))
frequencies = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
spectrum = rng.standard_normal(np.floor(K / 2).astype(int))
spectrum = np.concatenate((spectrum, np.conj(spectrum)))
L = 13
x = np.linspace(-2 * L, 2 * L, 1000)
signal = (spectrum[None, :] * np.exp(1j * x[:, None] * freqs[None, :])).sum(axis=-1).real
windowed_signal = signal.copy()
windowed_signal[np.abs(x) > np.floor(L / 2)] = 0
sampling_locs = np.floor(np.arange(L) - np.floor(L / 2)).astype(int)
sampled_signal = (spectrum[None, :] * np.exp(1j * sampling_locs[:, None] * freqs[None, :])).sum(axis=-1).real

with plt.xkcd():
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot([frequencies[0], frequencies[-1]], [0, 0], '-', color=colors[-3])
    plt.stem(freqs, spectrum, markerfmt='C0o', linefmt='C0-', basefmt='C0-', use_line_collection=True)
    plt.xlim(-2 * np.pi, 2 * np.pi)
    plt.title('Fourier domain')
    plt.subplot(2, 1, 2)
    plt.title('Spatial domain')
    plt.plot([x[0], x[-1]], [0, 0], '-', color=colors[-3])
    plt.plot([-np.floor(L / 2), -np.floor(L / 2)],
             [np.min(signal), np.max(signal)], '--',
             color=colors[4])
    plt.plot([np.floor(L / 2), np.floor(L / 2)],
             [np.min(signal), np.max(signal)], '--',
             color=colors[4])
    plt.plot(x, signal)

periodic_sinc = lambda u: np.sinc(L * u / (2 * np.pi)) / np.sinc(u / (2 * np.pi))
low_pass_filter_periodised_spectrum_sinc = np.sum(
    spectrum[None, :] * periodic_sinc(frequencies[:, None] - freqs[None, :]), axis=-1)
sampling_freqs = np.linspace(-np.pi, np.pi, L + 1)[:-1]
sampled_low_pass_filter_periodised_spectrum_sinc = np.sum(
    spectrum[None, :] * periodic_sinc(sampling_freqs[:, None] - freqs[None, :]), axis=-1)

with plt.xkcd():
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot([frequencies[0], frequencies[-1]], [0, 0], '-', color=colors[-3])
    plt.plot([-np.pi, -np.pi],
             [-1, 1], '--',
             color=colors[3])
    plt.plot([np.pi, np.pi],
             [-1, 1], '--',
             color=colors[3])
    plt.plot(frequencies, periodic_sinc(frequencies), '-', color=colors[1])
    plt.xlim(-2 * np.pi, 2 * np.pi)
    plt.ylim(-0.5, 1.1)
    plt.title('Fourier domain')
    plt.subplot(2, 1, 2)
    plt.title('Spatial domain')
    plt.plot([x[0], x[-1]], [0, 0], '-', color=colors[-3])
    plt.plot([-np.floor(L / 2), -np.floor(L / 2)],
             [0, 1], '--',
             color=colors[4])
    plt.plot([np.floor(L / 2), np.floor(L / 2)],
             [0, 1], '--',
             color=colors[4])
    plt.stem(sampling_locs, 0 * sampled_signal + 1, markerfmt='C3s', linefmt='C1-', basefmt='C1-',
             use_line_collection=True)

with plt.xkcd():
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot([frequencies[0], frequencies[-1]], [0, 0], '-', color=colors[-3])
    plt.plot([-np.pi, -np.pi],
             [np.min(low_pass_filter_periodised_spectrum_sinc), np.max(low_pass_filter_periodised_spectrum_sinc)], '--',
             color=colors[3])
    plt.plot([np.pi, np.pi],
             [np.min(low_pass_filter_periodised_spectrum_sinc), np.max(low_pass_filter_periodised_spectrum_sinc)], '--',
             color=colors[3])
    plt.plot(frequencies, low_pass_filter_periodised_spectrum_sinc, '-', color=colors[1])
    plt.scatter(sampling_freqs, sampled_low_pass_filter_periodised_spectrum_sinc, marker='s', c=colors[3], zorder=4)
    plt.stem(freqs, spectrum, markerfmt='C0o', linefmt='C0-', basefmt='C0-', use_line_collection=True)
    plt.xlim(-2 * np.pi, 2 * np.pi)
    plt.title('Fourier domain')
    plt.subplot(2, 1, 2)
    plt.title('Spatial domain')
    plt.plot([x[0], x[-1]], [0, 0], '-', color=colors[-3])
    plt.plot([-np.floor(L / 2), -np.floor(L / 2)],
             [np.min(signal), np.max(signal)], '--',
             color=colors[4])
    plt.plot([np.floor(L / 2), np.floor(L / 2)],
             [np.min(signal), np.max(signal)], '--',
             color=colors[4])
    plt.plot(x, windowed_signal, '-', color=colors[0])
    plt.stem(sampling_locs, sampled_signal, markerfmt='C3s', linefmt='C1-', basefmt='C1-', use_line_collection=True)

fourier_sequence = np.fft.ifftshift(sampled_low_pass_filter_periodised_spectrum_sinc)
space_sequence = np.sqrt(fourier_sequence.size) * np.fft.fftshift(np.fft.ifft(fourier_sequence, norm='ortho').real)


# with plt.xkcd():
#     plt.figure()
#     plt.subplot(2, 1, 1)
#     plt.plot([frequencies[0], frequencies[-1]], [0, 0], '-', color=colors[-3])
#     plt.plot([-np.pi, -np.pi],
#              [np.min(low_pass_filter_periodised_spectrum_sinc), np.max(low_pass_filter_periodised_spectrum_sinc)], '--',
#              color=colors[3])
#     plt.plot([np.pi, np.pi],
#              [np.min(low_pass_filter_periodised_spectrum_sinc), np.max(low_pass_filter_periodised_spectrum_sinc)], '--',
#              color=colors[3])
#     plt.plot(frequencies, low_pass_filter_periodised_spectrum_sinc, '-', color=colors[1])
#     plt.scatter(sampling_freqs, sampled_low_pass_filter_periodised_spectrum_sinc, marker='s', c=colors[3], zorder=4)
#     plt.stem(freqs, spectrum, markerfmt='C0o', linefmt='C0-', basefmt='C0-', use_line_collection=True)
#     plt.xlim(-2 * np.pi, 2 * np.pi)
#     plt.title('Fourier domain')
#     plt.subplot(2, 1, 2)
#     plt.title('Spatial domain')
#     plt.plot([x[0], x[-1]], [0, 0], '-', color=colors[-3])
#     plt.plot([-np.floor(L / 2), -np.floor(L / 2)],
#              [np.min(signal), np.max(signal)], '--',
#              color=colors[4])
#     plt.plot([np.floor(L / 2), np.floor(L / 2)],
#              [np.min(signal), np.max(signal)], '--',
#              color=colors[4])
#     plt.plot(x, windowed_signal, '-', color=colors[0])
#     plt.stem(sampling_locs, sampled_signal, markerfmt='C3s', linefmt='C1-', basefmt='C1-', use_line_collection=True)
#     plt.scatter(sampling_locs, space_sequence, s=100, marker='*', c=colors[-1], zorder=4)
