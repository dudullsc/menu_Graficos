import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from scipy.signal import butter, filtfilt

def plot_fft(Ts, signal, fmin, fmax, color):
    signal = np.concatenate([signal, np.zeros(int(1e6))])
    ftt = fftshift(np.abs(fft(signal)) / len(signal))
    f = np.linspace(-1 / (2 * Ts), 1 / (2 * Ts), len(ftt))
    plt.plot(f, ftt, color)
    plt.xlim(fmin, fmax)
    plt.ylim(0, max(ftt) * 1.1)

def AMDSB():
    Tw = 0.001
    Ts = 1e-8

    F0 = 160e3
    E0 = 10

    Fm = 4e3
    Em = 1

    t = np.arange(0, Tw, Ts)

    e0 = E0 * np.cos(2 * np.pi * t * F0)
    em = Em * np.cos(2 * np.pi * t * Fm)
    k = 8
    m = k * (Em / E0)

    Ptotal = (1 / 50) * ((E0 ** 2) / 2 + 2 * (((m ** 2) * (E0 ** 2)) / 8))

    e = (1 + m * em) * e0

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(t, e0)
    plt.title('Portadora')

    plt.subplot(3, 1, 2)
    plt.plot(t, em)
    plt.title('Sinal Modulante')

    plt.subplot(3, 1, 3)
    plt.plot(t, e)
    plt.title(f'Sinal Modulado. m = {m}')

    plt.figure(2)
    plt.subplot(3, 1, 1)
    plot_fft(Ts, em, 0, 8e3, 'b')
    plt.title('Sinal de informação no domínio da frequência')

    plt.subplot(3, 1, 2)
    plot_fft(Ts, e0, 130e3, 190e3, 'b')
    plt.title('Portadora no domínio da frequência')

    plt.subplot(3, 1, 3)
    plot_fft(Ts, e, 130e3, 190e3, 'r')
    plt.title(f'Sinais transmitido (modulado) no domínio da frequência. Potência = {Ptotal} watts')

    plt.show()

def QAM8():
    bin = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    f = 4
    L = len(bin)
    k = 50
    bit1 = np.ones(k)
    bit0 = np.zeros(k)
    symbol = np.ones(3 * k)
    mbit = np.array([], dtype=int)
    mx = np.array([], dtype=float)
    my = np.array([], dtype=float)

    if 3 * (L // 3) != L:
        raise ValueError('Size of data must be multiple of 3')

    for n in range(0, L, 3):
        if bin[n] == 0 and bin[n + 1] == 0 and bin[n + 2] == 0:
            x, y, bit = symbol, np.zeros(symbol.shape), np.concatenate([bit0, bit0, bit0])
        elif bin[n] == 0 and bin[n + 1] == 0 and bin[n + 2] == 1:
            x, y, bit = 2 * symbol, np.zeros(symbol.shape), np.concatenate([bit0, bit0, bit1])
        elif bin[n] == 0 and bin[n + 1] == 1 and bin[n + 2] == 0:
            x, y, bit = np.zeros(symbol.shape), symbol, np.concatenate([bit0, bit1, bit0])
        elif bin[n] == 0 and bin[n + 1] == 1 and bin[n + 2] == 1:
            x, y, bit = np.zeros(symbol.shape), 2 * symbol, np.concatenate([bit0, bit1, bit1])
        elif bin[n] == 1 and bin[n + 1] == 0 and bin[n + 2] == 0:
            x, y, bit = -1 * symbol, np.zeros(symbol.shape), np.concatenate([bit1, bit0, bit0])
        elif bin[n] == 1 and bin[n + 1] == 0 and bin[n + 2] == 1:
            x, y, bit = -2 * symbol, np.zeros(symbol.shape), np.concatenate([bit1, bit0, bit1])
        elif bin[n] == 1 and bin[n + 1] == 1 and bin[n + 2] == 0:
            x, y, bit = np.zeros(symbol.shape), -1 * symbol, np.concatenate([bit1, bit1, bit0])
        elif bin[n] == 1 and bin[n + 1] == 1 and bin[n + 2] == 1:
            x, y, bit = np.zeros(symbol.shape), -2 * symbol, np.concatenate([bit1, bit1, bit1])

        mbit = np.concatenate([mbit, bit])
        mx = np.concatenate([mx, x])
        my = np.concatenate([my, y])

    v = np.linspace(0, 2 * np.pi * L, len(mx))
    msync = mx + 1j * my
    qam = np.real(msync) * np.cos(f * v) + np.imag(msync) * np.sin(f * v)
    Vn = np.random.normal(scale=0.1, size=len(qam)) + qam
    Vnx, Vny = Vn * np.cos(f * v), Vn * np.sin(f * v)
    b, a = butter(2, 0.04)
    Hx, Hy = filtfilt(b, a, Vnx), filtfilt(b, a, Vny)
    M = len(Hx)
    mdeb = np.array([], dtype=int)

    for m in range(int(1.5 * k), M, 3 * k):
        if -0.25 < Hx[m] < 0.25:
            if 0.25 < Hy[m] < 0.75:
                deb = np.concatenate([bit0, bit1, bit0])
            elif 0.75 < Hy[m] < 1.25:
                deb = np.concatenate([bit0, bit1, bit1])
            elif -0.25 > Hy[m] > -0.75:
                deb = np.concatenate([bit1, bit1, bit0])
            elif -0.75 > Hy[m] > -1.25:
                deb = np.concatenate([bit1, bit1, bit1])
        elif -0.25 < Hy[m] < 0.25:
            if 0.25 < Hx[m] < 0.75:
                deb = np.concatenate([bit0, bit0, bit0])
            elif 0.75 < Hx[m] < 1.25:
                deb = np.concatenate([bit0, bit0, bit1])
            elif -0.25 > Hx[m] > -0.75:
                deb = np.concatenate([bit1, bit0, bit0])
            elif -0.75 > Hx[m] > -1.25:
                deb = np.concatenate([bit1, bit0, bit1])

        mdeb = np.concatenate([mdeb, deb])

    plt.figure(1)
    plt.subplot(4, 1, 1)
    plt.plot(mbit, 'r', linewidth=2)
    plt.axis([0, k * L, -0.5, 1.5])
    plt.grid(True)
    plt.legend(['Data in'])

    plt.subplot(4, 1, 2)
    plt.plot(qam, 'm', linewidth=1.5)
    plt.axis([0, k * L, -2.5, 2.5])
    plt.grid(True)
    plt.legend(['QAM mod '])

    plt.subplot(4, 1, 3)
    plt.plot(Vn, 'g', linewidth=1.5)
    plt.axis([0, k * L, -2.5, 2.5])
    plt.grid(True)
    plt.legend(['QAM mod & AWGN'])

    plt.subplot(4, 1, 4)
    plt.plot(mdeb, 'k', linewidth=1.5)
    plt.axis([0, k * L, -0.5, 1.5])
    plt.grid(True)
    plt.legend(['Data out'])
    plt.show()

def FSK():
    bit0 = np.zeros(50)
    bit1 = np.ones(50)
    bin = np.concatenate([bit0, bit0, bit1, bit1, bit0, bit1, bit1, bit0, bit0, bit0])
    L = len(bin)
    A = 3
    fc1 = 4
    fc2 = 2
    time = np.arange(0, L, 0.1)
    s = np.zeros(len(time))

    for i in range(L):
        if bin[i] == 1:
            s[i * 10:(i + 1) * 10] = A * np.sin(2 * np.pi * fc1 * time[i * 10:(i + 1) * 10])
        else:
            s[i * 10:(i + 1) * 10] = A * np.sin(2 * np.pi * fc2 * time[i * 10:(i + 1) * 10])

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(bin)
    plt.title('Binary signal')

    plt.subplot(2, 1, 2)
    plt.plot(time, s)
    plt.title('FSK signal')

    plt.show()

def FM_Modu():
    fs = 8000
    fm = 2
    fc = 100
    Am = 1
    Ac = 2
    beta = 0.75
    t = np.linspace(0, 1, fs)

    m_t = Am * np.cos(2 * np.pi * fm * t)
    y_t = Ac * np.cos(2 * np.pi * fc * t + beta * np.sin(2 * np.pi * fm * t))

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t, m_t)
    plt.title('Sinal Modulante')

    plt.subplot(2, 1, 2)
    plt.plot(t, y_t)
    plt.title('Sinal FM Modulado')

    plt.show()

def BPSK():
    N = 10
    f = 100
    t = np.linspace(0, 1, 1000)
    carrier = np.cos(2 * np.pi * f * t)
    data = np.random.randint(0, 2, N)
    bpsk_signal = np.repeat(2 * data - 1, 100)

    t_extended = np.linspace(0, N, len(bpsk_signal))
    modulated_signal = carrier[:len(bpsk_signal)] * bpsk_signal

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.stem(data)
    plt.title('Data')

    plt.subplot(3, 1, 2)
    plt.plot(t_extended, bpsk_signal)
    plt.title('BPSK Signal')

    plt.subplot(3, 1, 3)
    plt.plot(t_extended, modulated_signal)
    plt.title('BPSK Modulated Signal')

    plt.show()

def ASK():
    t = np.linspace(0, 1, 1000)
    carrier = np.cos(2 * np.pi * 100 * t)
    data = np.random.randint(0, 2, 10)
    ask_signal = np.repeat(data, 100)

    t_extended = np.linspace(0, 10, len(ask_signal))
    modulated_signal = carrier[:len(ask_signal)] * ask_signal

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.stem(data)
    plt.title('Data')

    plt.subplot(3, 1, 2)
    plt.plot(t_extended, ask_signal)
    plt.title('ASK Signal')

    plt.subplot(3, 1, 3)
    plt.plot(t_extended, modulated_signal)
    plt.title('ASK Modulated Signal')

    plt.show()

def close_app():
    root.destroy()

root = tk.Tk()
root.title("Modulação de Sinais - SISTEMAS DE COMUNICAÇÃO ")
root.geometry("500x400")  # Ajusta o tamanho da janela (largura x altura)

# Adiciona um rótulo com a versão
version_label = tk.Label(root, text="Versão: 1.0", font=("Arial", 12))
version_label.pack(pady=10)

tk.Button(root, text="AMDSB", command=AMDSB).pack(pady=10)
tk.Button(root, text="QAM8", command=QAM8).pack(pady=10)
tk.Button(root, text="FSK", command=FSK).pack(pady=10)
tk.Button(root, text="FM Modulation", command=FM_Modu).pack(pady=10)
tk.Button(root, text="BPSK", command=BPSK).pack(pady=10)
tk.Button(root, text="ASK", command=ASK).pack(pady=10)
tk.Button(root, text="Fechar", command=close_app).pack(pady=10)

root.mainloop()
