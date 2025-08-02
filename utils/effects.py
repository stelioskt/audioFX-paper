import numpy as np
from pedalboard import Pedalboard, Reverb, Delay, Distortion, HighpassFilter, LowpassFilter, Chorus, Phaser


def apply_reverb(audio, sr, level):
    """Reverb: room size scales 0.1→0.9 over 1–10"""
    room_size = 0.1 + 0.08 * level
    board = Pedalboard([Reverb(room_size=room_size, wet_level=1)])
    return board(audio, sr)

def apply_delay(audio, sr, level):
    """Delay: time 0.05→0.32s, feedback 0.15→0.6"""
    delay_time = 0.05 + 0.027 * level
    feedback = 0.1 + 0.05 * level
    board = Pedalboard([Delay(delay_seconds=delay_time, feedback=feedback, mix=0.5)])
    return board(audio, sr)

def apply_distortion(audio, sr, level):
    """Distortion: drive 6.5→23dB"""
    drive_db = 5 + 1.8 * level
    board = Pedalboard([Distortion(drive_db=drive_db)])
    return board(audio, sr)

def apply_eq(audio, sr, level):
    """EQ: bandpass via highpass/lowpass"""
    low_cutoff = 100 + 400 * level       # 300→4100 Hz
    high_cutoff = sr/2 - 600 * level      # sr/2 -600→sr/2 -6000 Hz
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=low_cutoff),
        LowpassFilter(cutoff_frequency_hz=high_cutoff)
    ])
    processed = board(audio, sr)
    # normalize
    return processed / np.max(np.abs(processed))

def apply_chorus(audio, sr, level):
    """Chorus: rate 0.6→2.5Hz, depth 0.19→1.0, feedback 0.12→0.3"""
    rate_hz = 0.5 + 0.2 * level
    depth = 0.1 + 0.09 * level
    feedback = 0.1 + 0.02 * level
    board = Pedalboard([Chorus(rate_hz=rate_hz, depth=depth, feedback=feedback, mix=0.5)])
    return board(audio, sr)

def apply_phaser(audio, sr, level):
    """Phaser: rate 0.25→0.7Hz, depth 0.28→1.6, feedback 0.12→0.4"""
    rate_hz = 0.2 + 0.1 * level
    depth = 0.2 + 0.14 * level
    feedback = 0.1 + 0.03 * level
    board = Pedalboard([Phaser(rate_hz=rate_hz, depth=depth, feedback=feedback, mix=1)])
    return board(audio, sr)