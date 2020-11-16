#!/usr/bin/env python3
"""
Utils file containing functions mapping text and audio to features.
"""

from typing import List

import numpy as np
import spacy

from surfboard.sound import Waveform

nlp = spacy.load('en_core_web_sm')


def text_to_features(text: str) -> np.ndarray:
    """Uses spacy to extract word vectors for every word in the input
    text. Averages those word vectors.
    Args:
        text (str): A string input sentence or document.
    Returns:
        np.ndarray: The averaged word vectors for every word in the
            sentence.
    """
    split_text: List[str] = text.split()
    word_vectors: List[np.ndarray] = [nlp(word).vector for word in split_text]
    return np.mean(word_vectors, 0)

def audio_to_features(audio: np.ndarray, sample_rate: int=44100):
    """Uses Surfboard to extract 13 averaged MFCCs over time.
    First load the waveform, then extract features.
    Args:
        audio (np.ndarray): The 1D waveform.
        sample_rate (int): The sample rate of the waveform.
    Returns:
        np.ndarray: The extracted audio features. 
    """
    waveform: Waveform = Waveform(signal=audio, sample_rate=sample_rate)
    averaged_mfccs: np.ndarray = waveform.mfcc().mean(0)
    return averaged_mfccs