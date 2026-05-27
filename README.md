# TapLeak: Recovering WhatsApp Conversations via LLM-Augmented Acoustic Keystroke Inference

Acoustic side-channel attack on virtual QWERTY keyboards using only the device's built-in microphone.
Raw audio → segmentation → acoustic features → k-NN classification → Trie filtering → GPT-2 reranking → recovered text.

## Repository Structure

```
├── Ashee Jain/
│   ├── ML_model.ipynb            # Feature extraction, MRMR, k-NN classifier
│   ├── Trie_Word_search.ipynb    # Trie-based word reconstruction
│   ├── Generate_sentences.ipynb  # GPT-2 perplexity reranking
│   └── Recordings/
│       └── Internal mic/
│           ├── Train/            # Per-character training recordings
│           ├── Test/             # Per-character test recordings
│           └── Bottom/           # Segmented tap audio + sentence recordings
├── figures/                      # Paper figures (confusion matrix, etc.)
├── ASCA_Keyboard.tex             # Paper source (LaTeX)
└── Confusion Matrix.png          # k-NN confusion matrix (26 classes)
```

## Dataset

- **Device:** iPhone 6 (internal bottom-facing microphone)
- **Recording app:** dB Meter (iOS), silent background recording
- **Characters:** a–z (26 classes)
- **Protocol:** 52 recordings — 1 training session + 1 test session per character, recorded on different days
- **Format:** M4A → converted to mono WAV at 44.1 kHz

## Attack Pipeline

| Stage | Method | Key Parameter |
|-------|--------|---------------|
| Bandpass filter | 4th-order Butterworth | 200–500 Hz |
| Peak detection | `scipy.signal.find_peaks` | height = max/2, distance = 550 samples |
| Feature extraction | 14 acoustic features | Pre-emphasis α = 0.97 |
| Feature selection | MRMR | Top-3: Spectral Kurtosis, Pitch, Short-Time Energy |
| Classification | k-NN | k = 10, top-5 candidates per tap |
| Word reconstruction | Trie (Google-10000-English) | 98% search-space compression |
| Sentence reranking | GPT-2 perplexity | Threshold PPL < 100 |

## Requirements

```
scipy
librosa
scikit-learn
numpy
transformers
torch
```

See individual notebooks for full dependency details. GPU recommended for GPT-2 inference (tested on Google Colab T4).

## Results

- **Top-5 character accuracy:** 100% (correct character always in top-5 candidates)
- **Sentence recovery:** correct sentence in top-3 ranked hypotheses; top-1 in controlled trials
- **Trie compression:** 625 → 7 candidates for a 4-character word (98.9% reduction)
