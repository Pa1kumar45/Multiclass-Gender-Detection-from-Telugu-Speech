# Gender Detection from Audio - Detailed Project Explanation

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Code Walkthrough - Cell by Cell](#code-walkthrough)
4. [Feature Extraction Techniques](#feature-extraction-techniques)
5. [Data Processing Pipeline](#data-processing-pipeline)
6. [Output Format](#output-format)
7. [Mathematical Concepts](#mathematical-concepts)

---

## 🎯 Project Overview

### Purpose

This project implements a **Gender Detection System** using audio features extracted from voice recordings. The system processes WAV audio files from male and female speakers, extracts acoustic features (pitch and MFCCs), and saves them in a structured Excel file for machine learning analysis.

### Workflow

```
Audio Files (.wav) → Feature Extraction → DataFrame → Excel Export
   (Male/Female)      (Pitch + MFCCs)    (Pandas)   (.xlsx file)
```

### Key Objectives

- Extract **pitch** (fundamental frequency) from voice recordings
- Extract **MFCCs** (Mel-Frequency Cepstral Coefficients) for voice characteristics
- Create a labeled dataset for gender classification
- Save features in Excel format for further analysis/modeling

---

## 🛠️ Technology Stack

| Library     | Purpose                                     | Version Used |
| ----------- | ------------------------------------------- | ------------ |
| `pandas`    | Data manipulation and Excel export          | 2.3.3        |
| `numpy`     | Numerical computations and array operations | 2.3.3        |
| `librosa`   | Audio processing and feature extraction     | 0.11.0       |
| `openpyxl`  | Excel file writing (pandas dependency)      | 3.1.5        |
| `soundfile` | Audio file I/O (librosa dependency)         | 0.13.1       |
| `scipy`     | Scientific computing (librosa dependency)   | 1.16.2       |

---

## 📝 Code Walkthrough - Cell by Cell

### **Cell 1: Import Dependencies**

```python
import os
```

**Line-by-line explanation:**

- **Purpose**: Import the operating system interface module
- **Usage in project**: Used for file path operations (`os.path.join`, `os.listdir`)
- **Why needed**: Navigate folder structures and list audio files

```python
import pandas as pd
```

- **Purpose**: Import pandas library for data manipulation
- **Alias**: `pd` is the standard convention
- **Usage**: Create DataFrames to organize features, then export to Excel
- **Key methods used**: `pd.DataFrame()`, `.to_excel()`

```python
import numpy as np
```

- **Purpose**: Import NumPy for numerical computing
- **Alias**: `np` is the standard convention
- **Usage**: Array operations, mathematical functions, padding, NaN handling
- **Key methods used**: `np.median()`, `np.isnan()`, `np.pad()`

```python
import librosa
```

- **Purpose**: Import librosa - the core audio analysis library
- **Usage**: Load audio files, extract pitch (YIN algorithm), compute MFCCs
- **Key methods used**:
  - `librosa.load()` - load audio files
  - `librosa.yin()` - pitch detection
  - `librosa.feature.mfcc()` - extract MFCCs
  - `librosa.to_mono()` - convert stereo to mono

```python
from google.colab import drive
from google.colab import files
```

- **Purpose**: Google Colab-specific imports for cloud environment
- **`drive`**: Mount Google Drive to access stored audio files
- **`files`**: Download generated Excel file to local machine
- **Note**: These will cause errors in local VS Code environment (should be removed/commented for local execution)

```python
import warnings
warnings.filterwarnings("ignore")
```

- **Purpose**: Suppress warning messages during execution
- **Why needed**: Librosa and audio processing often generate deprecation warnings that clutter output
- **Effect**: Only errors will be shown, not warnings

---

### **Cell 2: Mount Google Drive**

```python
drive.mount("/content/drive", force_remount=True)
```

**Line-by-line explanation:**

- **`drive.mount()`**: Colab function to mount Google Drive filesystem
- **`"/content/drive"`**: Mount point - where Drive appears in Colab filesystem
- **`force_remount=True`**:
  - Unmount if already mounted
  - Ensures fresh connection
  - Useful when rerunning notebook
- **Output**: Prompts for Google account authentication
- **Result**: Drive files accessible at `/content/drive/MyDrive/...`

**⚠️ Local Execution Note**: This cell must be modified for local execution. Replace with:

```python
# Local paths instead of Colab mount
female_folder = r"C:\path\to\your\female_audio"
male_folder = r"C:\path\to\your\male_audio"
```

---

### **Cell 3: Define Data Folders**

```python
female_folder = "/content/drive/MyDrive/te_in_female"
```

**Explanation:**

- **Variable**: `female_folder` stores the path to female voice recordings
- **Path structure**:
  - `/content/drive/MyDrive/` - Google Drive root in Colab
  - `te_in_female` - folder containing female WAV files
- **Naming**: "te_in" likely refers to "Telugu Input" or training/test input data
- **Expected contents**: `.wav` files of female voices

```python
male_folder = "/content/drive/MyDrive/te_in_male"
```

**Explanation:**

- **Variable**: `male_folder` stores the path to male voice recordings
- **Path structure**: Same as female folder but for male samples
- **Expected contents**: `.wav` files of male voices

**Data Organization:**

```
MyDrive/
├── te_in_female/
│   ├── female_001.wav
│   ├── female_002.wav
│   └── ...
└── te_in_male/
    ├── male_001.wav
    ├── male_002.wav
    └── ...
```

---

### **Cell 4: Feature Extraction Functions & Processing Loop**

This is the **core cell** containing feature extraction logic and audio processing.

#### **Function 1: `get_pitch()` - Pitch Detection**

```python
def get_pitch(y, sr=16000, fmin=75.0, fmax=400.0):
```

**Function signature:**

- **`y`**: Audio time series (numpy array of audio samples)
- **`sr=16000`**: Sample rate in Hz (default 16kHz)
  - Why 16kHz? Standard for speech processing (good quality, manageable size)
- **`fmin=75.0`**: Minimum expected pitch frequency (Hz)
  - Covers low male voices (~75 Hz is very low bass)
- **`fmax=400.0`**: Maximum expected pitch frequency (Hz)
  - Covers high female voices (~400 Hz is soprano range)
- **Returns**: Median pitch in Hz (float), or 0.0 if extraction fails

```python
    """
    Estimate pitch (Hz) using librosa.yin.
    Returns median pitch of voiced frames; returns 0 if no voiced frames found.
    """
```

- **Docstring**: Explains function purpose and behavior
- **"Voiced frames"**: Parts of audio with vocal cord vibration (not silence/noise)
- **YIN algorithm**: Time-domain pitch detection algorithm (accurate for speech)

```python
    try:
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256)
```

**Line breakdown:**

- **`try` block**: Error handling for pitch extraction
- **`librosa.yin()`**: YIN pitch detection algorithm
  - **`y`**: Input audio signal
  - **`fmin`, `fmax`**: Pitch range constraints
  - **`sr`**: Sample rate
  - **`frame_length=2048`**: Window size for analysis (2048 samples ≈ 128ms at 16kHz)
  - **`hop_length=256`**: Step size between windows (256 samples ≈ 16ms)
- **`f0`**: Output array of fundamental frequencies (one per frame)
  - Contains `NaN` for unvoiced frames (silence, noise)
  - Contains Hz values for voiced frames

```python
        voiced = f0[~np.isnan(f0)]
```

**Filter voiced frames:**

- **`np.isnan(f0)`**: Boolean array marking NaN values
- **`~`**: NOT operator (inverts boolean array)
- **`f0[~np.isnan(f0)]`**: Fancy indexing - keeps only non-NaN values
- **Result**: `voiced` contains only valid pitch values

```python
        return float(np.median(voiced)) if len(voiced) > 0 else 0.0
```

**Return median pitch:**

- **`len(voiced) > 0`**: Check if any voiced frames exist
- **`np.median(voiced)`**: Compute median pitch
  - **Why median?** Robust to outliers (better than mean for pitch)
  - Typical values: ~120 Hz (male), ~220 Hz (female)
- **`float()`**: Convert numpy type to Python float
- **`else 0.0`**: Return 0 if no voiced frames (e.g., silent file)

```python
    except:
        return 0.0
```

**Error handling:**

- Catches any exception during pitch extraction
- Returns 0.0 as fallback value
- **Why broad except?** Librosa can fail on corrupted/unusual audio formats

---

#### **Function 2: `get_mfcc()` - MFCC Feature Extraction**

```python
def get_mfcc(y, sr=16000, n_mfcc=10, fixed_length=40000):
```

**Function signature:**

- **`y`**: Audio time series
- **`sr=16000`**: Sample rate (16kHz)
- **`n_mfcc=10`**: Number of MFCC coefficients to extract
  - Typical range: 13-40 for speech
  - 10 is on the lower end (capturing main spectral envelope)
- **`fixed_length=40000`**: Standardize audio length (samples)
  - 40000 samples at 16kHz = 2.5 seconds
- **Returns**: Flattened MFCC array of exactly 110 elements

```python
    """
    Extracts MFCCs and returns a flattened vector of fixed length (110 elements).
    """
```

- Guarantees consistent output size for machine learning

```python
    y = librosa.to_mono(y)
```

**Convert to mono:**

- **Purpose**: Ensure single-channel audio
- **Why?** Some files may be stereo; MFCC extraction expects mono
- **Method**: Averages left/right channels if stereo
- **Effect**: `y` becomes 1D array

```python
    y = y[:fixed_length]
```

**Truncate to fixed length:**

- **Slicing**: Keep only first 40000 samples
- **Effect**: Long audio files are cut to 2.5 seconds
- **Why?** Standardize input size for consistent feature dimensions

```python
    if len(y) < fixed_length:
        y = np.pad(y, (0, fixed_length - len(y)), 'constant')
```

**Pad short audio:**

- **Condition**: Audio shorter than 2.5 seconds
- **`np.pad()`**: Add zeros to end of array
  - **`(0, fixed_length - len(y))`**: Pad 0 samples at start, `N` at end
  - **`'constant'`**: Pad with constant value (default 0)
- **Result**: All audio is exactly 40000 samples

```python
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=4000)
```

**Extract MFCCs:**

- **`librosa.feature.mfcc()`**: Mel-Frequency Cepstral Coefficients extraction
  - **`y`**: Input audio (now exactly 40000 samples)
  - **`sr`**: Sample rate
  - **`n_mfcc=10`**: Extract 10 coefficients per frame
  - **`hop_length=4000`**: Large hop = fewer frames
    - 40000 samples / 4000 hop = 10 frames
- **Output shape**: `(10, 10)` - 10 MFCCs × 10 time frames = 100 values
- **What are MFCCs?** Compact representation of spectral envelope (vocal tract shape)

```python
    mfcc_flat = mfcc.flatten()
```

**Flatten to 1D:**

- **`flatten()`**: Convert 2D array to 1D
- **Shape**: `(10, 10)` → `(100,)`
- **Order**: Row-major (frame 1 all coeffs, frame 2 all coeffs, ...)

```python
    desired_len = 110
```

**Target length:**

- Fixed output size for dataset consistency
- 110 features per audio file

```python
    if len(mfcc_flat) < desired_len:
        mfcc_flat = np.pad(mfcc_flat, (0, desired_len - len(mfcc_flat)), 'constant')
```

**Pad to 110:**

- If MFCCs < 110 (shouldn't happen with settings above), add zeros
- Ensures exactly 110 elements

```python
    else:
        mfcc_flat = mfcc_flat[:desired_len]
```

**Truncate to 110:**

- If MFCCs > 110, keep only first 110
- Safety measure for edge cases

```python
    return mfcc_flat
```

**Return:**

- 1D numpy array of exactly 110 MFCC values

---

#### **Processing Loop Initialization**

```python
features = []
```

**Feature storage:**

- Empty list to accumulate all extracted features
- Each element will be: `[filepath, pitch, mfcc1, mfcc2, ..., mfcc110, gender]`
- Final size: `N` audio files × 113 values per file

```python
counter = 0
```

**Progress counter:**

- Tracks total files processed
- Used for console output to monitor progress

---

#### **Female Audio Processing Loop**

```python
for audio_file in os.listdir(female_folder):
```

**Iterate through female folder:**

- **`os.listdir(female_folder)`**: Get list of all files/folders in path
- **`audio_file`**: Variable holding each filename (e.g., "voice001.wav")

```python
    if audio_file.endswith('.wav'):
```

**Filter WAV files:**

- **`endswith('.wav')`**: Check if filename ends with .wav extension
- **Why?** Ignore non-audio files (.txt, .DS_Store, etc.)
- Only processes actual audio files

```python
        file_path = os.path.join(female_folder, audio_file)
```

**Build full path:**

- **`os.path.join()`**: Safely combine folder and filename
  - Handles OS-specific path separators (`/` vs `\`)
- **Result**: `/content/drive/MyDrive/te_in_female/voice001.wav`

```python
        try:
            y, sr = librosa.load(file_path, sr=16000, mono=True)
```

**Load audio file:**

- **`librosa.load()`**: Read audio file into memory
  - **`file_path`**: Full path to WAV file
  - **`sr=16000`**: Resample to 16kHz (if original is different)
  - **`mono=True`**: Force mono output
- **Returns**:
  - **`y`**: Audio time series (numpy array of samples)
  - **`sr`**: Actual sample rate (should be 16000)

```python
            pitch = get_pitch(y, sr)
```

**Extract pitch:**

- Call custom `get_pitch()` function
- **Input**: Audio signal and sample rate
- **Output**: Single float (median pitch in Hz)

```python
            mfcc_features = get_mfcc(y, sr)
```

**Extract MFCCs:**

- Call custom `get_mfcc()` function
- **Input**: Audio signal and sample rate
- **Output**: 1D array of 110 MFCC values

```python
            features.append([file_path, pitch] + mfcc_features.tolist() + ['female'])
```

**Build feature row:**

- **`[file_path, pitch]`**: Start with filepath and pitch (2 elements)
- **`mfcc_features.tolist()`**: Convert numpy array to Python list (110 elements)
- **`+ ['female']`**: Add gender label (1 element)
- **Total**: 1 + 1 + 110 + 1 = **113 columns**
- **Structure**: `[filepath, pitch, mfcc1...mfcc110, gender]`
- **`features.append()`**: Add this row to dataset list

```python
            counter += 1
            print(f"Processed {counter} female files")
```

**Progress tracking:**

- **`counter += 1`**: Increment file count
- **`print()`**: Display progress message
  - **f-string**: Format string with variable
  - Shows: "Processed 1 female files", "Processed 2 female files", ...

```python
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
```

**Error handling:**

- **`except Exception as e`**: Catch any error during processing
- **`e`**: Exception object with error details
- **`print()`**: Display which file failed and why
- **Effect**: Continue processing other files even if one fails

---

#### **Male Audio Processing Loop**

```python
for audio_file in os.listdir(male_folder):
    if audio_file.endswith('.wav'):
        file_path = os.path.join(male_folder, audio_file)
        try:
            y, sr = librosa.load(file_path, sr=16000, mono=True)

            pitch = get_pitch(y, sr)
            mfcc_features = get_mfcc(y, sr)

            features.append([file_path, pitch] + mfcc_features.tolist() + ['male'])
            counter += 1
            print(f"Processed {counter} total files")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
```

**Identical logic to female loop, with differences:**

1. **`male_folder`**: Processes different directory
2. **`['male']`**: Label is 'male' instead of 'female'
3. **Print message**: Shows "total files" to track combined count

---

### **Cell 5: Save Features to Excel**

```python
if len(features) > 0:
    print(f"\nTotal features extracted: {len(features)}")
else:
    print("\nNo features extracted!")
```

**Data validation:**

- **`len(features)`**: Count rows in feature list
- **If positive**: Show count (e.g., "Total features extracted: 500")
- **If zero**: Warn user (no data was extracted)
- **`\n`**: Newline for cleaner output

```python
columns = ['audio_file', 'pitch'] + [f'mfcc{i}' for i in range(1, 111)] + ['gender']
```

**Define column names:**

- **`['audio_file', 'pitch']`**: First 2 columns
- **List comprehension**: `[f'mfcc{i}' for i in range(1, 111)]`
  - **`range(1, 111)`**: Numbers 1 to 110
  - **f-string**: Creates 'mfcc1', 'mfcc2', ..., 'mfcc110'
  - **Result**: 110 MFCC column names
- **`+ ['gender']`**: Last column
- **Total**: 1 + 1 + 110 + 1 = **113 columns**

```python
feature_df = pd.DataFrame(features, columns=columns)
```

**Create DataFrame:**

- **`pd.DataFrame()`**: Convert list to pandas DataFrame
  - **`features`**: List of lists (rows)
  - **`columns`**: Column name labels
- **Result**: Structured table with named columns
- **Shape**: `(N rows, 113 columns)` where N = total audio files

```python
print("\nSample feature rows:")
print(feature_df.head())
```

**Display preview:**

- **`feature_df.head()`**: Get first 5 rows of DataFrame
- **`print()`**: Show preview to user
- **Purpose**: Verify data looks correct before saving

```python
save_path = '/content/PAS_Features(new).xlsx'
```

**Define output path:**

- **`/content/`**: Colab temporary storage
- **`PAS_Features(new).xlsx`**: Filename
  - "PAS" likely stands for "Probability & Statistics" or project acronym
  - ".xlsx" is Excel format

```python
feature_df.to_excel(save_path, index=False)
```

**Export to Excel:**

- **`.to_excel()`**: Pandas method to write Excel file
  - **`save_path`**: Where to save file
  - **`index=False`**: Don't include row numbers as a column
- **Requires**: `openpyxl` library (installed as dependency)
- **Output**: Excel file with 113 columns and N rows

---

### **Cell 6: Download File (Colab)**

```python
files.download(save_path)
```

**Download to local machine:**

- **`files.download()`**: Colab function to download files
- **`save_path`**: Path to Excel file
- **Effect**: Browser prompts to download file
- **⚠️ Note**: Only works in Google Colab, not local environment

```python
print("\nFeatures saved as 'PAS_Features(new).xlsx'")
```

**Confirmation message:**

- Notify user of successful save
- Shows filename

---

## 🎵 Feature Extraction Techniques

### 1. **Pitch (Fundamental Frequency - F0)**

#### What is Pitch?

- **Definition**: The perceived frequency of a sound's fundamental vibration
- **Physical basis**: Vocal cord vibration rate
- **Unit**: Hertz (Hz) - cycles per second

#### Gender Differences

| Gender   | Typical Pitch Range | Average |
| -------- | ------------------- | ------- |
| Male     | 85-180 Hz           | ~120 Hz |
| Female   | 165-255 Hz          | ~220 Hz |
| Children | 250-300 Hz          | ~275 Hz |

#### YIN Algorithm

- **Type**: Time-domain pitch detection
- **Advantages**:
  - More accurate than autocorrelation
  - Robust to noise
  - Good for speech/voice
- **How it works**:
  1. Computes difference function (squared difference of signal with delayed copy)
  2. Applies cumulative mean normalization
  3. Finds absolute minimum in search range
  4. Uses parabolic interpolation for sub-sample accuracy

#### Implementation Details

```python
f0 = librosa.yin(y, fmin=75.0, fmax=400.0, sr=16000,
                 frame_length=2048, hop_length=256)
```

- **Frame analysis**: 2048 samples = 128ms windows
- **Hop size**: 256 samples = 16ms steps (87.5% overlap)
- **Search range**: 75-400 Hz covers all typical voices
- **Output**: Array of F0 values per frame (NaN for unvoiced)

---

### 2. **MFCCs (Mel-Frequency Cepstral Coefficients)**

#### What are MFCCs?

- **Definition**: Compact representation of the spectral envelope of sound
- **Purpose**: Capture vocal tract shape (determines vowels, voice timbre)
- **Inspiration**: Human auditory perception (Mel scale)

#### Mathematical Pipeline

```
Audio Signal → MFCC Extraction
     ↓
1. Pre-emphasis (high-pass filter)
     ↓
2. Frame blocking (40000 samples → chunks)
     ↓
3. Windowing (Hamming window)
     ↓
4. FFT (Fast Fourier Transform)
     ↓
5. Mel Filter Bank (40 triangular filters)
     ↓
6. Log compression
     ↓
7. DCT (Discrete Cosine Transform)
     ↓
8. Keep first 10 coefficients
```

#### Mel Scale

- **Formula**: `Mel(f) = 2595 * log10(1 + f/700)`
- **Purpose**: Perceptual scale (human ears are logarithmic)
- **Effect**: More resolution at low frequencies (< 1kHz)

#### Why MFCCs for Gender Detection?

1. **Vocal tract length**: Males have longer vocal tracts
   - Affects formant frequencies (resonances)
   - Captured by MFCC spectral envelope
2. **Voice quality**: Breathiness, roughness differ by gender
3. **Spectral tilt**: Males have more energy in low frequencies
4. **Compact**: 10 coefficients capture essential info

#### Implementation Details

```python
mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=10, hop_length=4000)
```

- **10 MFCCs**: Lower coefficients (0-9)
  - MFCC 0: Energy (often excluded in some systems)
  - MFCC 1-9: Spectral shape
- **Hop length 4000**: Large step for global features (not time-sensitive)
  - 40000 / 4000 = 10 frames
- **Output shape**: (10 MFCCs, 10 frames) = 100 values
- **Padding to 110**: Ensures fixed size (safety margin)

---

## 🔄 Data Processing Pipeline

### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Data Collection                                     │
│ - Audio files stored in Google Drive                        │
│ - Organized by gender: te_in_female/, te_in_male/          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ Step 2: File Loading                                         │
│ - List all .wav files in each folder                        │
│ - Load audio with librosa.load()                            │
│ - Resample to 16kHz, convert to mono                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ Step 3: Audio Preprocessing                                  │
│ - Truncate/pad to 40000 samples (2.5 seconds)              │
│ - Ensures consistent input dimensions                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
┌─────────▼──────────┐  ┌────────▼──────────┐
│ Step 4a: Pitch     │  │ Step 4b: MFCCs    │
│ - YIN algorithm    │  │ - FFT + Mel       │
│ - Get median F0    │  │ - DCT → 10 coeffs │
│ - Output: 1 value  │  │ - 10 frames       │
│                    │  │ - Output: 110 vals│
└─────────┬──────────┘  └────────┬──────────┘
          │                      │
          └───────────┬──────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ Step 5: Feature Concatenation                                │
│ - Combine: [filepath, pitch, mfcc1..mfcc110, gender]       │
│ - Total: 113 features per audio file                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ Step 6: DataFrame Construction                               │
│ - Convert list of features to pandas DataFrame              │
│ - Assign column names                                        │
│ - Shape: (N_files, 113)                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ Step 7: Export to Excel                                      │
│ - Save as .xlsx file                                         │
│ - Download to local machine (Colab)                         │
│ - Ready for ML model training                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Output Format

### Excel File Structure

**Filename**: `PAS_Features(new).xlsx`

**Dimensions**: N rows × 113 columns

**Column Schema**:

| Column Index | Column Name     | Data Type | Description             | Example Value                       |
| ------------ | --------------- | --------- | ----------------------- | ----------------------------------- |
| 0            | audio_file      | string    | Full path to audio file | `/content/drive/.../female_001.wav` |
| 1            | pitch           | float     | Median F0 in Hz         | 215.3                               |
| 2-111        | mfcc1...mfcc110 | float     | MFCC coefficients       | -145.2, 23.4, ...                   |
| 112          | gender          | string    | Gender label            | 'female' or 'male'                  |

### Sample Data Row

```python
[
  '/content/drive/MyDrive/te_in_female/voice001.wav',  # audio_file
  218.5,                                                # pitch
  -142.3, 25.1, 12.8, ..., 0.3,                        # mfcc1...mfcc110
  'female'                                              # gender
]
```

### Statistical Properties

**Pitch Column**:

- Male voices: ~100-150 Hz
- Female voices: ~180-250 Hz
- Missing/silent files: 0.0

**MFCC Columns**:

- Range: Typically -500 to +500
- MFCC1 (largest): Often negative, large magnitude
- MFCC2-10: Decreasing magnitude
- Later MFCCs: Capture fine spectral details

**Gender Column**:

- Categorical: 'male' or 'female'
- Can be encoded: 0=male, 1=female for ML models

---

## 🧮 Mathematical Concepts

### 1. **Audio Signal Representation**

#### Time Domain

```
Audio signal: y(t) where t is time
Discrete: y[n] where n is sample index
Sample rate: sr = 16000 Hz
Duration: T = len(y) / sr seconds
```

#### Example

- 40000 samples at 16kHz = 2.5 seconds
- Each sample represents amplitude at moment in time

---

### 2. **Frequency Domain (Fourier Analysis)**

#### Fourier Transform Concept

```
Any signal can be decomposed into sum of sine waves:
y(t) = A₁·sin(2πf₁t) + A₂·sin(2πf₂t) + ...
```

#### FFT (Fast Fourier Transform)

- Converts time-domain signal to frequency-domain
- Shows which frequencies are present and their amplitudes
- Used in MFCC computation

---

### 3. **Pitch Detection Mathematics**

#### Autocorrelation Method (simplified)

```
R(τ) = Σ y[n] · y[n+τ]
```

- Find lag `τ` where signal matches itself
- Period T corresponds to 1/f₀

#### YIN Improvement

```
d(τ) = Σ (y[n] - y[n+τ])²  # Difference function
d'(τ) = d(τ) / [(1/τ)Σd(j)]  # Normalized
```

- More robust than autocorrelation
- Better handling of subharmonics

---

### 4. **Mel Scale**

#### Formula

```
m = 2595 · log₁₀(1 + f/700)
```

#### Inverse

```
f = 700 · (10^(m/2595) - 1)
```

#### Example Conversions

| Frequency (Hz) | Mel  |
| -------------- | ---- |
| 100            | 150  |
| 500            | 550  |
| 1000           | 1000 |
| 2000           | 1550 |
| 4000           | 2300 |

**Observation**: Linear below 1kHz, logarithmic above

---

### 5. **Cepstral Analysis**

#### Cepstrum Definition

```
Cepstrum = IFFT(log(|FFT(signal)|))
```

- "Spectrum of a spectrum"
- Separates source (vocal cords) from filter (vocal tract)

#### MFCC Computation

```
1. X(f) = FFT(windowed_audio)
2. S(m) = Mel_filterbank(|X(f)|²)
3. L(m) = log(S(m))
4. C(n) = DCT(L(m))
5. MFCC = C(1:10)  # Keep first 10 coeffs
```

---

### 6. **Padding & Truncation Math**

#### Fixed-Length Standardization

```python
target_length = 40000

if len(y) < target_length:
    y_new = [y[0], y[1], ..., y[len-1], 0, 0, ..., 0]
    # Pad with (40000 - len(y)) zeros

elif len(y) > target_length:
    y_new = [y[0], y[1], ..., y[39999]]
    # Keep only first 40000 samples
```

#### Why Necessary?

- Machine learning models require fixed input dimensions
- Ensures MFCC matrix is always same shape
- Simplifies batch processing

---

## 🎯 Use Cases & Applications

### Immediate Applications

1. **Gender Classification Model**:

   - Use extracted features to train ML classifier
   - Algorithms: SVM, Random Forest, Neural Networks
   - Accuracy expectation: 95-98%

2. **Voice Authentication**:

   - Verify speaker identity
   - Combined with gender as prior information

3. **Speech Recognition Enhancement**:
   - Gender-specific acoustic models
   - Improved accuracy for ASR systems

### Extended Research

1. **Age Estimation**: Pitch correlates with age
2. **Emotion Detection**: MFCCs capture voice quality
3. **Accent Classification**: Spectral patterns differ by accent
4. **Speaker Verification**: Voice biometrics

---

## 📈 Performance Considerations

### Computational Complexity

| Operation      | Time Complexity | Note          |
| -------------- | --------------- | ------------- |
| File loading   | O(n)            | n = file size |
| Pitch (YIN)    | O(n log n)      | Per frame     |
| MFCC           | O(n log n)      | FFT dominates |
| Total per file | ~1-3 seconds    | On modern CPU |

### Optimization Tips

1. **Batch Processing**: Process files in parallel
2. **Caching**: Save extracted features, don't recompute
3. **GPU Acceleration**: Not applicable (librosa is CPU-based)
4. **Sampling**: Use lower sample rate for faster processing (trade-off: quality)

---

## 🛠️ Potential Improvements

### Feature Engineering

1. **Add Delta MFCCs**: Capture temporal dynamics
   ```python
   delta_mfcc = librosa.feature.delta(mfcc)
   ```
2. **Pitch Statistics**: Min, max, std (not just median)
3. **Spectral Features**:
   - Spectral centroid (brightness)
   - Spectral rolloff
   - Zero-crossing rate
4. **Formant Frequencies**: F1, F2, F3 (vowel characteristics)

### Code Quality

1. **Error Logging**: Write errors to file, not just print
2. **Progress Bar**: Use `tqdm` for better progress tracking
   ```python
   from tqdm import tqdm
   for audio_file in tqdm(os.listdir(folder)):
   ```
3. **Configuration File**: External config for paths, parameters
4. **Unit Tests**: Validate feature extraction functions

### Data Augmentation

1. **Add noise**: Improve robustness
2. **Pitch shifting**: Simulate different speakers
3. **Time stretching**: Vary speech rate

---

## 🧪 Testing & Validation

### Recommended Tests

1. **Feature Distribution**:

   ```python
   feature_df.describe()  # Statistical summary
   feature_df.hist(figsize=(20,15))  # Histograms
   ```

2. **Gender Separation**:

   ```python
   female_pitch = feature_df[feature_df['gender']=='female']['pitch']
   male_pitch = feature_df[feature_df['gender']=='male']['pitch']
   print(f"Female avg: {female_pitch.mean():.1f} Hz")
   print(f"Male avg: {male_pitch.mean():.1f} Hz")
   ```

3. **Missing Values**:

   ```python
   feature_df.isnull().sum()  # Count NaNs per column
   ```

4. **Pitch=0 Analysis**:
   ```python
   zero_pitch = feature_df[feature_df['pitch'] == 0]
   print(f"{len(zero_pitch)} files with failed pitch extraction")
   ```

---

## 📚 References & Resources

### Libraries Documentation

- **Librosa**: https://librosa.org/doc/latest/
- **Pandas**: https://pandas.pydata.org/docs/
- **NumPy**: https://numpy.org/doc/

### Academic Papers

1. **YIN Algorithm**: De Cheveigné, A., & Kawahara, H. (2002). "YIN, a fundamental frequency estimator for speech and music."
2. **MFCCs**: Davis, S., & Mermelstein, P. (1980). "Comparison of parametric representations for monosyllabic word recognition."

### Tutorials

- Librosa Audio Feature Extraction: https://librosa.org/doc/latest/tutorial.html
- Speech Processing: https://www.coursera.org/learn/audio-signal-processing

---

## 🎓 Learning Outcomes

After understanding this project, you should know:

✅ How to load and preprocess audio files  
✅ Pitch detection using YIN algorithm  
✅ MFCC extraction and its mathematical foundation  
✅ Feature engineering for audio classification  
✅ Data organization for machine learning  
✅ Pandas DataFrame manipulation  
✅ Error handling in data processing pipelines

---

##  Next Steps

1. **Train ML Model**:
   - Load Excel file: `pd.read_excel('PAS_Features(new).xlsx')`
   - Split data: `train_test_split()`
   - Train classifier: `RandomForestClassifier()`, `SVC()`, etc.
   - Evaluate: Accuracy, F1-score, confusion matrix

2. **Visualization**:
   - Plot pitch distributions by gender
   - T-SNE visualization of MFCC space
   - Waveform and spectrogram plots

3. **Real-Time Detection**:
   - Microphone input
   - Online feature extraction
   - Live gender prediction

4. **Web Application**:
   - Upload audio file
   - Display predicted gender
   - Show confidence score

---

**Document Created**: For detailed explanation of Gender Detection from Audio project  
**Code Language**: Python 3.x  
**Environment**: Google Colab / Local Jupyter  
**Course**: IT 302 - Probability & Statistics Lab

---

*End of Part 1 Documentation*


---
---

# PART 2: Machine Learning Model Training & Evaluation

##  Overview of PASProject2.ipynb

This notebook is the **second phase** of the Gender Detection project. While PASProject1 extracted features from audio files, this notebook:
1. **Loads** the extracted features from Excel
2. **Trains** a Random Forest classifier  
3. **Evaluates** model performance
4. **Visualizes** feature importance
5. **Makes predictions** on the entire dataset
6. **Saves** comparison results

---

##  Complete Workflow

```
PASProject1.ipynb  PASProject2.ipynb
(Feature Extraction)  (Model Training)
                            
PAS_Features(new).xlsx  ML Model  Predictions
   (113 columns)        (RF Classifier)  (comparison_results)
```

---

##  Code Walkthrough - Cell by Cell

### **Cell 1: Import Machine Learning Libraries**

