# Depth Matching for Gamma Curves

A Streamlit-based application for **depth matching gamma and natural gamma curves from LAS files**.

This tool allows users to upload a single LAS file, select a **reference curve** and a **curve to shift**, and automatically determine the **optimal depth shift** using correlation-based matching.

It is designed for **borehole logging, televiewer, and geophysical well log QC workflows**.

---

## Features

- Upload a **single LAS file**
- Automatically detects likely gamma-type curves  
  (e.g. `NG`, `GAMMA`, `GR`)
- Select:
  - **Reference Curve**
  - **Curve to Shift**
- Adjustable matching parameters:
  - `resample_step`
  - `shift_min`
  - `shift_max`
  - `shift_step`
  - `smooth_sigma`
- Calculates:
  - **recommended shift**
  - **correlation coefficient**
- Visualises:
  - **before matching**
  - **after matching**
  - **correlation vs shift**
- Download correlation table as CSV

---

## Methodology

The app performs the following steps:

1. Reads the uploaded LAS file
2. Extracts the selected curves
3. Cleans null and invalid values
4. Resamples both curves to a common depth grid
5. Applies Gaussian smoothing
6. Computes correlation over a user-defined shift range
7. Finds the shift with the highest correlation
8. Displays the recommended shift

---

## How the Shift is Calculated

The app computes the correlation coefficient for multiple shift values within the user-defined range:

```text
shift_min → shift_max
```

## Installation

### Clone the repository

```bash
git clone https://github.com/harikrishnannalinakumar/Gamma-Matching.git
cd Gamma-Matching
```

### Clone the repository
```bash
pip install -r requirements.txt
```

### Run the app
``` bash
streamlit run depth_match_streamlit.py
```
