# Requirements

## Python

Tested with Python 3.10.

```
opencv-python==4.10.0.84
numpy==2.0.0,<3
pandas==2.2.2
colour-science==0.4.5
colour-checker-detection==0.2.3
openpyxl==3.1.2
ultralytics==8.4.69
matplotlib==3.8.4
Pillow==10.3.0
rawpy==0.27.0
```

Copy the package list above into a `requirements.txt` file, then install with:

```
pip install -r requirements.txt
```

## R

Required for the validation scripts in `validation/colour_calibration/` and `validation/morphometrics/`. Tested with R 4.3.

```
install.packages(c("here", "dplyr", "readr", "ggplot2", "patchwork"))
```
