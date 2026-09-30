# PenPal

Run the green tape camera filter from the repository root:

```powershell
.\CV\.venv\Scripts\python.exe .\CV\isolate_green.py
```

The window shows green pixels from camera `0` against black; press `Q` or `Esc`, or close the window, to exit. Change the number in `cv2.VideoCapture(0)` if needed.

The HSV range `(45, 60, 40)` to `(65, 255, 255)` was sampled from the laptop camera photo `WIN_20260930_19_16_08_Pro.jpg`, using OpenCV's hue scale of 0–179. Different camera white balance or lighting may require adjusting `lower_green` and `upper_green`; other objects of the same color will also remain visible.

Dependencies are listed in `CV/requirements.txt`.
