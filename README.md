# smart_parking 🅿️

An AI-powered parking lot monitoring system using YOLOv11 and computer vision to detect and track available parking spots in real-time.

## Features

- **🚗 Real-time Detection**: YOLOv11-based vehicle detection
- **📍 Space Tracking**: Monitor predefined parking slots with live occupancy status
- **🌙 Night Mode**: Advanced image processing for low-light conditions
- **📊 Live Dashboard**: Real-time statistics (total, occupied, available spots)
- **🎯 Occupancy Analysis**: Calculates percentage occupancy per slot

## Quick Start

### Installation

```bash
git clone https://github.com/yoyo2-hub/smart_parking.git
cd smart_parking
pip install opencv-python ultralytics numpy scipy shapely
```

### Usage

**1. Define parking spaces:**
```bash
python spot_selector.py
```
- Left-click to add corner points (4 per space)
- Right-click to delete, 'R' to reset, 'Q' to save

**2. (Optional) Enhance night footage:**
```bash
python nuit.py
```

**3. Run monitoring:**
```bash
python parking_monitor.py
```

## Project Structure

```
smart_parking/
├── nuit.py                    # Night mode transformer
├── spot_selector.py           # Interactive slot selector
├── parking_monitor.py         # Main monitoring system
├── parking_slots.pkl          # Saved spaces
└── video.mp4                  # Input video
```

## Configuration

Edit parameters in `parking_monitor.py`:

```python
VIDEO_PATH = "video.mp4"
FRAME_SKIP = 3                 # Process every Nth frame
BUFFER_SIZE = 12               # Smoothing buffer
OCCUPANCY_THRESHOLD = 45       # Detection sensitivity (%)
```

## How It Works

1. **Frame Capture** → **Camera Tracking** (ORB features) → **Vehicle Detection** (YOLOv11)
2. **Spatial Analysis** (Polygon intersection) → **Occupancy Smoothing** → **Visualization**

## Output

Visual indicators:
- 🟢 Green = Available
- 🔴 Red = Occupied

Real-time dashboard showing total, occupied, and available spots.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera movement errors | Increase `BUFFER_SIZE` |
| Poor low-light detection | Run `nuit.py` or lower `conf` threshold |
| False positives | Increase `OCCUPANCY_THRESHOLD` to 55-60 |

**Made with ❤️ for smarter parking management**
