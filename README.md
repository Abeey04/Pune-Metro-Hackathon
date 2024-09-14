# Pune Metro Hackathon: Person Fall Detection System

## Project Overview

This repository contains the code and documentation for the Person Fall Detection System, developed as part of the Pune Metro Hackathon. The system aims to detect individuals who have fallen onto metro tracks, providing a crucial safety mechanism for metro operations.

### Problem Statement

The challenge was to create a robust system capable of detecting persons who have fallen onto metro tracks, thereby entering a designated danger zone. This system is designed to raise an alarm in such scenarios, potentially preventing accidents and saving lives.

## Table of Contents

1. [Technical Overview](#technical-overview)
2. [System Architecture](#system-architecture)
3. [Models Used](#models-used)
4. [Dataset](#dataset)
5. [Implementation Details](#implementation-details)
6. [Results](#results)
7. [Future Improvements](#future-improvements)
8. [Contributing](#contributing)
9. [License](#license)

## Technical Overview

The Person Fall Detection System utilizes two primary components:

1. **Person Detection**: Implemented using YOLOv7
2. **Track Segmentation**: Implemented using YOLOv5

These components work in tandem to identify persons and define the danger zone (metro tracks). The system then calculates the intersection between detected persons and the danger zone to determine if a fall has occurred.

## System Architecture

The system follows this high-level workflow:

1. Input video frame
2. Parallel processing:
   a. Person detection using YOLOv7
   b. Track segmentation using YOLOv5
3. Polygon creation from segmentation mask
4. Intersection calculation between person bounding boxes and track polygon
5. Alarm triggering based on intersection results

### Outputs
Output showing the track, i.e the danger zone is being updated every frame, even when the metro arrives, the danger zone basically adjusts itself automatically:
![](https://github.com/Abeey04/Pune-Metro-Hackathon/blob/bbd9fb8a6d5118b5962606fa092f252ce2e5d038/Outputs/output.gif)
![](https://youtu.be/w5n-f5stuvA)
[Output 2]()
## Models Used

### YOLOv7 for Person Detection

- **Purpose**: To detect and localize persons in each frame
- **Modifications**: Fine-tuned to detect only persons
- **Output**: Bounding boxes around detected persons

### YOLOv5 for Track Segmentation

- **Purpose**: To segment the metro tracks (danger zone) in each frame
- **Modifications**: Fine-tuned for instance segmentation of metro tracks
- **Output**: Segmentation mask of the metro tracks

## Dataset

The dataset used for this project was provided by the Pune Metro Hackathon organizers. It includes:

- Images of metro stations and tracks
- Annotations for person locations
- Annotations for track locations and boundaries

This dataset was used to fine-tune both the YOLOv7 and YOLOv5 models for their respective tasks.

## Implementation Details

### Person Detection

1. Fine-tuned YOLOv7 on the provided dataset
2. Configured to output only person class detections
3. Applied to each frame of the input video

### Track Segmentation

1. Fine-tuned YOLOv5 for instance segmentation of metro tracks
2. Applied to each frame to generate a segmentation mask of the tracks

### Intersection Calculation

1. Used Shapely library to create a polygon from the track segmentation mask
2. For each detected person:
   a. Created a polygon from the bounding box
   b. Calculated the intersection between the person polygon and track polygon
3. If intersection exists, classify as a potential fall event

```

## Results

[Include information about the system's performance, accuracy, and any metrics from the hackathon evaluation]

## Future Improvements

1. Implement real-time processing for live video feeds
2. Enhance the system to work under various lighting conditions
3. Integrate with metro station alarm systems
4. Develop a user interface for security personnel
5. Explore the use of 3D sensors for more accurate depth perception

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Developed with ❤️ for the Pune Metro Hackathon
