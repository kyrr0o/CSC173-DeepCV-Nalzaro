# CSC173 Deep Computer Vision Project Progress Report
**Student:** Kaycee T. Nalzaro, 2021-1756
**Date:** December 15, 2025 
**Repository:** https://github.com/kyrr0o/CSC173-DeepCV-Nalzaro.git

---

## Current Status

| Milestone | Status | Notes |
|-----------|--------|-------|
| Problem Definition | Completed | Drainage risk assessment using video-based CV |
| Dataset Collection | In Progress | Initial drainage and water-flow videos gathered |
| Inference Pipeline | In Progress | YOLOv8 + optical flow integrated |
| Risk Scoring Logic | In Progress | Thresholds and weights under testing |
| Quantitative Evaluation | Pending | No supervised metrics yet |
| Final Documentation | Not Started | Planned after pipeline stabilization |


## 1. Dataset Progress
- **Data type:** Video-based dataset (raw video files)
- **Current videos:** Four initial drainage and water-flow video samples representing different observed conditions:
  - Clean drainage (minimal visible waste, normal flow)
  - Light trash accumulation (scattered waste with visible water movement)
  - Heavy trash accumulation (significant waste coverage)
  - Clogged or near-clogged drainage (limited or stagnant water flow)
- **Data usage:** Videos are processed frame-by-frame for inference
- **Train/Val/Test split:** Not applicable (no supervised training phase)
- **Preprocessing applied:**
  - Frame resizing to fixed width (640 px)
  - Grayscale conversion for optical flow computation

**Notes:**  
At this stage, the project focuses on validating the video inference pipeline rather than collecting a large labeled dataset. To improve data quality and realism,
future iterations will include manually captured drainage videos recorded under controlled conditions. This approach aims to reduce issues commonly observed in online videos, 
such as excessive motion blur, artificial speed-up, and compression artifacts. Additional locally captured videos will also be used to evaluate robustness under varying lighting and water flow conditions.

## 2. Model & Inference Progress

- **Model used:** YOLOv8n (pretrained, no fine-tuning yet)
- **Framework:** PyTorch + Ultralytics YOLOv8 + OpenCV
- **Inference mode:** Frame-level object detection from video
- **Additional CV method:** Farneback optical flow for motion estimation

### Current Inference Outputs
| Metric | Description |
|------|------------|
| Trash Area (%) | Estimated proportion of frame covered by detected objects |
| Optical Flow Magnitude | Mean motion magnitude between consecutive frames |
| Normalized Flow | Flow magnitude normalized to [0,1] |
| Risk Score | Weighted fusion of trash density and inverse flow |
| Risk Level | LOW / MODERATE / HIGH classification |

**Status:**  
The inference pipeline is functional and produces annotated output videos with bounding boxes, flow statistics, and risk labels per frame.

## 3. Challenges Encountered & Solutions

| Issue | Status | Resolution |
|------|--------|------------|
| False detections from non-waste objects | Ongoing | Adjusting confidence thresholds and scoring logic |
| Sensitivity to lighting conditions | Ongoing | Testing on varied video samples |
| Risk threshold tuning | In Progress | Empirical adjustment of weights and thresholds |
| Lack of labeled data | Expected | Using pretrained model and proxy metrics |

## 4. Next Steps (Before Final Submission)
- [ ] Refine risk scoring thresholds and weights
- [ ] Add basic runtime metrics (FPS / inference time)
- [ ] Test pipeline on additional video samples
- [ ] Improve visualization and overlays
- [ ] Record 5-min demo video
- [ ] Write complete README.md with results
- [ ] Record and export final demo video
- [ ] Write complete README.md with final results and discussion
