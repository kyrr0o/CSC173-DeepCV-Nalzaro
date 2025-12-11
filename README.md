# CanalGuard AI: A CCTV-Based Deep Learning Model for Monitoring Water Flow, Trash Accumulation, and Clogging Risk in Urban Drainage Systems

CSC173 Intelligent Systems Final Project
Mindanao State University - Iligan Institute of Technology
Student: Kaycee T. Nalzaro, 2021-1756
Semester: AY 2025-2026 Sem 1

## Abstract
Urban flooding in the Philippines is frequently caused not only by heavy rainfall but also by clogged drainage canals that accumulate trash and experience reduced water flow. Manual inspection is inconsistent, unsafe during bad weather, and cannot provide continuous monitoring.  
This project introduces **DrainSight**, a lightweight computer vision pipeline that uses:
- a **pretrained YOLOv8 model** to detect trash and debris,  
- **optical flow** to estimate water movement, and  
- a combined **clogging risk score** (LOW / MODERATE / HIGH)  
