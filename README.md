Robot Validation Platform

A visualization and validation platform for mobile robot navigation, trajectory analysis, and virtual-physical mapping in Substitutional Reality.

Overview

This project provides a validation and analysis platform for mobile robot experiments in Substitutional Reality (SR). The platform integrates robot navigation management, virtual-physical mapping verification, synchronized playback, and trajectory analysis.

The system combines ROS2, Unity, Meta Quest 3, ArUco Marker tracking, and a data visualization platform to support robot control, experiment validation, and navigation analysis.

⸻

Features

* Mobile robot navigation management
* Virtual-physical mapping verification
* ArUco Marker tracking
* Synchronized playback of:
    * Robot telemetry
    * Real-world video
    * MR video
* DTW (Dynamic Time Warping) trajectory analysis
* Navigation event tracking
* Data visualization dashboard

⸻

How to Use

1. Record robot navigation experiments and export navigation logs.
2. Import CSV files and recorded videos into the platform.
3. Replay experiments, visualize trajectories, and analyze virtual-physical mapping results.

⸻

System Architecture

Components

1. Mobile Robot (ROS2)
2. Robot Management Interface (Unity)
3. MR Headset (Meta Quest 3)
4. Robot Data Visualization Platform

Technology Stack

* ROS2
* Unity
* Meta XR SDK
* ArUco Marker
* Python
* HTML
* JavaScript

⸻

Validation Scenarios

The platform was validated using four navigation paths:

* Two-point Route
* Triangle Route
* Square Route
* Hexagon Route

For each route:

* 10 navigation laps were executed
* Navigation time was recorded
* Trajectory data was collected
* DTW analysis was performed

⸻

DTW Trajectory Analysis

The platform supports Dynamic Time Warping (DTW) analysis for comparing:

* Robot navigation trajectory
* ArUco tracking trajectory

Metrics include:

* DTW Distance
* Normalized DTW Distance

These metrics are used to evaluate virtual-physical mapping consistency.

⸻
