# 🪐 Exoplanet Detection with Machine Learning

A deep learning project that uses PyTorch and other libraries to predict whether Kepler Objects of Interest (KOIs) are confirmed exoplanets based on their observable characteristics.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Team](#team)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🌟 Overview

This project implements a binary classification neural network to identify confirmed exoplanets from NASA's Kepler mission data. The model analyzes five key planetary features and predicts whether a candidate is a confirmed exoplanet with a confidence score.

**Key Highlights:**
- 🎯 Binary classification (Confirmed vs. Not Confirmed)
- 🧠 Multi-layer neural network with dropout regularization
- 📊 Trained on real NASA Kepler mission data
- 💻 Interactive command-line interface for custom predictions
- 📈 Comprehensive performance metrics (accuracy, precision, recall, F1-score)

## ✨ Features

The model uses five observable planetary characteristics:

1. **Orbital Period (koi_period)** - Time to complete one orbit around the star (days)
2. **Transit Duration (koi_duration)** - How long the planet blocks the star's light (hours)
3. **Transit Depth (koi_depth)** - Reduction in star brightness during transit (parts per million)
4. **Planet Radius (koi_prad)** - Size of the planet relative to Earth (Earth radii)
5. **Stellar Radius (koi_srad)** - Size of the host star relative to our Sun (Solar radii)

## 📊 Dataset

- **Source:** NASA Exoplanet Archive - Kepler Cumulative Dataset
- **Date:** October 5, 2025 (cumulative_2025.10.05_12.30.33.csv)
- **Link:** [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- **Preprocessing:** 
  - Removed rows with missing values
  - Binary labeling: 1 for CONFIRMED, 0 for NOT CONFIRMED

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA Exoplanet Archive** for providing the comprehensive Kepler mission dataset
- **Kepler Mission** scientists and engineers for their groundbreaking work in exoplanet discovery
- The astronomy and machine learning communities for their invaluable resources

## 📧 Team

Cesar Pimentel - cpimentelortiz23@gmail.com

Pranshu Shah - hoangky271106@gmail.com

Ky Trieu - pranshushah2024@gmail.com

Richard Juuko - email

Jonathan Joseph - jpjoseph8145@gmail.com

---

⭐ If you found this project helpful, please consider giving it a star!

**Built for the Nasa Space Apps Hackathon**
