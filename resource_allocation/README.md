# Wildfire Resource Allocation with Reinforcement Learning (Work In Progress)

## Overview

This folder has an reinforcement learning pipeline for wildfire resource allocation. The code simulates a wildfire environment using synthetic terrain, population, weather, and environmental sensitivity data (I will integrate the outputs of the ConvLSTM model) on a spatial grid and applies PPO (Proximal Policy Optimization) for resource deployment decisions.

**Key Points:**
- **Experimental Prototype:**  
  The current version uses made up data (since I need to get the data). It does not integrate ConvLSTM model outputs, but the structure is modular so it shouldn't be too hard in the future.
- **Visualization:**  
  The code produces visual outputs (animations, individual frames, final state maps) to help see the simulation.
- **Work-In-Progress:**  
  The current results are not very good (high negative rewards), but they do indicate room for improvement through reward tuning and integration of realistic models. It is a solid baseline though so hopefully someone can help me with this part.

---

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Testing

1. **Open File:**
  - Run the cells in order 

