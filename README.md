# JAX-based Differentiable Simulation Tutorial: Detector Optimization

A comprehensive tutorial demonstrating how to use JAX for optimizing detector parameters in physics simulations through automatic differentiation.

## 🎯 Problem Setup

This tutorial tackles a realistic detector optimization problem:

- **Light Source**: Collimated light beam along the X-axis with exponential intensity decay
- **Sensor**: Positioned at distance R=1 with adjustable angular size (δφ, δθ)
- **Goal**: Optimize sensor dimensions to maximize light collection while maintaining target area

## 🔬 Physical Model

### Light Source Model
The collimated light source follows an exponential intensity profile:
```
I(φ, θ) = exp(-|φ|/φ_decay) × exp(-|θ|/θ_decay)
```
- φ_decay = 5° (azimuthal decay parameter)
- θ_decay = 10° (polar decay parameter)

### Sensor Model
- **Position**: Fixed at (φ_center, θ_center) on unit sphere
- **Size**: Variable angular dimensions (δφ, δθ)
- **Detection**: Ray intersection with sensor angular bounds

### Loss Function
```
Loss = 1/total_w + (sensor_goal_area - sensor_area)²
```
- `total_w`: Total weight of rays hitting the sensor
- `sensor_goal_area`: Target sensor area (0.1)
- `sensor_area`: Current sensor area (δφ × δθ)

## 🛠️ Installation

```bash
pip install jax jaxlib numpy matplotlib
```

## 💻 Usage

Run the complete tutorial:
```bash
python detector_optimization_tutorial.py
```

This will:
1. Sample 10,000 rays with realistic intensity weighting
2. Run optimization from 3 different sensor positions
3. Generate comprehensive visualization plots
4. Save results to `optimization_results.png`

## 📈 Visualization

The tutorial generates a 5-panel visualization showing:

1. **Loss Evolution**: Convergence behavior over iterations
2. **δφ Evolution**: Optimization of azimuthal sensor size
3. **δθ Evolution**: Optimization of polar sensor size  
4. **Light Collection**: Total weighted rays hitting sensor
5. **Sensor Area**: Convergence toward target area

## 🔧 Customization

Key parameters you can modify:

```python
N_RAYS = 10000              # Number of sampled rays
SENSOR_GOAL_AREA = 0.1      # Target sensor area
PHI_DECAY = 5.0             # Intensity decay in φ direction
THETA_DECAY = 10.0          # Intensity decay in θ direction
learning_rate = 0.1         # Gradient descent step size
n_steps = 250               # Optimization iterations
```

## 📄 License

MIT License - feel free to use this tutorial for educational and research purposes.