# Linear Regression Visualization with Gradient Descent

This project demonstrates the implementation of linear regression using the gradient descent method with an animated visualization showing the model's learning process.

## Description

This project uses Python to:
1. Generate synthetic dataset
2. Implement simple linear regression
3. Use gradient descent to find optimal parameters
4. Visualize the learning process with animation

## Features

- Synthetic data generation with noise
- Linear function implementation
- Parameter optimization using gradient descent
- Animated visualization of learning process
- Real-time gradient value display

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

1. Clone this repository
2. Install required dependencies:
```bash
pip install numpy matplotlib
```

## Usage

1. Open `Main.ipynb` using Jupyter Notebook
2. Run all cells in sequence
3. Observe the animation showing the model's learning process

## Code Explanation

### 1. Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
```

### 2. Data Generation
```python
jumlah_data = 100
y = np.array([i*0.1+np.random.randn() for i in range(jumlah_data)])
x = np.array([i*0.1 for i in range(jumlah_data)])
```

### 3. Linear Function
```python
def ucup_linear(x, gradient):
    y = gradient*x
    return y
```

### 4. Gradient Descent
```python
learning_rate = 0.1
for i in range(1,jumlah_data):
    y_prediction = ucup_linear(x[i],m_prediction)
    y_actual = y[i]
    error = y_actual - y_prediction
    delta_m = learning_rate*error/x[i]
    m_prediction = m_prediction + delta_m
```

### 5. Visualization
- Using Matplotlib for plotting
- Animation using FuncAnimation
- Displaying actual data and predictions
- Real-time gradient value updates

## Results

The visualization will show:
- Actual data points (blue)
- Prediction line (red)
- Current gradient value
- Reference grid
- Informative labels and legend

## Contributing

Feel free to contribute by creating pull requests or reporting issues.

## License

MIT License 