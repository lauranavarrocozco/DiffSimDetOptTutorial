"""
JAX-based Differentiable Simulation Tutorial: Detector Optimization

This tutorial demonstrates how to use JAX for optimizing detector parameters
in a physics simulation with a collimated light source.

Problem setup:
- Collimated light source along X-axis with exponential intensity decay
- Sensor at distance R=1 with adjustable position (phi_center, theta_center) and size (delta_phi, delta_theta)
- Goal: Optimize sensor size to minimize loss function
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# Random key for reproducibility
key = jax.random.PRNGKey(42)

# Constants
R = 1.0  # Fixed distance to sensor
N_RAYS = 10000  # Number of rays to sample
SENSOR_GOAL_AREA = 0.1  # Target sensor area
PHI_DECAY = 5.0  # degrees - intensity decay parameter for phi
THETA_DECAY = 10.0  # degrees - intensity decay parameter for theta

def spherical_to_cartesian(phi, theta, r=1.0):
    """Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        phi: Azimuthal angle (0 to 2π)
        theta: Polar angle (0 to π/2 for +X hemisphere)
        r: Radial distance
    
    Returns:
        (x, y, z) coordinates
    """
    x = r * jnp.cos(phi) * jnp.sin(theta)
    y = r * jnp.sin(phi) * jnp.sin(theta)
    z = r * jnp.cos(theta)
    return x, y, z

def light_intensity(phi, theta):
    """Calculate light intensity based on collimated source model.
    
    Intensity decreases exponentially with opening angles from X-axis.
    
    Args:
        phi: Azimuthal angles in radians
        theta: Polar angles in radians
    
    Returns:
        Intensity weights for each ray
    """
    phi_deg = jnp.degrees(phi)
    theta_deg = jnp.degrees(theta)
    
    # Exponential decay in both phi and theta directions
    intensity = jnp.exp(-jnp.abs(phi_deg) / PHI_DECAY) * jnp.exp(-jnp.abs(theta_deg) / THETA_DECAY)
    return intensity

def sample_rays(key, n_rays):
    """Sample N rays uniformly in phi, theta over +X hemisphere.
    
    Args:
        key: JAX random key
        n_rays: Number of rays to sample
    
    Returns:
        phi, theta arrays for sampled rays
    """
    key1, key2 = jax.random.split(key)
    
    # Sample uniformly in phi: [-π/2, π/2] for +X hemisphere
    phi = jax.random.uniform(key1, (n_rays,), minval=-jnp.pi/2, maxval=jnp.pi/2)
    
    # Sample uniformly in cos(theta) for uniform distribution on sphere
    # theta: [0, π/2] for +X hemisphere
    cos_theta = jax.random.uniform(key2, (n_rays,), minval=0.0, maxval=1.0)
    theta = jnp.arccos(cos_theta)
    
    return phi, theta

def sensor_area_from_angles(delta_phi, delta_theta):
    """Calculate sensor area from angular dimensions.
    
    For small angles, solid angle ≈ delta_phi * delta_theta
    
    Args:
        delta_phi: Angular width in phi direction
        delta_theta: Angular width in theta direction
    
    Returns:
        Sensor area (solid angle)
    """
    return delta_phi * delta_theta

def ray_intersects_sensor(ray_phi, ray_theta, sensor_phi_center, sensor_theta_center, delta_phi, delta_theta):
    """Check if rays intersect with sensor area.
    
    Args:
        ray_phi, ray_theta: Ray directions
        sensor_phi_center, sensor_theta_center: Sensor center position
        delta_phi, delta_theta: Sensor angular dimensions
    
    Returns:
        Boolean mask indicating which rays intersect the sensor
    """
    # Check if ray is within sensor angular bounds
    phi_in_range = jnp.abs(ray_phi - sensor_phi_center) <= delta_phi / 2
    theta_in_range = jnp.abs(ray_theta - sensor_theta_center) <= delta_theta / 2
    
    return phi_in_range & theta_in_range

@jax.jit
def compute_loss(delta_phi, delta_theta, ray_phi, ray_theta, ray_weights, 
                sensor_phi_center, sensor_theta_center, sensor_goal_area):
    """Compute the loss function for optimization.
    
    Loss = 1/total_w + (sensor_goal_area - sensor_area)²
    
    Args:
        delta_phi, delta_theta: Sensor size parameters to optimize
        ray_phi, ray_theta: Pre-sampled ray directions
        ray_weights: Pre-computed ray intensity weights
        sensor_phi_center, sensor_theta_center: Fixed sensor position
        sensor_goal_area: Target sensor area
    
    Returns:
        Loss value
    """
    # Calculate sensor area
    sensor_area = sensor_area_from_angles(delta_phi, delta_theta)
    
    # Find rays that intersect the sensor
    intersecting = ray_intersects_sensor(ray_phi, ray_theta, sensor_phi_center, 
                                       sensor_theta_center, delta_phi, delta_theta)
    
    # Total weight of intersecting rays
    total_w = jnp.sum(ray_weights * intersecting)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    total_w = jnp.maximum(total_w, epsilon)
    
    # Loss function
    loss = 1.0 / total_w + (sensor_goal_area - sensor_area) ** 2
    
    return loss

def optimize_sensor_size(sensor_phi_center, sensor_theta_center, ray_phi, ray_theta, ray_weights,
                        initial_delta_phi=0.2, initial_delta_theta=0.2, learning_rate=0.01, n_steps=1000):
    """Optimize sensor size using gradient descent.
    
    Args:
        sensor_phi_center, sensor_theta_center: Fixed sensor position
        ray_phi, ray_theta: Pre-sampled ray directions
        ray_weights: Pre-computed ray intensity weights
        initial_delta_phi, initial_delta_theta: Initial sensor size
        learning_rate: Learning rate for gradient descent
        n_steps: Number of optimization steps
    
    Returns:
        Optimization history and final parameters
    """
    # Initialize parameters
    delta_phi = initial_delta_phi
    delta_theta = initial_delta_theta
    
    # Create gradient function
    grad_fn = jax.grad(compute_loss, argnums=(0, 1))
    
    # Storage for optimization history
    history = {
        'delta_phi': [],
        'delta_theta': [],
        'loss': [],
        'total_w': [],
        'sensor_area': []
    }
    
    print(f"Optimizing sensor at position (phi={jnp.degrees(sensor_phi_center):.1f}°, theta={jnp.degrees(sensor_theta_center):.1f}°)")
    print(f"Initial: delta_phi={jnp.degrees(delta_phi):.2f}°, delta_theta={jnp.degrees(delta_theta):.2f}°")
    
    for step in range(n_steps):
        # Compute gradients
        grad_delta_phi, grad_delta_theta = grad_fn(delta_phi, delta_theta, ray_phi, ray_theta, 
                                                  ray_weights, sensor_phi_center, sensor_theta_center, 
                                                  SENSOR_GOAL_AREA)
        
        # Update parameters
        delta_phi -= learning_rate * grad_delta_phi
        delta_theta -= learning_rate * grad_delta_theta
        
        # Ensure positive values
        delta_phi = jnp.maximum(delta_phi, 0.01)  # Minimum size constraint
        delta_theta = jnp.maximum(delta_theta, 0.01)
        
        # Compute current loss and metrics
        loss = compute_loss(delta_phi, delta_theta, ray_phi, ray_theta, ray_weights,
                          sensor_phi_center, sensor_theta_center, SENSOR_GOAL_AREA)
        
        # Additional metrics for monitoring
        intersecting = ray_intersects_sensor(ray_phi, ray_theta, sensor_phi_center, 
                                           sensor_theta_center, delta_phi, delta_theta)
        total_w = jnp.sum(ray_weights * intersecting)
        sensor_area = sensor_area_from_angles(delta_phi, delta_theta)
        
        # Store history
        history['delta_phi'].append(float(delta_phi))
        history['delta_theta'].append(float(delta_theta))
        history['loss'].append(float(loss))
        history['total_w'].append(float(total_w))
        history['sensor_area'].append(float(sensor_area))
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step:4d}: Loss={loss:.6f}, delta_phi={jnp.degrees(delta_phi):.2f}°, "
                  f"delta_theta={jnp.degrees(delta_theta):.2f}°, total_w={total_w:.3f}")
    
    final_results = {
        'delta_phi': float(delta_phi),
        'delta_theta': float(delta_theta),
        'final_loss': float(loss),
        'history': history
    }
    
    print(f"Final: delta_phi={jnp.degrees(delta_phi):.2f}°, delta_theta={jnp.degrees(delta_theta):.2f}°, Loss={loss:.6f}")
    return final_results

def plot_optimization_results(results_list, positions_list):
    """Plot optimization results for all sensor positions."""
    fig, axes = plt.subplots(1, 5, figsize=(12, 4))
    
    colors = ['blue', 'red', 'green']
    
    # Create legend labels for all positions
    legend_labels = []
    for i, pos in enumerate(positions_list):
        legend_labels.append(f"Pos {i+1}: φ={jnp.degrees(pos[0]):.1f}°, θ={jnp.degrees(pos[1]):.1f}°")
    
    for i, (results, pos) in enumerate(zip(results_list, positions_list)):
        history = results['history']
        label = legend_labels[i]
        
        # Loss evolution
        axes[0].plot(history['loss'], color=colors[i], label=label)
        
        # Delta phi evolution
        axes[1].plot(np.degrees(history['delta_phi']), color=colors[i], label=label)
        
        # Delta theta evolution
        axes[2].plot(np.degrees(history['delta_theta']), color=colors[i], label=label)
        
        # Total weight evolution
        axes[3].plot(history['total_w'], color=colors[i], label=label)
        
        # Sensor area vs goal
        axes[4].plot(history['sensor_area'], color=colors[i], label=label)
    
    # Configure axes
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Iteration')
    axes[0].set_title('Loss Evolution')
    axes[0].set_yscale('log')
    
    axes[1].set_ylabel('δφ (degrees)')
    axes[1].set_xlabel('Iteration')
    axes[1].set_title('δφ Evolution')
    
    axes[2].set_ylabel('δθ (degrees)')
    axes[2].set_xlabel('Iteration')
    axes[2].set_title('δθ Evolution')
    
    axes[3].set_ylabel('Total Weight')
    axes[3].set_xlabel('Iteration')
    axes[3].set_title('Collected Light Weight')
    
    axes[4].axhline(y=SENSOR_GOAL_AREA, color='black', linestyle=':')
    axes[4].text(axes[4].get_xlim()[1] * 0.3, SENSOR_GOAL_AREA + 0.005, 'Target Area', 
                 verticalalignment='bottom', color='black', fontsize=10)
    axes[4].set_ylabel('Sensor Area')
    axes[4].set_xlabel('Iteration')
    axes[4].set_title('Sensor Area vs Target')
    axes[4].set_ylim(0, SENSOR_GOAL_AREA*1.2)
    
    # Create a single legend at the top center
    fig.legend(legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)  # Make room for the legend
    plt.savefig('optimization_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the optimization tutorial."""
    print("=== JAX-based Differentiable Simulation Tutorial ===")
    print("Optimizing detector size for collimated light source\n")
    
    # Sample rays once for all optimizations
    print(f"Sampling {N_RAYS} rays...")
    ray_phi, ray_theta = sample_rays(key, N_RAYS)
    ray_weights = light_intensity(ray_phi, ray_theta)
    
    print(f"Total sampled light intensity: {jnp.sum(ray_weights):.3f}")
    print(f"Average ray weight: {jnp.mean(ray_weights):.6f}\n")
    
    # Define 3 different sensor positions for optimization
    sensor_positions = [
        (jnp.radians(0), jnp.radians(0)),      # φ=0°, θ=30°
        (jnp.radians(15), jnp.radians(45)),     # φ=15°, θ=45°
        (jnp.radians(-20), jnp.radians(60))     # φ=-20°, θ=60°
    ]
    
    results_list = []
    
    # Run optimization for each sensor position
    for i, (phi_center, theta_center) in enumerate(sensor_positions):
        print(f"\n{'='*50}")
        print(f"OPTIMIZATION {i+1}/3")
        print(f"{'='*50}")
        
        results = optimize_sensor_size(
            phi_center, theta_center, ray_phi, ray_theta, ray_weights,
            initial_delta_phi=jnp.radians(np.random.uniform(5,15)),  # Start with 10° width
            initial_delta_theta=jnp.radians(np.random.uniform(2,20)),  # Start with 10° height
            learning_rate=0.1,
            n_steps=250
        )
        
        results_list.append(results)
    
    # Plot results
    print(f"\n{'='*50}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*50}")
    
    for i, (results, pos) in enumerate(zip(results_list, sensor_positions)):
        print(f"Position {i+1} (φ={jnp.degrees(pos[0]):.1f}°, θ={jnp.degrees(pos[1]):.1f}°):")
        print(f"  Optimal δφ = {jnp.degrees(results['delta_phi']):.2f}°")
        print(f"  Optimal δθ = {jnp.degrees(results['delta_theta']):.2f}°")
        print(f"  Final loss = {results['final_loss']:.6f}")
        print(f"  Final area = {results['history']['sensor_area'][-1]:.4f} (target: {SENSOR_GOAL_AREA})")
        print()
    
    # Create visualization
    plot_optimization_results(results_list, sensor_positions)
    
    print("Optimization complete! Results saved to 'optimization_results.png'")

if __name__ == "__main__":
    main()