"""
Particle Swarm Optimization (PSO) Algorithm Implementation

This module provides a flexible and efficient implementation of the PSO algorithm
for continuous optimization problems.

Author: EU97
Date: 2025
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PSOConfig:
    """Configuration parameters for PSO algorithm."""
    n_particles: int = 30
    n_dimensions: int = 4
    n_iterations: int = 100
    w: float = 0.7  # Inertia weight
    c1: float = 1.5  # Cognitive parameter
    c2: float = 1.5  # Social parameter
    w_min: float = 0.4  # Minimum inertia weight (for adaptive)
    w_max: float = 0.9  # Maximum inertia weight (for adaptive)
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    adaptive_inertia: bool = False
    verbose: bool = True


class Particle:
    """
    Represents a single particle in the swarm.
    
    Attributes:
        position: Current position in search space
        velocity: Current velocity
        best_position: Personal best position found
        best_fitness: Fitness value at personal best position
    """
    
    def __init__(self, n_dimensions: int, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Initialize a particle with random position and velocity.
        
        Args:
            n_dimensions: Dimensionality of the search space
            bounds: Optional tuple of (lower_bounds, upper_bounds)
        """
        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            self.position = np.random.uniform(lower_bounds, upper_bounds, n_dimensions)
            velocity_range = (upper_bounds - lower_bounds) * 0.1
            self.velocity = np.random.uniform(-velocity_range, velocity_range, n_dimensions)
        else:
            self.position = np.random.uniform(0, 1, n_dimensions)
            self.velocity = np.random.uniform(-0.1, 0.1, n_dimensions)
        
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
    
    def update_velocity(self, global_best_position: np.ndarray, w: float, c1: float, c2: float):
        """
        Update particle velocity based on PSO equations.
        
        Args:
            global_best_position: Best position found by entire swarm
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
        """
        r1 = np.random.random(self.position.shape)
        r2 = np.random.random(self.position.shape)
        
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Update particle position based on current velocity.
        
        Args:
            bounds: Optional tuple of (lower_bounds, upper_bounds) to enforce
        """
        self.position += self.velocity
        
        # Enforce bounds if provided
        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            self.position = np.clip(self.position, lower_bounds, upper_bounds)
    
    def evaluate(self, fitness_func: Callable) -> float:
        """
        Evaluate fitness at current position and update personal best.
        
        Args:
            fitness_func: Function to evaluate fitness
            
        Returns:
            Current fitness value
        """
        fitness = fitness_func(self.position)
        
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
        
        return fitness


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization algorithm for continuous optimization.
    
    The PSO algorithm maintains a swarm of particles that explore the search space,
    guided by their own best-known positions and the global best-known position.
    """
    
    def __init__(self, config: Optional[PSOConfig] = None, **kwargs):
        """
        Initialize the PSO optimizer.
        
        Args:
            config: PSOConfig object with algorithm parameters
            **kwargs: Alternative way to provide parameters
        """
        if config is not None:
            self.config = config
        else:
            self.config = PSOConfig(**kwargs)
        
        self.particles: List[Particle] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness: float = float('inf')
        self.history: Dict[str, List] = {
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'diversity': []
        }
    
    def _initialize_swarm(self):
        """Initialize all particles in the swarm."""
        self.particles = [
            Particle(self.config.n_dimensions, self.config.bounds)
            for _ in range(self.config.n_particles)
        ]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'diversity': []
        }
    
    def _calculate_diversity(self) -> float:
        """
        Calculate swarm diversity (average distance from centroid).
        
        Returns:
            Diversity measure
        """
        positions = np.array([p.position for p in self.particles])
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return np.mean(distances)
    
    def _get_adaptive_inertia(self, iteration: int) -> float:
        """
        Calculate adaptive inertia weight that decreases linearly over iterations.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Adaptive inertia weight
        """
        w_max = self.config.w_max
        w_min = self.config.w_min
        n_iterations = self.config.n_iterations
        
        return w_max - (w_max - w_min) * iteration / n_iterations
    
    def optimize(
        self, 
        fitness_func: Callable[[np.ndarray], float],
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None
    ) -> Tuple[np.ndarray, float, Dict[str, List]]:
        """
        Run the PSO optimization algorithm.
        
        Args:
            fitness_func: Objective function to minimize
            callback: Optional callback function called each iteration with 
                     (iteration, best_fitness, best_position)
        
        Returns:
            Tuple of (best_position, best_fitness, history)
        """
        # Initialize swarm
        self._initialize_swarm()
        
        if self.config.verbose:
            logger.info(f"Starting PSO optimization with {self.config.n_particles} particles")
            logger.info(f"Dimensions: {self.config.n_dimensions}, Iterations: {self.config.n_iterations}")
        
        # Main optimization loop
        for iteration in range(self.config.n_iterations):
            # Get current inertia weight
            if self.config.adaptive_inertia:
                w = self._get_adaptive_inertia(iteration)
            else:
                w = self.config.w
            
            fitness_values = []
            
            # Evaluate all particles
            for particle in self.particles:
                fitness = particle.evaluate(fitness_func)
                fitness_values.append(fitness)
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
            
            # Update velocities and positions
            for particle in self.particles:
                particle.update_velocity(
                    self.global_best_position,
                    w,
                    self.config.c1,
                    self.config.c2
                )
                particle.update_position(self.config.bounds)
            
            # Record history
            self.history['best_fitness'].append(self.global_best_fitness)
            self.history['mean_fitness'].append(np.mean(fitness_values))
            self.history['std_fitness'].append(np.std(fitness_values))
            self.history['diversity'].append(self._calculate_diversity())
            
            # Progress logging
            if self.config.verbose and (iteration % 10 == 0 or iteration == self.config.n_iterations - 1):
                logger.info(
                    f"Iteration {iteration + 1}/{self.config.n_iterations}: "
                    f"Best Fitness = {self.global_best_fitness:.6f}, "
                    f"Mean Fitness = {np.mean(fitness_values):.6f}, "
                    f"Diversity = {self.history['diversity'][-1]:.6f}"
                )
            
            # Call callback if provided
            if callback is not None:
                callback(iteration, self.global_best_fitness, self.global_best_position)
        
        if self.config.verbose:
            logger.info(f"\nOptimization complete!")
            logger.info(f"Best fitness: {self.global_best_fitness:.6f}")
            logger.info(f"Best position: {self.global_best_position}")
        
        return self.global_best_position, self.global_best_fitness, self.history


def test_pso():
    """Test PSO on benchmark functions."""
    
    # Sphere function: f(x) = sum(x_i^2)
    def sphere(x):
        return np.sum(x**2)
    
    # Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    print("Testing PSO on Sphere function (minimum at origin)...")
    config = PSOConfig(
        n_particles=30,
        n_dimensions=5,
        n_iterations=100,
        bounds=(np.array([-5.0] * 5), np.array([5.0] * 5)),
        adaptive_inertia=True
    )
    
    pso = ParticleSwarmOptimizer(config)
    best_pos, best_fit, history = pso.optimize(sphere)
    
    print(f"\nSphere Function Results:")
    print(f"Best position: {best_pos}")
    print(f"Best fitness: {best_fit}")
    print(f"Expected: 0.0")
    
    print("\n" + "="*50 + "\n")
    print("Testing PSO on Rastrigin function (minimum at origin)...")
    
    pso2 = ParticleSwarmOptimizer(config)
    best_pos2, best_fit2, history2 = pso2.optimize(rastrigin)
    
    print(f"\nRastrigin Function Results:")
    print(f"Best position: {best_pos2}")
    print(f"Best fitness: {best_fit2}")
    print(f"Expected: 0.0")


if __name__ == "__main__":
    test_pso()
