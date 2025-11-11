"""
Unit tests for PSO algorithm

Author: EU97
Date: 2024
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.pso import ParticleSwarmOptimizer, PSOConfig, Particle


class TestParticle:
    """Test cases for Particle class."""
    
    def test_particle_initialization(self):
        """Test particle initialization."""
        particle = Particle(n_dimensions=5)
        
        assert particle.position.shape == (5,)
        assert particle.velocity.shape == (5,)
        assert particle.best_position.shape == (5,)
        assert particle.best_fitness == float('inf')
    
    def test_particle_with_bounds(self):
        """Test particle initialization with bounds."""
        lower_bounds = np.array([0, 0, 0])
        upper_bounds = np.array([1, 1, 1])
        particle = Particle(n_dimensions=3, bounds=(lower_bounds, upper_bounds))
        
        assert np.all(particle.position >= 0)
        assert np.all(particle.position <= 1)
    
    def test_particle_evaluation(self):
        """Test particle fitness evaluation."""
        particle = Particle(n_dimensions=3)
        
        def sphere_func(x):
            return np.sum(x**2)
        
        fitness = particle.evaluate(sphere_func)
        
        assert isinstance(fitness, float)
        assert fitness >= 0
        assert particle.best_fitness <= float('inf')
    
    def test_particle_update_velocity(self):
        """Test velocity update."""
        particle = Particle(n_dimensions=3)
        global_best = np.array([0.5, 0.5, 0.5])
        
        old_velocity = particle.velocity.copy()
        particle.update_velocity(global_best, w=0.7, c1=1.5, c2=1.5)
        
        assert not np.array_equal(particle.velocity, old_velocity)
    
    def test_particle_update_position(self):
        """Test position update."""
        particle = Particle(n_dimensions=3)
        old_position = particle.position.copy()
        
        particle.velocity = np.array([0.1, 0.1, 0.1])
        particle.update_position()
        
        assert not np.array_equal(particle.position, old_position)


class TestPSOConfig:
    """Test cases for PSOConfig."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = PSOConfig()
        
        assert config.n_particles == 30
        assert config.n_dimensions == 4
        assert config.n_iterations == 100
        assert config.w == 0.7
        assert config.c1 == 1.5
        assert config.c2 == 1.5
    
    def test_config_custom(self):
        """Test custom configuration."""
        config = PSOConfig(
            n_particles=50,
            n_dimensions=10,
            n_iterations=200,
            w=0.8,
            adaptive_inertia=True
        )
        
        assert config.n_particles == 50
        assert config.n_dimensions == 10
        assert config.n_iterations == 200
        assert config.w == 0.8
        assert config.adaptive_inertia == True


class TestParticleSwarmOptimizer:
    """Test cases for PSO optimizer."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        config = PSOConfig(n_particles=20, n_dimensions=5, n_iterations=50)
        pso = ParticleSwarmOptimizer(config)
        
        assert pso.config.n_particles == 20
        assert pso.config.n_dimensions == 5
        assert pso.global_best_fitness == float('inf')
    
    def test_sphere_function_optimization(self):
        """Test optimization on sphere function."""
        def sphere(x):
            return np.sum(x**2)
        
        config = PSOConfig(
            n_particles=30,
            n_dimensions=5,
            n_iterations=50,
            bounds=(np.array([-5.0]*5), np.array([5.0]*5)),
            verbose=False
        )
        
        pso = ParticleSwarmOptimizer(config)
        best_pos, best_fit, history = pso.optimize(sphere)
        
        # Check convergence (should be close to zero)
        assert best_fit < 1.0
        assert len(history['best_fitness']) == 50
        
        # Check that fitness improved
        assert history['best_fitness'][-1] <= history['best_fitness'][0]
    
    def test_rastrigin_function_optimization(self):
        """Test optimization on Rastrigin function."""
        def rastrigin(x):
            return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        
        config = PSOConfig(
            n_particles=50,
            n_dimensions=3,
            n_iterations=100,
            bounds=(np.array([-5.12]*3), np.array([5.12]*3)),
            adaptive_inertia=True,
            verbose=False
        )
        
        pso = ParticleSwarmOptimizer(config)
        best_pos, best_fit, history = pso.optimize(rastrigin)
        
        # Rastrigin is harder, so we just check improvement
        assert best_fit < 30  # Should find something reasonable
        assert history['best_fitness'][-1] < history['best_fitness'][0]
    
    def test_adaptive_inertia(self):
        """Test adaptive inertia weight."""
        config = PSOConfig(
            n_particles=10,
            n_dimensions=3,
            n_iterations=10,
            adaptive_inertia=True,
            w_min=0.4,
            w_max=0.9,
            verbose=False
        )
        
        pso = ParticleSwarmOptimizer(config)
        
        # Test inertia calculation at different iterations
        w_start = pso._get_adaptive_inertia(0)
        w_mid = pso._get_adaptive_inertia(5)
        w_end = pso._get_adaptive_inertia(9)
        
        assert w_start == 0.9
        assert w_end == 0.4
        assert w_start > w_mid > w_end
    
    def test_diversity_calculation(self):
        """Test swarm diversity calculation."""
        config = PSOConfig(n_particles=10, n_dimensions=3, verbose=False)
        pso = ParticleSwarmOptimizer(config)
        pso._initialize_swarm()
        
        diversity = pso._calculate_diversity()
        
        assert isinstance(diversity, float)
        assert diversity >= 0
    
    def test_callback_function(self):
        """Test callback function during optimization."""
        callback_data = []
        
        def callback(iteration, best_fitness, best_position):
            callback_data.append({
                'iteration': iteration,
                'fitness': best_fitness,
                'position': best_position.copy()
            })
        
        def sphere(x):
            return np.sum(x**2)
        
        config = PSOConfig(
            n_particles=10,
            n_dimensions=3,
            n_iterations=10,
            verbose=False
        )
        
        pso = ParticleSwarmOptimizer(config)
        pso.optimize(sphere, callback=callback)
        
        assert len(callback_data) == 10
        assert callback_data[0]['iteration'] == 0
        assert callback_data[-1]['iteration'] == 9
    
    def test_bounds_enforcement(self):
        """Test that bounds are enforced."""
        def sphere(x):
            return np.sum(x**2)
        
        lower = np.array([0, 0, 0])
        upper = np.array([1, 1, 1])
        
        config = PSOConfig(
            n_particles=20,
            n_dimensions=3,
            n_iterations=50,
            bounds=(lower, upper),
            verbose=False
        )
        
        pso = ParticleSwarmOptimizer(config)
        best_pos, best_fit, history = pso.optimize(sphere)
        
        # Check that all particles are within bounds
        for particle in pso.particles:
            assert np.all(particle.position >= lower)
            assert np.all(particle.position <= upper)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
