# Theory and Background

Mathematical foundations and theory behind PSO and Portfolio Optimization.

## Table of Contents

1. [Particle Swarm Optimization](#particle-swarm-optimization)
2. [Modern Portfolio Theory](#modern-portfolio-theory)
3. [Sharpe Ratio Optimization](#sharpe-ratio-optimization)
4. [Convergence Analysis](#convergence-analysis)
5. [References](#references)

---

## Particle Swarm Optimization

### Overview

Particle Swarm Optimization (PSO) is a population-based stochastic optimization algorithm inspired by the social behavior of bird flocking and fish schooling. Developed by Kennedy and Eberhart in 1995, PSO has become one of the most popular metaheuristic algorithms due to its simplicity and effectiveness.

### Algorithm Concept

In PSO, a swarm of particles explores the search space, where each particle represents a potential solution. Particles move through the space influenced by:

1. **Inertia**: Tendency to continue in current direction
2. **Cognitive Component**: Attraction to particle's personal best
3. **Social Component**: Attraction to swarm's global best

### Mathematical Formulation

#### Position and Velocity Updates

For each particle $i$ at iteration $t$:

**Velocity Update:**
$$v_i(t+1) = w \cdot v_i(t) + c_1 \cdot r_1 \cdot (p_i - x_i(t)) + c_2 \cdot r_2 \cdot (g - x_i(t))$$

**Position Update:**
$$x_i(t+1) = x_i(t) + v_i(t+1)$$

Where:
- $v_i(t)$ = velocity of particle $i$ at iteration $t$
- $x_i(t)$ = position of particle $i$ at iteration $t$
- $w$ = inertia weight (typically 0.4-0.9)
- $c_1$ = cognitive parameter (typically 1.5-2.0)
- $c_2$ = social parameter (typically 1.5-2.0)
- $r_1, r_2$ = random numbers in $[0, 1]$
- $p_i$ = personal best position of particle $i$
- $g$ = global best position of swarm

#### Adaptive Inertia Weight

The inertia weight can be adapted over iterations to balance exploration and exploitation:

$$w(t) = w_{max} - \frac{w_{max} - w_{min}}{T_{max}} \cdot t$$

Where:
- $w_{max}$ = maximum inertia weight (e.g., 0.9)
- $w_{min}$ = minimum inertia weight (e.g., 0.4)
- $T_{max}$ = maximum number of iterations
- $t$ = current iteration

**Effect:**
- High $w$ (early): Emphasize exploration (global search)
- Low $w$ (late): Emphasize exploitation (local search)

### Parameter Selection

#### Swarm Size ($N$)

The number of particles affects:
- **Exploration**: More particles → better coverage
- **Computational cost**: More particles → slower
- **Recommendation**: $N = 20-50$ for most problems

#### Cognitive and Social Parameters ($c_1, c_2$)

- **$c_1$ (cognitive)**: Self-confidence
  - High $c_1$ → particles trust their own experience
  - Typical: $c_1 \in [1.5, 2.0]$

- **$c_2$ (social)**: Swarm confidence
  - High $c_2$ → particles trust swarm knowledge
  - Typical: $c_2 \in [1.5, 2.0]$

**Balanced setting**: $c_1 = c_2 = 1.5$ or $c_1 = c_2 = 2.0$

### Convergence Criteria

PSO terminates when:

1. **Maximum iterations** reached
2. **Fitness threshold** achieved: $f(g) < \epsilon$
3. **Stagnation** detected: No improvement for $k$ iterations
4. **Diversity collapse**: Swarm diversity below threshold

### Diversity Measure

Swarm diversity at iteration $t$:

$$D(t) = \frac{1}{N} \sum_{i=1}^{N} \|x_i(t) - \bar{x}(t)\|$$

Where:
- $\bar{x}(t) = \frac{1}{N} \sum_{i=1}^{N} x_i(t)$ = swarm centroid
- High $D$ → exploration phase
- Low $D$ → exploitation phase

### Advantages and Limitations

**Advantages:**
- ✓ Simple implementation
- ✓ Few parameters to tune
- ✓ Effective for continuous optimization
- ✓ Parallelizable
- ✓ Derivative-free

**Limitations:**
- ✗ Can converge prematurely
- ✗ Performance depends on parameters
- ✗ No guarantee of global optimum
- ✗ May struggle with high dimensions

---

## Modern Portfolio Theory

### Overview

Modern Portfolio Theory (MPT), introduced by Harry Markowitz in 1952, provides a mathematical framework for constructing portfolios that optimize expected return for a given level of risk.

### Portfolio Return

The expected return of a portfolio is the weighted average of individual asset returns:

$$E[R_p] = \sum_{i=1}^{n} w_i E[R_i] = \mathbf{w}^T \mathbf{\mu}$$

Where:
- $w_i$ = weight of asset $i$ in portfolio
- $E[R_i]$ = expected return of asset $i$
- $\mathbf{w}$ = weight vector $[w_1, w_2, ..., w_n]^T$
- $\mathbf{\mu}$ = expected return vector $[E[R_1], ..., E[R_n]]^T$

### Portfolio Risk

Portfolio variance (risk) incorporates both individual variances and covariances:

$$\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_{ij} = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$$

Portfolio volatility (standard deviation):

$$\sigma_p = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}$$

Where:
- $\sigma_{ij}$ = covariance between assets $i$ and $j$
- $\mathbf{\Sigma}$ = covariance matrix

### Diversification

**Key insight**: Portfolio risk is less than the weighted average of individual risks due to less-than-perfect correlation.

$$\sigma_p < \sum_{i=1}^{n} w_i \sigma_i \quad \text{(if correlations < 1)}$$

### Constraints

**Basic constraints:**

1. **Full investment**: $\sum_{i=1}^{n} w_i = 1$

2. **Long-only** (no short selling): $w_i \geq 0 \quad \forall i$

3. **Box constraints**: $w_{min} \leq w_i \leq w_{max}$

**Advanced constraints:**

4. **Sector limits**: $\sum_{i \in S_k} w_i \leq L_k$ for sector $k$

5. **Turnover limits**: $\sum_{i=1}^{n} |w_i - w_i^{old}| \leq T$

6. **Cardinality**: Number of non-zero positions $\leq K$

### Efficient Frontier

The efficient frontier is the set of portfolios that:
- Maximize return for a given risk level
- Minimize risk for a given return level

**Parametric form:**

For target return $\mu_t$:

$$
\begin{aligned}
\min_{\mathbf{w}} \quad & \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} \\
\text{subject to} \quad & \mathbf{w}^T \mathbf{\mu} = \mu_t \\
& \sum_{i=1}^{n} w_i = 1 \\
& w_i \geq 0 \quad \forall i
\end{aligned}
$$

### Capital Market Line (CML)

When a risk-free asset is available, the CML represents the best risk-return trade-off:

$$E[R_p] = R_f + \frac{E[R_M] - R_f}{\sigma_M} \sigma_p$$

Where:
- $R_f$ = risk-free rate
- $R_M$ = market portfolio return
- $\sigma_M$ = market portfolio volatility

---

## Sharpe Ratio Optimization

### Definition

The Sharpe ratio measures risk-adjusted return:

$$SR = \frac{E[R_p] - R_f}{\sigma_p} = \frac{\mathbf{w}^T \mathbf{\mu} - R_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}}$$

Where:
- $E[R_p]$ = expected portfolio return
- $R_f$ = risk-free rate
- $\sigma_p$ = portfolio volatility

### Interpretation

- **SR > 1**: Good risk-adjusted returns
- **SR > 2**: Very good risk-adjusted returns
- **SR > 3**: Excellent risk-adjusted returns
- **SR < 0**: Portfolio underperforms risk-free asset

### Optimization Problem

Maximize Sharpe ratio:

$$
\begin{aligned}
\max_{\mathbf{w}} \quad & \frac{\mathbf{w}^T \mathbf{\mu} - R_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}} \\
\text{subject to} \quad & \sum_{i=1}^{n} w_i = 1 \\
& w_i \geq 0 \quad \forall i
\end{aligned}
$$

### PSO Formulation

For PSO (minimization), we use negative Sharpe ratio:

$$f(\mathbf{w}) = -\frac{\mathbf{w}^T \mathbf{\mu} - R_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}}$$

With penalty for constraint violations:

$$f_{penalty}(\mathbf{w}) = f(\mathbf{w}) + \lambda_1 \left|\sum_{i=1}^{n} w_i - 1\right| + \lambda_2 \sum_{i=1}^{n} \max(0, -w_i)$$

### Alternative Risk Measures

**1. Sortino Ratio** (downside risk):

$$\text{Sortino} = \frac{E[R_p] - R_f}{\sigma_{downside}}$$

**2. Calmar Ratio** (maximum drawdown):

$$\text{Calmar} = \frac{E[R_p]}{|\text{MaxDD}|}$$

**3. Information Ratio** (vs benchmark):

$$\text{IR} = \frac{E[R_p] - E[R_b]}{\text{TrackingError}}$$

---

## Convergence Analysis

### PSO Convergence

**Theorem** (Clerc & Kennedy, 2002): PSO converges if constriction factor:

$$\chi = \frac{2}{\left|2 - \phi - \sqrt{\phi^2 - 4\phi}\right|}$$

where $\phi = c_1 + c_2 > 4$.

**Practical convergence**: Monitor convergence metrics:

1. **Fitness improvement**: $\Delta f = f(t-1) - f(t)$
2. **Diversity collapse**: $D(t) < \epsilon_D$
3. **Velocity decay**: $\|\mathbf{v}(t)\| < \epsilon_v$

### Convergence Rate

Expected convergence rate depends on:

- **Problem characteristics**: Modality, separability, dimensionality
- **Parameter settings**: $w, c_1, c_2$
- **Population size**: Larger → better but slower

**Empirical guidelines:**

| Problem Type | Typical Iterations |
|--------------|-------------------|
| Unimodal | 50-100 |
| Multimodal | 100-300 |
| High-dimensional | 300-1000 |

### No Free Lunch Theorem

**Key insight**: No algorithm is universally best for all problems.

**Implication**: 
- PSO excels at continuous, smooth problems
- May struggle with discontinuous or highly constrained problems
- Hybrid approaches can combine strengths

---

## References

### Foundational Papers

1. **Kennedy, J., & Eberhart, R. (1995)**. "Particle swarm optimization." *Proceedings of ICNN'95-International Conference on Neural Networks*, 4, 1942-1948.

2. **Markowitz, H. (1952)**. "Portfolio Selection." *The Journal of Finance*, 7(1), 77-91.

3. **Sharpe, W. F. (1966)**. "Mutual Fund Performance." *The Journal of Business*, 39(1), 119-138.

### Advanced PSO

4. **Clerc, M., & Kennedy, J. (2002)**. "The particle swarm-explosion, stability, and convergence in a multidimensional complex space." *IEEE Transactions on Evolutionary Computation*, 6(1), 58-73.

5. **Shi, Y., & Eberhart, R. (1998)**. "A modified particle swarm optimizer." *IEEE International Conference on Evolutionary Computation*, 69-73.

### Portfolio Optimization

6. **Michaud, R. O. (1989)**. "The Markowitz optimization enigma: Is 'optimized' optimal?" *Financial Analysts Journal*, 45(1), 31-42.

7. **DeMiguel, V., Garlappi, L., & Uppal, R. (2009)**. "Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy?" *The Review of Financial Studies*, 22(5), 1915-1953.

### Books

8. **Engelbrecht, A. P. (2007)**. *Computational Intelligence: An Introduction*. Wiley.

9. **Fabozzi, F. J., Kolm, P. N., Pachamanova, D. A., & Focardi, S. M. (2007)**. *Robust Portfolio Optimization and Management*. Wiley.

10. **Kennedy, J., Eberhart, R. C., & Shi, Y. (2001)**. *Swarm Intelligence*. Morgan Kaufmann.

### Online Resources

- **PSO Visualization**: [https://www.particleswarm.info/](https://www.particleswarm.info/)
- **Portfolio Theory**: [Investopedia](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)
- **Python Implementation**: [scikit-opt](https://scikit-opt.github.io/)

---

## Mathematical Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\mathbf{w}$ | Portfolio weight vector |
| $\mathbf{\mu}$ | Expected return vector |
| $\mathbf{\Sigma}$ | Covariance matrix |
| $\sigma_p$ | Portfolio volatility |
| $R_f$ | Risk-free rate |
| $SR$ | Sharpe ratio |
| $\mathbf{x}_i(t)$ | Position of particle $i$ at time $t$ |
| $\mathbf{v}_i(t)$ | Velocity of particle $i$ at time $t$ |
| $p_i$ | Personal best position |
| $g$ | Global best position |
| $w$ | Inertia weight |
| $c_1, c_2$ | Acceleration coefficients |

---

## Appendix: Proof Sketches

### A. PSO Convergence Proof (Simplified)

**Claim**: PSO converges to a fixed point under certain conditions.

**Proof outline**:
1. Consider expected values: $E[\mathbf{v}(t+1)]$ and $E[\mathbf{x}(t+1)]$
2. At equilibrium: $E[\mathbf{v}(t+1)] = 0$
3. This implies: $\mathbf{x}^* = \frac{c_1 p_i + c_2 g}{c_1 + c_2}$
4. With proper parameter selection, $\{\mathbf{x}(t)\}$ converges

### B. Efficient Frontier Characterization

**Claim**: The efficient frontier is a hyperbola in mean-variance space.

**Proof outline**:
1. Consider Lagrangian of optimization problem
2. First-order conditions yield: $\mathbf{w}^* = \lambda \mathbf{\Sigma}^{-1}(\mathbf{\mu} - R_f \mathbf{1})$
3. Substituting back: $\sigma_p^2 = a\mu_p^2 + b\mu_p + c$
4. This is a conic section (hyperbola)

---

*This documentation provides the theoretical foundation for understanding and applying PSO to portfolio optimization problems.*
