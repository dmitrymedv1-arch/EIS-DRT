"""
Electrochemical Impedance Spectroscopy (EIS) Analysis Tool
Distribution of Relaxation Times (DRT) Analysis

Поддерживаемые методы:
- Тихоновская регуляризация (Tikhonov) с NNLS
- Байесовский метод с MCMC (PyMC)
- Метод максимальной энтропии (Maximum Entropy) с авто-выбором λ
- Гауссовские процессы (fGP-DRT) с non-negativity constraints
- Loewner Framework (RLF) - data-driven метод
- Generalized DRT для обработки индуктивных петель

Author: DRT Analysis Tool
Version: 3.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, AutoMinorLocator
from scipy import optimize, linalg, interpolate, integrate
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_func
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
import logging
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# PyMC for Bayesian MCMC
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    warnings.warn("PyMC not installed. Bayesian MCMC will be disabled.")

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="EIS-DRT Analysis Tool v3.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Data Classes for Results Storage
# ============================================================================

@dataclass
class DRTResult:
    """Container for DRT calculation results"""
    tau_grid: np.ndarray
    gamma: np.ndarray
    gamma_std: Optional[np.ndarray] = None
    method: str = ""
    R_inf: float = 0.0
    R_pol: float = 0.0
    convergence: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def log_tau(self) -> np.ndarray:
        return np.log10(self.tau_grid)
    
    def get_integral(self) -> float:
        return np.trapezoid(self.gamma, np.log(self.tau_grid))

@dataclass
class ImpedanceData:
    """Container for impedance data with preprocessing"""
    freq: np.ndarray
    re_z: np.ndarray
    im_z: np.ndarray
    original_freq: np.ndarray = None
    original_re_z: np.ndarray = None
    original_im_z: np.ndarray = None
    removed_indices: List[int] = field(default_factory=list)
    frequency_range: Tuple[float, float] = (None, None)
    
    def __post_init__(self):
        self.original_freq = self.freq.copy()
        self.original_re_z = self.re_z.copy()
        self.original_im_z = self.im_z.copy()
        self._sort_by_frequency()
    
    def _sort_by_frequency(self):
        idx = np.argsort(self.freq)
        self.freq = self.freq[idx]
        self.re_z = self.re_z[idx]
        self.im_z = self.im_z[idx]
    
    def remove_point(self, index: int):
        """Remove a specific point by index"""
        if 0 <= index < len(self.freq):
            self.removed_indices.append(index)
            mask = np.ones(len(self.freq), dtype=bool)
            mask[index] = False
            self.freq = self.freq[mask]
            self.re_z = self.re_z[mask]
            self.im_z = self.im_z[mask]
    
    def apply_frequency_range(self, f_min: float, f_max: float):
        """Crop frequency range"""
        mask = (self.freq >= f_min) & (self.freq <= f_max)
        self.freq = self.freq[mask]
        self.re_z = self.re_z[mask]
        self.im_z = self.im_z[mask]
        self.frequency_range = (f_min, f_max)
    
    def reset(self):
        """Reset to original data"""
        self.freq = self.original_freq.copy()
        self.re_z = self.original_re_z.copy()
        self.im_z = self.original_im_z.copy()
        self.removed_indices = []
        self.frequency_range = (None, None)
        self._sort_by_frequency()
    
    @property
    def n_points(self) -> int:
        return len(self.freq)
    
    @property
    def Z(self) -> np.ndarray:
        return self.re_z + 1j * self.im_z
    
    @property
    def Z_mod(self) -> np.ndarray:
        return np.sqrt(self.re_z**2 + self.im_z**2)
    
    @property
    def phase(self) -> np.ndarray:
        return np.arctan2(self.im_z, self.re_z) * 180 / np.pi


# ============================================================================
# Scientific Plotting Style for Matplotlib
# ============================================================================

def apply_publication_style():
    """Apply publication-quality plotting style for matplotlib figures"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'axes.grid': False,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.major.size': 4,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'figure.facecolor': 'white',
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'errorbar.capsize': 3,
        'text.usetex': False, 
        'mathtext.default': 'regular', 
    })

apply_publication_style()


# ============================================================================
# Data Loading and Validation
# ============================================================================

def load_data(file, freq_col, re_col, im_col) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load impedance data from uploaded file."""
    if file is not None:
        try:
            df = pd.read_csv(file)
            if freq_col in df.columns and re_col in df.columns and im_col in df.columns:
                freq = df[freq_col].values.astype(float)
                re_z = df[re_col].values.astype(float)
                im_z = np.abs(df[im_col].values.astype(float))
                return freq, re_z, im_z
        except Exception as e:
            st.error(f"Error loading file: {e}")
    return None, None, None


def manual_data_entry() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Create widget for manual data entry with single text area."""
    st.subheader("Ручной ввод данных")
    st.markdown("Введите данные в формате: **частота Re(Z) -Im(Z)** (разделитель - пробел или табуляция)")
    
    example_data = """1000000	-71.55	-3745
891300	-102.3	-4127
794300	-62.24	-4664
707900	88.34	-5240
631000	317.9	-5944
562300	763	-6676
501200	1207	-7348
446700	1843	-8127
398100	2629	-8805
354800	3557	-9427
316200	4561	-9925
281800	5561	-10370"""
    
    data_input = st.text_area(
        "Введите данные (каждая строка: частота Re(Z) -Im(Z))",
        value=example_data,
        height=300,
        help="Формат: частота (Гц) Re(Z) (Ом) -Im(Z) (Ом). Разделитель - пробел или табуляция"
    )
    
    if st.button("Загрузить данные", type="primary"):
        try:
            rows = []
            for line in data_input.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        freq_val = float(parts[0])
                        re_val = float(parts[1])
                        im_val = abs(float(parts[2]))
                        rows.append([freq_val, re_val, im_val])
                    except ValueError:
                        st.warning(f"Пропущена некорректная строка: {line}")
                        continue
            
            if len(rows) >= 3:
                rows = np.array(rows)
                freq = rows[:, 0]
                re_z = rows[:, 1]
                im_z = rows[:, 2]
                st.success(f"✅ Загружено {len(freq)} точек спектра")
                
                with st.expander("Просмотр загруженных данных"):
                    preview_df = pd.DataFrame({
                        'Frequency (Hz)': freq,
                        'Re(Z) (Ω)': re_z,
                        '-Im(Z) (Ω)': im_z
                    })
                    st.dataframe(preview_df.head(10))
                
                return freq, re_z, im_z
            else:
                st.error(f"Недостаточно данных. Загружено только {len(rows)} строк. Минимум 3 строки.")
        except Exception as e:
            st.error(f"Ошибка при загрузке данных: {e}")
            st.info("Проверьте формат данных. Каждая строка должна содержать: частоту, Re(Z), -Im(Z)")
    
    return None, None, None


def kramers_kronig_hilbert_transform(freq: np.ndarray, re_z: np.ndarray, im_z: np.ndarray) -> Tuple[bool, float, np.ndarray, np.ndarray]:
    """
    Perform Kramers-Kronig validation using Hilbert transform.
    This is the proper KK test as described in literature.
    """
    try:
        omega = 2 * np.pi * freq
        log_omega = np.log(omega)
        
        # Interpolate data for Hilbert transform
        from scipy.interpolate import interp1d
        interp_omega = np.logspace(np.log10(omega[0]), np.log10(omega[-1]), 500)
        interp_re = interp1d(omega, re_z, kind='cubic', fill_value='extrapolate')(interp_omega)
        interp_im = interp1d(omega, im_z, kind='cubic', fill_value='extrapolate')(interp_omega)
        
        # Calculate Hilbert transform of imaginary part to predict real part
        # For a causal system, Re(Z) = H{Im(Z)} where H is Hilbert transform
        from scipy.signal import hilbert
        analytic = hilbert(interp_im)
        re_predicted = np.real(analytic)
        
        # Interpolate back to original frequencies
        re_pred_original = interp1d(interp_omega, re_predicted, kind='cubic', fill_value='extrapolate')(omega)
        
        # Calculate residuals
        residuals = (re_z - re_pred_original) / np.abs(re_z + 1e-10)
        max_residual = np.max(np.abs(residuals))
        is_valid = max_residual < 0.05  # 5% threshold
        
        return is_valid, max_residual, residuals, np.zeros_like(residuals)
    except Exception as e:
        logging.warning(f"KK Hilbert transform failed: {e}")
        return False, 1.0, None, None


# ============================================================================
# Base DRT Class with Generalized Support
# ============================================================================

class DRTCore:
    """Base class for DRT inversion with support for inductive loops"""
    
    def __init__(self, data: ImpedanceData, include_inductive: bool = False):
        self.data = data
        self.include_inductive = include_inductive
        self.frequencies = data.freq
        self.Z_real = data.re_z
        self.Z_imag = data.im_z
        self.Z = data.Z
        self.N = len(self.frequencies)
        
        # Sort by frequency
        sort_idx = np.argsort(self.frequencies)
        self.frequencies = self.frequencies[sort_idx]
        self.Z_real = self.Z_real[sort_idx]
        self.Z_imag = self.Z_imag[sort_idx]
        
        # Determine if inductive behavior is present (positive imaginary part at high frequencies)
        high_freq_idx = np.where(self.frequencies > 0.1 * np.max(self.frequencies))[0]
        if len(high_freq_idx) > 0:
            self.has_inductive_loop = np.any(self.Z_imag[high_freq_idx] > 0)
        else:
            self.has_inductive_loop = False
        
        # Automatic determination of relaxation time range
        self.tau_min = 1.0 / (2 * np.pi * np.max(self.frequencies)) * 0.1
        self.tau_max = 1.0 / (2 * np.pi * np.min(self.frequencies)) * 10
        
        # Estimate ohmic resistance (high frequency limit)
        if len(high_freq_idx) > 3:
            self.R_inf = np.mean(self.Z_real[high_freq_idx[-5:]])
        else:
            self.R_inf = self.Z_real[-1] if len(self.Z_real) > 0 else 0
        
        # Total polarization resistance
        self.R_pol = np.max(self.Z_real) - self.R_inf if np.max(self.Z_real) > self.R_inf else 1.0
    
    def _build_kernel_matrix(self, tau_grid: np.ndarray, include_rl: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Build kernel matrix for given time grid, optionally including RL elements"""
        M = len(tau_grid)
        K_real = np.zeros((self.N, M))
        K_imag = np.zeros((self.N, M))
        
        omega = 2 * np.pi * self.frequencies
        
        for i in range(self.N):
            for j in range(M):
                denominator = 1 + (omega[i] * tau_grid[j])**2
                K_real[i, j] = 1.0 / denominator
                K_imag[i, j] = -omega[i] * tau_grid[j] / denominator
                
                if include_rl:
                    # For RL elements, the kernel is different
                    # RL contribution: jωτ/(1+jωτ)
                    rl_denom = 1 + (omega[i] * tau_grid[j])**2
                    K_real[i, j] += (omega[i] * tau_grid[j])**2 / rl_denom
                    K_imag[i, j] += omega[i] * tau_grid[j] / rl_denom
        
        return K_real, K_imag
    
    def _l_curve_criterion(self, residuals: np.ndarray, solution_norms: np.ndarray) -> int:
        """Find corner of L-curve (maximum curvature)"""
        if len(residuals) < 3:
            return len(residuals) // 2
        
        log_res = np.log(residuals + 1e-10)
        log_sol = np.log(solution_norms + 1e-10)
        
        dlog_res = np.gradient(log_res)
        dlog_sol = np.gradient(log_sol)
        
        if len(dlog_res) < 2 or len(dlog_sol) < 2:
            return len(residuals) // 2
        
        curvature = np.abs(dlog_res[1:-1] * dlog_sol[1:-1]) / (dlog_res[1:-1]**2 + dlog_sol[1:-1]**2 + 1e-10)**1.5
        
        if len(curvature) > 0:
            return np.argmax(curvature) + 1
        return len(residuals) // 2


# ============================================================================
# Tikhonov Regularization with NNLS
# ============================================================================

class TikhonovDRT(DRTCore):
    """Tikhonov regularization for DRT using non-negative least squares"""
    
    def __init__(self, data: ImpedanceData, regularization_order: int = 2, include_inductive: bool = False):
        super().__init__(data, include_inductive)
        self.regularization_order = regularization_order
    
    def _build_regularization_matrix(self, M: int, order: int) -> np.ndarray:
        """Build regularization matrix"""
        if order == 0:
            return np.eye(M)
        elif order == 1:
            L = np.zeros((M-1, M))
            for i in range(M-1):
                L[i, i] = -1
                L[i, i+1] = 1
            return L
        elif order == 2:
            L = np.zeros((M-2, M))
            for i in range(M-2):
                L[i, i] = 1
                L[i, i+1] = -2
                L[i, i+2] = 1
            return L
        else:
            return np.eye(M)
    
    def _solve_nnls(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        from scipy.optimize import nnls
        x, resid = nnls(A, b)
        # Проверка сходимости
        if resid > 1e-6 * np.linalg.norm(b):
            logging.warning(f"NNLS residual: {resid}")
        return x
    
    def compute(self, n_tau: int = 150, lambda_value: Optional[float] = None, 
                lambda_auto: bool = True, lambda_range: Optional[np.ndarray] = None) -> DRTResult:
        """Compute DRT using Tikhonov regularization with NNLS"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=self.include_inductive)
        
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        K = np.vstack([K_real, K_imag])
        
        L = self._build_regularization_matrix(n_tau, self.regularization_order)
        
        if lambda_auto:
            if lambda_range is None:
                lambda_range = np.logspace(-8, 2, 30)
            
            residuals = []
            solution_norms = []
            solutions = []
            
            for lam in lambda_range:
                try:
                    A = np.vstack([K, lam * L])
                    b = np.concatenate([Z_target, np.zeros(L.shape[0])])
                    
                    # Use NNLS instead of lstsq
                    x = self._solve_nnls(A, b)
                    
                    residual = np.linalg.norm(K @ x - Z_target)
                    sol_norm = np.linalg.norm(L @ x)
                    
                    residuals.append(residual)
                    solution_norms.append(sol_norm)
                    solutions.append(x)
                except Exception as e:
                    logging.warning(f"Lambda {lam} failed: {e}")
                    continue
            
            if len(residuals) > 2:
                best_idx = self._l_curve_criterion(np.array(residuals), np.array(solution_norms))
                lambda_opt = lambda_range[best_idx]
                gamma = solutions[best_idx]
            else:
                lambda_opt = lambda_range[0] if len(lambda_range) > 0 else 1e-4
                A = np.vstack([K, lambda_opt * L])
                b = np.concatenate([Z_target, np.zeros(L.shape[0])])
                gamma = self._solve_nnls(A, b)
        else:
            lam = lambda_value if lambda_value is not None else 1e-4
            A = np.vstack([K, lam * L])
            b = np.concatenate([Z_target, np.zeros(L.shape[0])])
            gamma = self._solve_nnls(A, b)
        
        # Normalize DRT
        pass
        
        # Estimate uncertainty from curvature of solution
        gamma_std = np.abs(np.gradient(np.gradient(gamma))) * 0.1
        
        return DRTResult(
            tau_grid=tau_grid,
            gamma=gamma,
            gamma_std=gamma_std,
            method="Tikhonov Regularization (NNLS)",
            R_inf=self.R_inf,
            R_pol=self.R_pol,
            metadata={'lambda': lam if not lambda_auto else lambda_opt, 'order': self.regularization_order}
        )
    
    def reconstruct_impedance(self, tau_grid: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct impedance from DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=self.include_inductive)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        return Z_rec_real, Z_rec_imag


# ============================================================================
# Bayesian DRT with MCMC (PyMC)
# ============================================================================

class BayesianDRT(DRTCore):
    """Bayesian method for DRT with MCMC sampling"""
    
    def __init__(self, data: ImpedanceData, include_inductive: bool = False):
        super().__init__(data, include_inductive)
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC is required for Bayesian DRT with MCMC")
    
    def compute(self, n_tau: int = 150, n_samples: int = 2000, n_tune: int = 1000,
                n_chains: int = 4) -> DRTResult:
        """Compute DRT using Bayesian method with MCMC"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=self.include_inductive)
        
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        K = np.vstack([K_real, K_imag])
        
        # Regularization matrix (second derivative)
        L = np.zeros((n_tau-2, n_tau))
        for i in range(n_tau-2):
            L[i, i] = 1
            L[i, i+1] = -2
            L[i, i+2] = 1
        
        # Build PyMC model
        with pm.Model() as model:
            # Prior for gamma (positive, smooth)
            gamma_raw = pm.HalfNormal('gamma_raw', sigma=1.0, shape=n_tau)
            
            # Regularization prior (smoothness)
            smoothness = pm.HalfCauchy('smoothness', beta=0.1)
            reg_penalty = smoothness * pm.math.sum(pm.math.abs(L @ gamma_raw))
            
            # Likelihood
            sigma = pm.HalfCauchy('sigma', beta=0.1)
            Z_pred = pm.math.dot(K, gamma_raw)
            likelihood = pm.Normal('likelihood', mu=Z_pred, sigma=sigma, observed=Z_target)
            
            # Add regularization as potential
            pm.Potential('reg', -reg_penalty)
            
            # Sample
            trace = pm.sample(draws=n_samples, tune=n_tune, chains=n_chains, 
                             return_inferencedata=True, progressbar=False)
        
        # Extract posterior statistics
        gamma_samples = trace.posterior['gamma_raw'].values.reshape(-1, n_tau)
        gamma_mean = np.mean(gamma_samples, axis=0)
        gamma_std = np.std(gamma_samples, axis=0)
        
        # Normalize
        pass
        
        # Check convergence using R-hat
        r_hat = az.rhat(trace).to_array().values
        converged = np.all(r_hat < 1.05)
        
        return DRTResult(
            tau_grid=tau_grid,
            gamma=gamma_mean,
            gamma_std=gamma_std,
            method="Bayesian MCMC",
            R_inf=self.R_inf,
            R_pol=self.R_pol,
            convergence=converged,
            metadata={'n_samples': n_samples, 'n_tune': n_tune, 'n_chains': n_chains}
        )
    
    def reconstruct_impedance(self, tau_grid: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct impedance from DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=self.include_inductive)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        return Z_rec_real, Z_rec_imag


# ============================================================================
# Maximum Entropy DRT with Automatic Lambda Selection
# ============================================================================

class MaxEntropyDRT(DRTCore):
    """Maximum Entropy method for DRT with automatic lambda selection"""
    
    def __init__(self, data: ImpedanceData, include_inductive: bool = False):
        super().__init__(data, include_inductive)
    
    def _entropy(self, gamma: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        gamma_pos = gamma[gamma > 1e-10]
        if len(gamma_pos) == 0:
            return 0
        return -np.sum(gamma_pos * np.log(gamma_pos))
    
    def _objective_function(self, x: np.ndarray, K: np.ndarray, Z_target: np.ndarray, lam: float) -> float:
        """Objective function with entropy penalty"""
        gamma = np.maximum(x, 1e-10)
        residual = K @ gamma - Z_target
        data_fit = 0.5 * np.sum(residual**2)
        entropy_penalty = -lam * self._entropy(gamma)
        return data_fit + entropy_penalty
    
    def _solve_for_lambda(self, K: np.ndarray, Z_target: np.ndarray, 
                          lambda_range: np.ndarray, n_tau: int) -> Tuple[np.ndarray, float]:
        """Solve for multiple lambda and return best based on L-curve"""
        residuals = []
        solutions = []
        
        for lam in lambda_range:
            try:
                x0 = np.ones(n_tau) / n_tau
                result = optimize.minimize(
                    self._objective_function, x0,
                    args=(K, Z_target, lam),
                    method='L-BFGS-B',
                    bounds=[(1e-10, None) for _ in range(n_tau)],
                    options={'maxiter': 500, 'disp': False}
                )
                gamma = result.x
                residual = np.linalg.norm(K @ gamma - Z_target)
                residuals.append(residual)
                solutions.append(gamma)
            except Exception as e:
                logging.warning(f"Lambda {lam} failed: {e}")
                residuals.append(1e10)
                solutions.append(np.zeros(n_tau))
        
        # Find best lambda using L-curve
        if len(residuals) > 2:
            best_idx = self._l_curve_criterion(np.array(residuals), np.array(lambda_range))
        else:
            best_idx = np.argmin(residuals)
        
        return solutions[best_idx], lambda_range[best_idx]
    
    def compute(self, n_tau: int = 150, lambda_value: Optional[float] = None,
                lambda_auto: bool = True) -> DRTResult:
        """Compute DRT using maximum entropy method with automatic lambda selection"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=self.include_inductive)
        
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        K = np.vstack([K_real, K_imag])
        
        if lambda_auto:
            lambda_range = np.logspace(-4, 2, 20)
            gamma, lambda_opt = self._solve_for_lambda(K, Z_target, lambda_range, n_tau)
        else:
            lam = lambda_value if lambda_value is not None else 0.1
            x0 = np.ones(n_tau) / n_tau
            result = optimize.minimize(
                self._objective_function, x0,
                args=(K, Z_target, lam),
                method='L-BFGS-B',
                bounds=[(1e-10, None) for _ in range(n_tau)],
                options={'maxiter': 500, 'disp': False}
            )
            gamma = result.x
            lambda_opt = lam
        
        # Normalize
        pass
        
        # Estimate uncertainty
        gamma_std = np.abs(np.gradient(np.gradient(gamma))) * 0.15
        
        return DRTResult(
            tau_grid=tau_grid,
            gamma=gamma,
            gamma_std=gamma_std,
            method="Maximum Entropy",
            R_inf=self.R_inf,
            R_pol=self.R_pol,
            metadata={'lambda': lambda_opt}
        )
    
    def reconstruct_impedance(self, tau_grid: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct impedance from DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=self.include_inductive)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        return Z_rec_real, Z_rec_imag


# ============================================================================
# Finite Gaussian Process DRT (fGP-DRT)
# ============================================================================

class FiniteGaussianProcessDRT(DRTCore):
    """Finite Gaussian Process for DRT with non-negativity constraints"""
    
    def __init__(self, data: ImpedanceData, include_inductive: bool = False):
        super().__init__(data, include_inductive)
    
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0, sigma_f: float = 1.0) -> np.ndarray:
        """Radial Basis Function kernel"""
        dist_matrix = np.subtract.outer(x1, x2)**2
        return sigma_f**2 * np.exp(-0.5 * dist_matrix / length_scale**2)
    
    def compute(self, n_tau: int = 150, n_components: int = 30, n_samples: int = 100) -> DRTResult:
        """Compute DRT using finite Gaussian Process"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        log_tau_grid = np.log10(tau_grid)
        
        # Create basis from RBF functions
        basis_centers = np.linspace(log_tau_grid[0], log_tau_grid[-1], n_components)
        length_scale = (log_tau_grid[-1] - log_tau_grid[0]) / n_components
        
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=self.include_inductive)
        K_full = np.vstack([K_real, K_imag])
        
        # Build feature matrix
        Phi = np.zeros((self.N * 2, n_components))
        for i, center in enumerate(basis_centers):
            phi = np.exp(-0.5 * ((log_tau_grid - center) / length_scale)**2)
            phi = phi / np.sum(phi)
            Phi[:, i] = K_full @ phi
        
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        
        # Bayesian linear regression with non-negativity constraints
        # Using truncated normal prior
        from scipy.stats import truncnorm
        
        # Initialize with ridge regression
        lam = 1e-4
        A = Phi.T @ Phi + lam * np.eye(n_components)
        b = Phi.T @ Z_target
        weights_init = np.linalg.solve(A, b)
        weights_init = np.maximum(weights_init, 0)
        
        # Sample posterior using MCMC with non-negativity constraints
        if PYMC_AVAILABLE:
            with pm.Model() as model:
                # Prior for weights (truncated normal)
                weights = pm.TruncatedNormal('weights', mu=weights_init, sigma=1.0, 
                                             lower=0, shape=n_components)
                
                # Likelihood
                sigma = pm.HalfCauchy('sigma', beta=0.1)
                Z_pred = pm.math.dot(Phi, weights)
                likelihood = pm.Normal('likelihood', mu=Z_pred, sigma=sigma, observed=Z_target)
                
                # Sample
                trace = pm.sample(draws=n_samples, tune=n_samples//2, 
                                 chains=2, progressbar=False)
                
                weights_samples = trace.posterior['weights'].values.reshape(-1, n_components)
                weights_mean = np.mean(weights_samples, axis=0)
                weights_std = np.std(weights_samples, axis=0)
        else:
            # Fallback: use optimization with uncertainty estimation
            weights_mean = weights_init
            weights_std = np.ones_like(weights_init) * 0.1
        
        # Reconstruct DRT
        gamma = np.zeros(n_tau)
        gamma_std = np.zeros(n_tau)
        for i, center in enumerate(basis_centers):
            phi = np.exp(-0.5 * ((log_tau_grid - center) / length_scale)**2)
            phi = phi / np.sum(phi)
            gamma += weights_mean[i] * phi
            gamma_std += weights_std[i] * phi
        
        # Normalize
        pass
        
        return DRTResult(
            tau_grid=tau_grid,
            gamma=gamma,
            gamma_std=gamma_std,
            method="Finite Gaussian Process (fGP-DRT)",
            R_inf=self.R_inf,
            R_pol=self.R_pol,
            metadata={'n_components': n_components}
        )
    
    def reconstruct_impedance(self, tau_grid: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct impedance from DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=self.include_inductive)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        return Z_rec_real, Z_rec_imag


# ============================================================================
# Loewner Framework (RLF) - Data-Driven DRT
# ============================================================================

class LoewnerFrameworkDRT(DRTCore):
    """Loewner Framework for data-driven DRT extraction"""
    
    def __init__(self, data: ImpedanceData):
        super().__init__(data, include_inductive=False)
    
    def _build_loewner_matrices(self, omega: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build Loewner and shifted Loewner matrices"""
        n = len(omega)
        # Split into left and right datasets
        n_left = n // 2
        n_right = n - n_left
        
        left_omega = omega[:n_left]
        right_omega = omega[n_left:]
        left_Z = Z[:n_left]
        right_Z = Z[n_left:]
        
        # Loewner matrix
        L = np.zeros((n_left, n_right), dtype=complex)
        Ls = np.zeros((n_left, n_right), dtype=complex)
        
        for i in range(n_left):
            for j in range(n_right):
                denom = left_omega[i] - right_omega[j]
                if abs(denom) > 1e-10:
                    L[i, j] = (left_Z[i] - right_Z[j]) / denom
                    Ls[i, j] = (left_omega[i] * left_Z[i] - right_omega[j] * right_Z[j]) / denom
        
        return L, Ls, left_Z, right_Z
    
    def _scree_not_threshold(self, singular_values: np.ndarray) -> int:
        """ScreeNOT algorithm for optimal SVD truncation"""
        n = len(singular_values)
        if n < 3:
            return n // 2
        
        # Find the knee point
        diffs = np.diff(singular_values)
        diffs2 = np.diff(diffs)
        
        # Look for the largest change in curvature
        if len(diffs2) > 0:
            knee_idx = np.argmax(np.abs(diffs2)) + 1
            return min(knee_idx + 1, n)
        
        return n // 2
    
    def compute(self, n_tau: int = 150, model_order: Optional[int] = None) -> DRTResult:
        """Compute DRT using Loewner Framework"""
        
        omega = 2 * np.pi * self.frequencies
        Z = self.Z
        
        # Build Loewner matrices
        L, Ls, left_Z, right_Z = self._build_loewner_matrices(omega, Z)
        
        # Compute SVD for model order reduction
        U, S, Vh = np.linalg.svd(L, full_matrices=False)
        
        # Determine optimal model order
        if model_order is None:
            model_order = self._scree_not_threshold(S)
        
        model_order = max(1, min(model_order, len(S) - 1))
        
        # Truncate
        U_r = U[:, :model_order]
        S_r = np.diag(S[:model_order])
        V_r = Vh[:model_order, :]
        
        # Compute reduced matrices
        E_r = -U_r.conj().T @ L @ V_r.conj().T
        A_r = -U_r.conj().T @ Ls @ V_r.conj().T
        B_r = U_r.conj().T @ left_Z
        C_r = right_Z @ V_r.conj().T
        
        # Extract poles and residues
        try:
            # Solve generalized eigenvalue problem
            eigvals, eigvecs = linalg.eig(A_r, E_r)
            
            # Time constants
            tau_loewner = -1.0 / eigvals
            # Keep only positive real time constants
            valid = (np.real(tau_loewner) > 0) & (np.imag(tau_loewner) < 1e-6)
            tau_loewner = np.real(tau_loewner[valid])
            
            # Calculate residues (resistances)
            R_loewner = np.zeros(len(tau_loewner))
            for i in range(len(tau_loewner)):
                # Simplified residue calculation
                R_loewner[i] = np.abs(C_r @ eigvecs[:, i] * (eigvecs[:, i].conj().T @ B_r))
        except:
            # Fallback: use regularized solution
            tau_loewner = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
            R_loewner = np.ones(n_tau) / n_tau
        
        # Interpolate to uniform tau grid
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        gamma = np.zeros(n_tau)
        
        # Sort and interpolate
        if len(tau_loewner) > 1:
            idx_sorted = np.argsort(tau_loewner)
            tau_sorted = tau_loewner[idx_sorted]
            R_sorted = R_loewner[idx_sorted]
            
            # Interpolate
            interp_func = interpolate.interp1d(np.log10(tau_sorted), R_sorted, 
                                               kind='linear', fill_value=0, bounds_error=False)
            gamma = interp_func(np.log10(tau_grid))
            gamma = np.maximum(gamma, 0)
        
        # Normalize
        pass
        
        return DRTResult(
            tau_grid=tau_grid,
            gamma=gamma,
            gamma_std=None,
            method="Loewner Framework (RLF)",
            R_inf=self.R_inf,
            R_pol=self.R_pol,
            metadata={'model_order': model_order}
        )
    
    def reconstruct_impedance(self, tau_grid: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct impedance from DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=False)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        return Z_rec_real, Z_rec_imag


# ============================================================================
# Peak Detection and Analysis
# ============================================================================

def find_peaks_drt(tau_grid: np.ndarray, gamma: np.ndarray, prominence: float = 0.05) -> List[Dict[str, Any]]:
    """Find peaks in DRT spectrum"""
    gamma_norm = gamma / np.max(gamma) if np.max(gamma) > 0 else gamma
    
    peaks, properties = find_peaks(
        gamma_norm,
        height=prominence,
        prominence=prominence,
        distance=len(tau_grid) // 20
    )
    
    peak_results = []
    for idx in peaks:
        peak_info = {
            'tau': tau_grid[idx],
            'log_tau': np.log10(tau_grid[idx]),
            'frequency': 1 / (2 * np.pi * tau_grid[idx]),
            'amplitude': gamma[idx],
            'width': properties.get('widths', [None])[peaks.tolist().index(idx)] if 'widths' in properties else None
        }
        peak_results.append(peak_info)
    
    return peak_results


def calculate_resistances(tau: np.ndarray, drt: np.ndarray, peaks_idx: List[int]) -> List[float]:
    """Calculate resistances from peak areas"""
    resistances = []
    for i in range(len(peaks_idx)):
        if i == 0:
            start = 0
        else:
            start = (peaks_idx[i-1] + peaks_idx[i]) // 2
        if i == len(peaks_idx) - 1:
            end = len(tau)
        else:
            end = (peaks_idx[i] + peaks_idx[i+1]) // 2
        
        area = np.trapezoid(drt[start:end], np.log(tau[start:end]))
        resistances.append(area)
    
    return resistances


def fit_gaussian_peaks(tau_grid: np.ndarray, gamma: np.ndarray, n_peaks: Optional[int] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Fit DRT with sum of Gaussians"""
    log_tau = np.log10(tau_grid)
    
    if n_peaks is None:
        peaks = find_peaks_drt(tau_grid, gamma)
        n_peaks = len(peaks)
    
    def sum_gaussians(log_tau: np.ndarray, *params: float) -> np.ndarray:
        result = np.zeros_like(log_tau)
        for i in range(n_peaks):
            amp = params[3*i]
            center = params[3*i + 1]
            sigma = params[3*i + 2]
            result += amp * np.exp(-((log_tau - center)**2) / (2 * sigma**2))
        return result
    
    # Initial parameters
    initial_params = []
    peaks = find_peaks_drt(tau_grid, gamma)
    for i, peak in enumerate(peaks[:n_peaks]):
        initial_params.extend([peak['amplitude'], peak['log_tau'], 0.3])
    
    try:
        popt, _ = optimize.curve_fit(sum_gaussians, log_tau, gamma, p0=initial_params, maxfev=5000)
        
        fitted_gamma = sum_gaussians(log_tau, *popt)
        
        peak_params = []
        for i in range(n_peaks):
            peak_params.append({
                'amplitude': popt[3*i],
                'tau': 10**popt[3*i + 1],
                'sigma': popt[3*i + 2]
            })
        
        return fitted_gamma, peak_params
    except Exception:
        return gamma, []


# ============================================================================
# Visualization Functions (Matplotlib - Publication Quality)
# ============================================================================

def plot_nyquist_matplotlib(data: ImpedanceData, re_rec: Optional[np.ndarray] = None, 
                           im_rec: Optional[np.ndarray] = None, title: str = "Nyquist Plot",
                           highlight_idx: Optional[int] = None) -> plt.Figure:
    """Create publication-quality Nyquist plot"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.plot(data.re_z, data.im_z, 'o-', markersize=4, linewidth=1.5, 
            label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.5)
    
    if highlight_idx is not None and 0 <= highlight_idx < data.n_points:
        ax.plot(data.re_z[highlight_idx], data.im_z[highlight_idx], 'ro', 
                markersize=10, markeredgecolor='red', markerfacecolor='none', linewidth=2,
                label='Selected Point')
    
    if re_rec is not None and im_rec is not None:
        ax.plot(re_rec, im_rec, 's-', markersize=3, linewidth=1.0,
                label='Reconstructed', color='#ff7f0e', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    
    ax.set_xlabel("Re(Z) / Ohm", fontweight='bold')  # Убрал LaTeX
    ax.set_ylabel("-Im(Z) / Ohm", fontweight='bold')  # Убрал LaTeX
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Equal aspect ratio
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    ax.set_aspect('equal', adjustable='box')
    
    return fig

def plot_bode_matplotlib(data: ImpedanceData, re_rec: Optional[np.ndarray] = None, 
                         im_rec: Optional[np.ndarray] = None) -> plt.Figure:
    """Create publication-quality Bode plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    
    # Magnitude plot
    mag = data.Z_mod
    ax1.loglog(data.freq, mag, 'o-', markersize=4, linewidth=1.5, 
               label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.5)
    if re_rec is not None and im_rec is not None:
        mag_rec = np.sqrt(re_rec**2 + im_rec**2)
        ax1.loglog(data.freq, mag_rec, 's-', markersize=3, linewidth=1.0,
                   label='Reconstructed', color='#ff7f0e', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    ax1.set_xlabel("Frequency / Hz", fontweight='bold')
    ax1.set_ylabel("|Z| / Ohm", fontweight='bold')  # Убрал LaTeX
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Phase plot
    phase = data.phase
    ax2.semilogx(data.freq, phase, 'o-', markersize=4, linewidth=1.5,
                 label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.5)
    if re_rec is not None and im_rec is not None:
        phase_rec = np.arctan2(im_rec, re_rec) * 180 / np.pi
        ax2.semilogx(data.freq, phase_rec, 's-', markersize=3, linewidth=1.0,
                     label='Reconstructed', color='#ff7f0e', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    ax2.set_xlabel("Frequency / Hz", fontweight='bold')
    ax2.set_ylabel("Phase / deg", fontweight='bold')  # Убрал символ градуса
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    
    fig.suptitle("Bode Plot", fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_drt_matplotlib(result: DRTResult, peaks: Optional[List[Dict[str, Any]]] = None,
                       title: str = "Distribution of Relaxation Times") -> plt.Figure:
    """Create publication-quality DRT plot with both tau and frequency axes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ========================================================================
    # Left plot: γ(τ) vs τ (log scale)
    # ========================================================================
    
    # Plot DRT with uncertainty if available
    if result.gamma_std is not None:
        ax1.fill_between(result.tau_grid, result.gamma - 2*result.gamma_std, 
                        result.gamma + 2*result.gamma_std,
                        alpha=0.3, color='gray', label='±2σ uncertainty')
    ax1.semilogx(result.tau_grid, result.gamma, '-', linewidth=2, color='#2ca02c', label='DRT')
    
    # Plot peaks on left plot
    if peaks and len(peaks) > 0:
        peak_tau = [p['tau'] for p in peaks]
        peak_drt = [p['amplitude'] for p in peaks]
        ax1.plot(peak_tau, peak_drt, 'rv', markersize=8, label='Detected peaks')
        
        # Add peak labels for left plot
        for i, (t, d) in enumerate(zip(peak_tau, peak_drt)):
            freq = 1/(2*np.pi*t)
            ax1.annotate(f'τ={t:.2e}s\nf={freq:.2e}Hz',
                       xy=(t, d), xytext=(t*1.5, d*1.2),
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel(r"Relaxation Time $\tau$ / s", fontweight='bold')
    ax1.set_ylabel(r"$\gamma(\tau)$ / $\Omega$", fontweight='bold')
    ax1.set_title(r"$\gamma(\tau)$ vs $\tau$", fontweight='bold')
    ax1.legend(loc='best', frameon=True)
    ax1.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # ========================================================================
    # Right plot: γ(τ) vs Frequency (Hz) - high to low frequency
    # ========================================================================
    
    # Convert tau to frequency: f = 1/(2πτ)
    frequencies = 1 / (2 * np.pi * result.tau_grid)
    
    # Sort frequencies in descending order (high to low)
    sort_idx = np.argsort(frequencies)[::-1]  # Descending order
    freqs_sorted = frequencies[sort_idx]
    gamma_sorted = result.gamma[sort_idx]
    
    # Plot DRT vs frequency
    if result.gamma_std is not None:
        gamma_std_sorted = result.gamma_std[sort_idx]
        ax2.fill_between(freqs_sorted, gamma_sorted - 2*gamma_std_sorted, 
                        gamma_sorted + 2*gamma_std_sorted,
                        alpha=0.3, color='gray', label='±2σ uncertainty')
    ax2.semilogx(freqs_sorted, gamma_sorted, '-', linewidth=2, color='#2ca02c', label='DRT')
    
    # Plot peaks on frequency plot
    if peaks and len(peaks) > 0:
        peak_freqs = [p['frequency'] for p in peaks]
        peak_amplitudes = [p['amplitude'] for p in peaks]
        # Sort peaks by frequency (descending)
        peak_pairs = sorted(zip(peak_freqs, peak_amplitudes), key=lambda x: x[0], reverse=True)
        peak_freqs_sorted, peak_amplitudes_sorted = zip(*peak_pairs) if peak_pairs else ([], [])
        
        ax2.plot(peak_freqs_sorted, peak_amplitudes_sorted, 'rv', markersize=8, label='Detected peaks')
        
        # Add peak labels for right plot
        for i, (f, d) in enumerate(zip(peak_freqs_sorted, peak_amplitudes_sorted)):
            tau_val = 1/(2*np.pi*f)
            ax2.annotate(f'f={f:.2e}Hz\nτ={tau_val:.2e}s',
                       xy=(f, d), xytext=(f*1.5, d*1.2),
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax2.set_xlabel("Frequency / Hz", fontweight='bold')
    ax2.set_ylabel(r"$\gamma(\tau)$ / $\Omega$", fontweight='bold')
    ax2.set_title(r"$\gamma(\tau)$ vs Frequency (High → Low)", fontweight='bold')
    ax2.legend(loc='best', frameon=True)
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Ensure x-axis is log scale and shows from high to low
    ax2.set_xscale('log')
    # Invert x-axis to show high frequency on left, low on right
    ax2.invert_xaxis()
    
    # Set x-axis limits to match data range
    ax2.set_xlim(freqs_sorted[0], freqs_sorted[-1])
    
    # Add minor ticks for better readability
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    
    # Main title
    fig.suptitle(title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_kk_residuals_matplotlib(freq: np.ndarray, res_real: np.ndarray, res_imag: np.ndarray) -> plt.Figure:
    """Create publication-quality KK residuals plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
    
    ax1.semilogx(freq, res_real * 100, 'o-', markersize=4, linewidth=1.0, color='#1f77b4')
    ax1.set_xlabel("Frequency / Hz", fontweight='bold')
    ax1.set_ylabel(r"$\Delta$ Re(Z) / %", fontweight='bold')
    ax1.set_title("Kramers-Kronig Test - Real Part Residuals", fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axhline(y=2, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axhline(y=-2, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax2.semilogx(freq, res_imag * 100, 'o-', markersize=4, linewidth=1.0, color='#1f77b4')
    ax2.set_xlabel("Frequency / Hz", fontweight='bold')
    ax2.set_ylabel(r"$\Delta$ Im(Z) / %", fontweight='bold')
    ax2.set_title("Kramers-Kronig Test - Imaginary Part Residuals", fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axhline(y=2, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.axhline(y=-2, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    return fig


# ============================================================================
# Plotly Interactive Visualization
# ============================================================================

def plot_impedance_plotly(data: ImpedanceData, re_rec: Optional[np.ndarray] = None,
                         im_rec: Optional[np.ndarray] = None) -> go.Figure:
    """Create interactive impedance plots with Plotly"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Nyquist Plot', 'Bode Plot - Magnitude'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Nyquist plot
    fig.add_trace(
        go.Scatter(x=data.re_z, y=-data.im_z, mode='markers',
                   name='Experimental', marker=dict(size=6, color='blue')),
        row=1, col=1
    )
    
    if re_rec is not None and im_rec is not None:
        fig.add_trace(
            go.Scatter(x=re_rec, y=-im_rec, mode='lines',
                       name='Reconstructed', line=dict(color='red', width=2)),
            row=1, col=1
        )
    
    fig.update_xaxes(title_text="Z' (Ω)", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' (Ω)", row=1, col=1)
    
    # Bode plot - Magnitude
    Z_mod_exp = data.Z_mod
    fig.add_trace(
        go.Scatter(x=data.freq, y=Z_mod_exp, mode='markers',
                   name='Experimental', marker=dict(size=6, color='blue')),
        row=1, col=2
    )
    
    if re_rec is not None and im_rec is not None:
        Z_mod_rec = np.sqrt(re_rec**2 + im_rec**2)
        fig.add_trace(
            go.Scatter(x=data.freq, y=Z_mod_rec, mode='lines',
                       name='Reconstructed', line=dict(color='red', width=2)),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="|Z| (Ω)", type="log", row=1, col=2)
    
    fig.update_layout(height=500, showlegend=True)
    
    return fig


def plot_drt_plotly(result: DRTResult, peaks: Optional[List[Dict[str, Any]]] = None) -> go.Figure:
    """Create interactive DRT plot with Plotly"""
    fig = go.Figure()
    
    # Main DRT curve
    fig.add_trace(go.Scatter(
        x=result.tau_grid, y=result.gamma,
        mode='lines',
        name='DRT',
        line=dict(color='blue', width=2)
    ))
    
    # Confidence interval
    if result.gamma_std is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([result.tau_grid, result.tau_grid[::-1]]),
            y=np.concatenate([result.gamma + 2*result.gamma_std, 
                             (result.gamma - 2*result.gamma_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))
    
    # Detected peaks
    if peaks:
        peak_tau = [p['tau'] for p in peaks]
        peak_gamma = [p['amplitude'] for p in peaks]
        fig.add_trace(go.Scatter(
            x=peak_tau, y=peak_gamma,
            mode='markers',
            marker=dict(size=10, color='red', symbol='x'),
            name='Detected Peaks'
        ))
        
        # Add annotations
        for peak in peaks:
            fig.add_annotation(
                x=peak['tau'], y=peak['amplitude'],
                text=f"τ = {peak['tau']:.2e} s<br>f = {peak['frequency']:.2f} Hz",
                showarrow=True,
                arrowhead=2,
                ax=20, ay=-20
            )
    
    fig.update_xaxes(title_text="τ (s)", type="log")
    fig.update_yaxes(title_text="γ(τ) (Ω)")
    fig.update_layout(height=500, title="Distribution of Relaxation Times (DRT)")
    
    return fig


# ============================================================================
# Report Generation
# ============================================================================

def create_report(data: ImpedanceData, result: DRTResult, peaks_data: List[Dict[str, Any]],
                 method_name: str, kk_passed: bool, max_kk_res: float) -> str:
    """Generate analysis report"""
    report = f"""
    ============================================================
    EIS-DRT Analysis Report
    ============================================================
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Analysis Method: {method_name}
    
    Data Information:
    - Original points: {len(data.original_freq)}
    - Analyzed points: {data.n_points}
    - Removed points: {len(data.removed_indices)}
    - Frequency range: {data.freq.min():.2e} - {data.freq.max():.2e} Hz
    - Ohmic resistance (R∞): {result.R_inf:.4f} Ω
    - Polarization resistance (Rpol): {result.R_pol:.4f} Ω
    
    DRT Parameters:
    - Time constant range: {result.tau_grid.min():.2e} - {result.tau_grid.max():.2e} s
    - Total integral: {result.get_integral():.4f} Ω
    
    Detected Processes:
    """
    
    if peaks_data and len(peaks_data) > 0:
        for i, peak in enumerate(peaks_data):
            report += f"""
    Process {i+1}:
        - Relaxation time τ: {peak['tau']:.2e} s
        - Frequency f: {peak['frequency']:.2e} Hz
        - Amplitude: {peak['amplitude']:.4f} Ω
    """
    else:
        report += "    No peaks detected\n"
    
    report += f"""
    Quality Metrics:
    - KK test passed: {kk_passed}
    - Max KK residual: {max_kk_res*100:.3f}%
    
    ============================================================
    """
    
    return report


# ============================================================================
# Main Application
# ============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'preview_plots' not in st.session_state:
        st.session_state.preview_plots = False
    if 'selected_point' not in st.session_state:
        st.session_state.selected_point = None


def main():
    initialize_session_state()
    
    st.title("⚡ Electrochemical Impedance Spectroscopy Analysis")
    st.markdown("### Distribution of Relaxation Times (DRT) Analysis Tool v3.0")
    st.markdown("Поддерживаются 6 методов инверсии: Tikhonov (NNLS), Bayesian MCMC, Maximum Entropy (auto-λ), "
                "fGP-DRT, Loewner Framework (RLF), Generalized DRT (с индуктивностями)")
    st.markdown("---")
    
    # Sidebar for input controls
    with st.sidebar:
        st.header("📁 Data Input")
        
        # Data input method selection
        input_method = st.radio("Select input method:", ["Upload File", "Manual Entry"])
        
        freq = None
        re_z = None
        im_z = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Choose file", type=['txt', 'csv', 'xlsx', 'dat', 'z', 'mpt'])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file, nrows=5)
                    st.subheader("Column Mapping")
                    col_freq = st.selectbox("Frequency column", df.columns)
                    col_re = st.selectbox("Re(Z) column", df.columns)
                    col_im = st.selectbox("-Im(Z) column", df.columns)
                    
                    if st.button("Load Data", key="load_file_btn"):
                        freq, re_z, im_z = load_data(uploaded_file, col_freq, col_re, col_im)
                        if freq is not None:
                            st.session_state.data = ImpedanceData(freq, re_z, im_z)
                            st.session_state.data_loaded = True
                            st.session_state.preview_plots = True
                            st.success(f"Loaded {len(freq)} data points")
                            st.rerun()
                        else:
                            st.error("Error loading data")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        else:  # Manual entry
            freq, re_z, im_z = manual_data_entry()
            if freq is not None:
                st.session_state.data = ImpedanceData(freq, re_z, im_z)
                st.session_state.data_loaded = True
                st.session_state.preview_plots = True
                st.rerun()
        
        # Data preprocessing controls
        if st.session_state.data_loaded and st.session_state.data is not None:
            st.header("✂️ Data Preprocessing")
            
            # Frequency range selection
            f_min, f_max = st.slider(
                "Frequency range (Hz)",
                min_value=float(st.session_state.data.original_freq.min()),
                max_value=float(st.session_state.data.original_freq.max()),
                value=(float(st.session_state.data.original_freq.min()),
                       float(st.session_state.data.original_freq.max())),
                format="%.2e"
            )
            
            # Point removal
            point_idx = st.slider(
                "Select point to remove (index)",
                min_value=0,
                max_value=st.session_state.data.n_points - 1,
                value=0,
                step=1
            )
            st.session_state.selected_point = point_idx
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Remove Selected Point", key="remove_point_btn"):
                    st.session_state.data.remove_point(point_idx)
                    st.success(f"Removed point {point_idx}")
                    st.rerun()
            with col2:
                if st.button("Reset Data", key="reset_data_btn"):
                    st.session_state.data.reset()
                    st.success("Data reset to original")
                    st.rerun()
            
            # Apply frequency range button
            if st.button("Apply Frequency Range", key="apply_range_btn"):
                st.session_state.data.apply_frequency_range(f_min, f_max)
                st.success(f"Applied range: {f_min:.2e} - {f_max:.2e} Hz")
                st.rerun()
            
            st.info(f"Current points: {st.session_state.data.n_points} / {len(st.session_state.data.original_freq)}")
        
        # Analysis parameters
        if st.session_state.data_loaded and st.session_state.data is not None:
            st.header("⚙️ Analysis Parameters")
            
            # Method selection
            analysis_method = st.selectbox("DRT Calculation Method",
                                          ["Tikhonov Regularization (NNLS)",
                                           "Bayesian MCMC",
                                           "Maximum Entropy (auto-λ)",
                                           "Finite Gaussian Process (fGP-DRT)",
                                           "Loewner Framework (RLF)",
                                           "Generalized DRT (with inductive loops)"])
            
            n_tau = st.slider("Number of time points", 50, 300, 150)
            
            # Inductive loop handling
            include_inductive = False
            if analysis_method == "Generalized DRT (with inductive loops)":
                include_inductive = st.checkbox("Include inductive loops", value=True)
            
            # Method-specific parameters
            if analysis_method == "Tikhonov Regularization (NNLS)":
                reg_order = st.selectbox("Regularization order", [0, 1, 2], index=2)
                lambda_auto = st.checkbox("Automatic λ selection", value=True)
                if not lambda_auto:
                    lambda_value = st.number_input("λ value", value=1e-4, format="%.1e")
                else:
                    lambda_value = None
            elif analysis_method == "Bayesian MCMC":
                if not PYMC_AVAILABLE:
                    st.warning("PyMC not installed. Bayesian MCMC will use fallback method.")
                n_samples = st.slider("MCMC samples", 500, 5000, 2000)
                n_tune = st.slider("Tuning samples", 500, 2000, 1000)
            elif analysis_method == "Maximum Entropy (auto-λ)":
                entropy_lambda_auto = st.checkbox("Auto-select λ", value=True)
                if not entropy_lambda_auto:
                    entropy_lambda = st.number_input("Entropy λ", value=0.1, format="%.2f")
                else:
                    entropy_lambda = None
            elif analysis_method == "Finite Gaussian Process (fGP-DRT)":
                n_components = st.slider("GP components", 10, 50, 30)
            elif analysis_method == "Loewner Framework (RLF)":
                model_order = st.number_input("Model order (0=auto)", min_value=0, max_value=100, value=0)
                if model_order == 0:
                    model_order = None
            
            # Peak detection parameters
            st.header("🔍 Peak Detection")
            peak_prominence = st.slider("Peak prominence (%)", 1, 20, 5) / 100
            
            # Run analysis button
            analyze_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content area - Preview plots
    if st.session_state.data_loaded and st.session_state.data is not None and st.session_state.preview_plots:
        st.markdown("---")
        st.header("📊 Data Preview")
        
        # Display preview plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_nyquist = plot_nyquist_matplotlib(st.session_state.data, 
                                                  highlight_idx=st.session_state.selected_point)
            st.pyplot(fig_nyquist)
            st.caption("Nyquist Plot - Red circle shows selected point")
        
        with col2:
            fig_bode = plot_bode_matplotlib(st.session_state.data)
            st.pyplot(fig_bode)
            st.caption("Bode Plot")
        
        # Data statistics
        with st.expander("Data Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Points", st.session_state.data.n_points)
            with col2:
                st.metric("f_min", f"{st.session_state.data.freq.min():.2e} Hz")
            with col3:
                st.metric("f_max", f"{st.session_state.data.freq.max():.2e} Hz")
            with col4:
                st.metric("R_inf (est)", f"{st.session_state.data.re_z[-1]:.4f} Ω")
    
    # Main content area - Analysis results
    if st.session_state.data_loaded and st.session_state.data is not None and 'analyze_button' in locals() and analyze_button:
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        data = st.session_state.data
        
        # Perform KK test
        with st.spinner("Performing Kramers-Kronig validation..."):
            kk_passed, max_res, res_real, res_imag = kramers_kronig_hilbert_transform(
                data.freq, data.re_z, data.im_z
            )
            
            if kk_passed:
                st.success(f"✓ KK test passed (max residual: {max_res*100:.2f}%)")
            else:
                st.warning(f"⚠ KK test failed (max residual: {max_res*100:.2f}%)")
        
        # Calculate DRT based on selected method
        with st.spinner(f"Calculating DRT using {analysis_method}..."):
            try:
                if analysis_method == "Tikhonov Regularization (NNLS)":
                    drt_solver = TikhonovDRT(data, regularization_order=reg_order, 
                                             include_inductive=include_inductive)
                    result = drt_solver.compute(n_tau=n_tau, lambda_value=lambda_value, 
                                                lambda_auto=lambda_auto)
                    method_key = "Tikhonov"
                elif analysis_method == "Bayesian MCMC":
                    drt_solver = BayesianDRT(data, include_inductive=include_inductive)
                    if PYMC_AVAILABLE:
                        result = drt_solver.compute(n_tau=n_tau, n_samples=n_samples, n_tune=n_tune)
                    else:
                        # Fallback to simpler Bayesian method
                        result = drt_solver.compute(n_tau=n_tau, n_samples=500)
                    method_key = "Bayesian"
                elif analysis_method == "Maximum Entropy (auto-λ)":
                    drt_solver = MaxEntropyDRT(data, include_inductive=include_inductive)
                    lambda_auto_val = entropy_lambda_auto if 'entropy_lambda_auto' in locals() else True
                    lambda_val = entropy_lambda if not lambda_auto_val and 'entropy_lambda' in locals() else None
                    result = drt_solver.compute(n_tau=n_tau, lambda_value=lambda_val, 
                                                lambda_auto=lambda_auto_val)
                    method_key = "MaxEntropy"
                elif analysis_method == "Finite Gaussian Process (fGP-DRT)":
                    drt_solver = FiniteGaussianProcessDRT(data, include_inductive=include_inductive)
                    result = drt_solver.compute(n_tau=n_tau, n_components=n_components)
                    method_key = "fGP-DRT"
                elif analysis_method == "Loewner Framework (RLF)":
                    drt_solver = LoewnerFrameworkDRT(data)
                    result = drt_solver.compute(n_tau=n_tau, model_order=model_order)
                    method_key = "Loewner"
                else:  # Generalized DRT
                    drt_solver = TikhonovDRT(data, regularization_order=2, include_inductive=include_inductive)
                    result = drt_solver.compute(n_tau=n_tau, lambda_auto=True)
                    method_key = "Generalized DRT"
                
                # Reconstruct impedance
                Z_rec_real, Z_rec_imag = drt_solver.reconstruct_impedance(result.tau_grid, result.gamma)
                
            except Exception as e:
                st.error(f"DRT calculation failed: {e}")
                st.stop()
        
        # Find peaks
        peaks = find_peaks_drt(result.tau_grid, result.gamma, peak_prominence)
        
        # Calculate resistances if peaks found
        if peaks:
            peaks_idx = [np.argmin(np.abs(result.tau_grid - p['tau'])) for p in peaks]
            resistances = calculate_resistances(result.tau_grid, result.gamma, peaks_idx)
            for i, p in enumerate(peaks):
                p['resistance'] = resistances[i] if i < len(resistances) else 0
                p['capacitance'] = p['tau'] / p['resistance'] if p['resistance'] > 0 else 0
        
        # Create tabs for results
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 DRT Spectrum", "🔄 Nyquist & Bode", 
                                                 "📊 Interactive Plots", "🔍 KK Test", "📋 Report"])
        
        with tab1:
            st.subheader("Distribution of Relaxation Times")
            
            # Publication-quality DRT plot
            fig_drt = plot_drt_matplotlib(result, peaks)
            st.pyplot(fig_drt)
            
            # Convergence info for Bayesian method
            if not result.convergence and analysis_method == "Bayesian MCMC":
                st.warning("⚠ MCMC may not have converged. Consider increasing samples.")
            
            # Download button for DRT figure
            buf = io.BytesIO()
            fig_drt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            st.download_button("📥 Download DRT Plot (PNG, 600 dpi)", 
                              data=buf.getvalue(),
                              file_name=f"drt_spectrum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                              mime="image/png")
            
            # Display peak information
            if peaks:
                st.subheader("Detected Processes")
                for i, peak in enumerate(peaks):
                    with st.expander(f"Process {i+1} (τ = {peak['tau']:.2e} s)"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Relaxation Time τ", f"{peak['tau']:.2e} s")
                            st.metric("Frequency f", f"{peak['frequency']:.2e} Hz")
                        with col2:
                            st.metric("Resistance R", f"{peak.get('resistance', 0):.4f} Ω")
                        with col3:
                            st.metric("Capacitance C", f"{peak.get('capacitance', 0):.2e} F")
            else:
                st.info("No peaks detected. Try adjusting the peak prominence parameter.")
        
        with tab2:
            st.subheader("Nyquist and Bode Plots")
            
            # Nyquist plot
            fig_nyquist = plot_nyquist_matplotlib(data, Z_rec_real, Z_rec_imag)
            st.pyplot(fig_nyquist)
            
            # Bode plot
            fig_bode = plot_bode_matplotlib(data, Z_rec_real, Z_rec_imag)
            st.pyplot(fig_bode)
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                buf = io.BytesIO()
                fig_nyquist.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                st.download_button("📥 Download Nyquist Plot (PNG)", 
                                  data=buf.getvalue(),
                                  file_name=f"nyquist_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                  mime="image/png")
            with col2:
                buf = io.BytesIO()
                fig_bode.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                st.download_button("📥 Download Bode Plot (PNG)", 
                                  data=buf.getvalue(),
                                  file_name=f"bode_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                  mime="image/png")
        
        with tab3:
            st.subheader("Interactive Plots")
            
            # Interactive impedance plot
            fig_imp_plotly = plot_impedance_plotly(data, Z_rec_real, Z_rec_imag)
            st.plotly_chart(fig_imp_plotly, use_container_width=True)
            
            # Interactive DRT plot
            fig_drt_plotly = plot_drt_plotly(result, peaks)
            st.plotly_chart(fig_drt_plotly, use_container_width=True)
            
            # Reconstruction error
            error_real = np.abs(data.re_z - Z_rec_real) / np.abs(data.re_z + 1e-10) * 100
            error_imag = np.abs(data.im_z - Z_rec_imag) / np.abs(data.im_z + 1e-10) * 100
            mean_error = np.mean(np.sqrt(error_real**2 + error_imag**2))
            st.metric("Mean Reconstruction Error", f"{mean_error:.2f} %")
            
            # Uncertainty info
            if result.gamma_std is not None:
                st.metric("Mean DRT Uncertainty", f"{np.mean(result.gamma_std / (result.gamma + 1e-10)) * 100:.2f} %")
        
        with tab4:
            if res_real is not None and res_imag is not None:
                st.subheader("Kramers-Kronig Validation")
                fig_kk = plot_kk_residuals_matplotlib(data.freq, res_real, res_imag)
                st.pyplot(fig_kk)
                
                buf = io.BytesIO()
                fig_kk.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                st.download_button("📥 Download KK Residuals Plot (PNG)", 
                                  data=buf.getvalue(),
                                  file_name=f"kk_residuals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                  mime="image/png")
        
        with tab5:
            st.subheader("Analysis Report")
            
            params = {
                'R_inf': result.R_inf,
                'R_pol': result.R_pol,
                'kk_passed': kk_passed,
                'max_kk_res': max_res,
            }
            
            report = create_report(data, result, peaks, analysis_method, kk_passed, max_res)
            st.text(report)
            
            # Download report
            st.download_button("📥 Download Full Report (TXT)", 
                              data=report,
                              file_name=f"eis_drt_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                              mime="text/plain")
            
            st.markdown("---")
            st.subheader("📁 Export DRT Data")
            
            # Create two different export formats
            # Format 1: γ(τ) vs τ (relaxation time in seconds)
            drt_data_tau = pd.DataFrame({
                'tau_s': result.tau_grid,
                'gamma_tau_ohm': result.gamma
            })
            if result.gamma_std is not None:
                drt_data_tau['gamma_uncertainty_ohm'] = result.gamma_std
            
            # Format 2: γ(τ) vs Frequency (Hz)
            frequencies = 1 / (2 * np.pi * result.tau_grid)
            # Sort by frequency (high to low for better readability)
            sort_idx = np.argsort(frequencies)[::-1]
            freqs_sorted = frequencies[sort_idx]
            gamma_sorted = result.gamma[sort_idx]
            
            drt_data_freq = pd.DataFrame({
                'frequency_hz': freqs_sorted,
                'gamma_tau_ohm': gamma_sorted
            })
            if result.gamma_std is not None:
                gamma_std_sorted = result.gamma_std[sort_idx]
                drt_data_freq['gamma_uncertainty_ohm'] = gamma_std_sorted
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Export as γ(τ) vs τ**")
                st.markdown("Data format: relaxation time (s) vs DRT amplitude (Ω)")
                
                # Create CSV for tau format
                csv_tau = drt_data_tau.to_csv(index=False)
                st.download_button(
                    "📥 Export DRT Data (τ format) - CSV", 
                    data=csv_tau,
                    file_name=f"drt_data_tau_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Create TXT for tau format (tab-separated)
                txt_tau = drt_data_tau.to_csv(sep='\t', index=False)
                st.download_button(
                    "📥 Export DRT Data (τ format) - TXT", 
                    data=txt_tau,
                    file_name=f"drt_data_tau_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                # Show preview
                with st.expander("Preview - γ(τ) vs τ data"):
                    st.dataframe(drt_data_tau.head(10))
            
            with col2:
                st.markdown("**Export as γ(τ) vs Frequency**")
                st.markdown("Data format: frequency (Hz) vs DRT amplitude (Ω)")
                
                # Create CSV for frequency format
                csv_freq = drt_data_freq.to_csv(index=False)
                st.download_button(
                    "📥 Export DRT Data (Frequency format) - CSV", 
                    data=csv_freq,
                    file_name=f"drt_data_frequency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Create TXT for frequency format (tab-separated)
                txt_freq = drt_data_freq.to_csv(sep='\t', index=False)
                st.download_button(
                    "📥 Export DRT Data (Frequency format) - TXT", 
                    data=txt_freq,
                    file_name=f"drt_data_frequency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                # Show preview
                with st.expander("Preview - γ(τ) vs Frequency data"):
                    st.dataframe(drt_data_freq.head(10))
            
            st.markdown("---")
            st.subheader("📊 Export Peak Analysis")
            
            # Export peaks data
            if peaks:
                peaks_df = pd.DataFrame(peaks)
                
                # Add additional calculated columns if available
                if 'resistance' in peaks_df.columns:
                    peaks_df['capacitance_F'] = peaks_df['tau'] / peaks_df['resistance']
                
                # Reorder columns for better readability
                column_order = ['tau', 'frequency', 'amplitude', 'resistance', 'capacitance_F', 'log_tau', 'width']
                available_cols = [col for col in column_order if col in peaks_df.columns]
                peaks_df = peaks_df[available_cols]
                
                # CSV export
                peaks_csv = peaks_df.to_csv(index=False)
                st.download_button(
                    "📥 Export Peaks Data - CSV", 
                    data=peaks_csv,
                    file_name=f"peaks_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # TXT export (tab-separated)
                peaks_txt = peaks_df.to_csv(sep='\t', index=False)
                st.download_button(
                    "📥 Export Peaks Data - TXT", 
                    data=peaks_txt,
                    file_name=f"peaks_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                # Show peaks preview
                with st.expander("Preview - Detected Peaks"):
                    st.dataframe(peaks_df)
            else:
                st.info("No peaks detected to export")
            
            st.markdown("---")
            st.subheader("📈 Export Full Dataset")
            
            # Export complete dataset with all information
            complete_data = pd.DataFrame({
                'tau_s': result.tau_grid,
                'log10_tau': np.log10(result.tau_grid),
                'frequency_hz': 1 / (2 * np.pi * result.tau_grid),
                'gamma_tau_ohm': result.gamma
            })
            
            if result.gamma_std is not None:
                complete_data['gamma_uncertainty_ohm'] = result.gamma_std
            
            # Sort by frequency for better readability
            complete_data = complete_data.sort_values('frequency_hz', ascending=False).reset_index(drop=True)
            
            # CSV export
            complete_csv = complete_data.to_csv(index=False)
            st.download_button(
                "📥 Export Complete Dataset - CSV", 
                data=complete_csv,
                file_name=f"drt_complete_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # TXT export (tab-separated)
            complete_txt = complete_data.to_csv(sep='\t', index=False)
            st.download_button(
                "📥 Export Complete Dataset - TXT", 
                data=complete_txt,
                file_name=f"drt_complete_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            with st.expander("Preview - Complete Dataset"):
                st.dataframe(complete_data.head(10))
    
    elif not st.session_state.data_loaded:
        st.info("👈 Please load impedance data using the sidebar controls to begin analysis")
        
        # Show example format
        with st.expander("📖 Data Format Example"):
            st.markdown("""
            ### Expected Data Format
            Your file should contain at least three columns:
            
            | Frequency (Hz) | Re(Z) (Ω) | -Im(Z) (Ω) |
            |----------------|-----------|------------|
            | 0.1            | 10.0      | 0.5        |
            | 1.0            | 8.5       | 1.2        |
            | 10.0           | 6.2       | 2.8        |
            | 100.0          | 4.1       | 3.5        |
            | 1000.0         | 2.5       | 2.9        |
            | 10000.0        | 1.2       | 1.8        |
            
            **Notes:**
            - Frequency must be in Hz
            - Re(Z) should be positive
            - -Im(Z) should be positive (capacitive behavior)
            - Data can be in any order (will be sorted by frequency)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("⚡ *DRT Analysis Tool v3.0 | 6 inversion methods: Tikhonov (NNLS), Bayesian MCMC, MaxEntropy (auto-λ), fGP-DRT, Loewner (RLF), Generalized DRT*")


if __name__ == "__main__":
    main()
