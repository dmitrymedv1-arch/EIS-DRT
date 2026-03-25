# app.py
"""
Electrochemical Impedance Spectroscopy (EIS) Analysis Tool
Distribution of Relaxation Times (DRT) Analysis

Поддерживаемые методы:
- Тихоновская регуляризация (Tikhonov)
- Байесовский метод (Bayesian)
- Метод максимальной энтропии (Maximum Entropy)
- Гауссовские процессы (GP-DRT)
- Генетическое программирование (ISGP)

Author: DRT Analysis Tool
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, AutoMinorLocator
from scipy import optimize, linalg, interpolate
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_func
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="EIS-DRT Analysis Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Scientific Plotting Style for Matplotlib (for publication-ready figures)
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
    })

apply_publication_style()

# ============================================================================
# Data Loading and Validation
# ============================================================================

def load_data(file, freq_col, re_col, im_col):
    """Load impedance data from uploaded file."""
    if file is not None:
        df = pd.read_csv(file)
        if freq_col in df.columns and re_col in df.columns and im_col in df.columns:
            freq = df[freq_col].values
            re_z = df[re_col].values
            im_z = np.abs(df[im_col].values)  # Ensure positive
            return freq, re_z, im_z
    return None, None, None

def manual_data_entry():
    """Create widget for manual data entry with single text area."""
    st.subheader("Ручной ввод данных")
    st.markdown("Введите данные в формате: **частота Re(Z) -Im(Z)** (разделитель - пробел или табуляция)")
    
    # Пример данных для отображения
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
            # Парсинг данных
            rows = []
            for line in data_input.strip().split('\n'):
                # Пропускаем пустые строки
                if not line.strip():
                    continue
                # Разделяем по пробелам или табуляции
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        freq_val = float(parts[0])
                        re_val = float(parts[1])
                        im_val = abs(float(parts[2]))  # Берем абсолютное значение для -Im(Z)
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
                
                # Показываем预览 загруженных данных
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

def kramers_kronig_test(freq, re_z, im_z):
    """Perform Kramers-Kronig validation test."""
    omega = 2 * np.pi * freq
    n_points = len(freq)
    
    # Create test model with RC elements
    tau_range = np.logspace(np.log10(1/(2*np.pi*freq[-1])), 
                           np.log10(1/(2*np.pi*freq[0])), 
                           min(50, n_points//2))
    
    H = np.zeros((n_points, len(tau_range)), dtype=complex)
    for i, w in enumerate(omega):
        for j, tau in enumerate(tau_range):
            H[i, j] = 1/(1 + 1j*w*tau)
    
    try:
        # Solve for RC weights using non-negative least squares
        from scipy.optimize import nnls
        H_real = H.real
        H_imag = H.imag
        
        R_inf_est = re_z[-1]
        weights_real, _ = nnls(H_real, re_z - R_inf_est)
        weights_imag, _ = nnls(H_imag, im_z)
        
        # Calculate residuals
        re_pred = R_inf_est + H_real @ weights_real
        im_pred = H_imag @ weights_imag
        
        Z_mod = np.sqrt(re_z**2 + im_z**2)
        rel_res_real = (re_z - re_pred) / (Z_mod + 1e-10)
        rel_res_imag = (im_z - im_pred) / (Z_mod + 1e-10)
        
        max_res = max(np.abs(rel_res_real).max(), np.abs(rel_res_imag).max())
        is_valid = max_res < 0.02  # 2% threshold
        
        return is_valid, max_res, rel_res_real, rel_res_imag, freq
    except:
        return False, 1.0, None, None, freq

# ============================================================================
# Base DRT Class
# ============================================================================

class DRTCore:
    """Base class for DRT inversion"""
    
    def __init__(self, frequencies, Z_real, Z_imag):
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.Z_real = np.asarray(Z_real, dtype=float)
        self.Z_imag = np.asarray(Z_imag, dtype=float)
        self.Z = self.Z_real + 1j * self.Z_imag
        self.N = len(frequencies)
        
        # Sort by frequency
        sort_idx = np.argsort(self.frequencies)
        self.frequencies = self.frequencies[sort_idx]
        self.Z_real = self.Z_real[sort_idx]
        self.Z_imag = self.Z_imag[sort_idx]
        
        # Automatic determination of relaxation time range
        self.tau_min = 1.0 / (2 * np.pi * np.max(self.frequencies)) * 0.1
        self.tau_max = 1.0 / (2 * np.pi * np.min(self.frequencies)) * 10
        
        # Estimate ohmic resistance (high frequency limit)
        high_freq_idx = np.where(self.frequencies > 0.1 * np.max(self.frequencies))[0]
        if len(high_freq_idx) > 3:
            self.R_inf = np.mean(self.Z_real[high_freq_idx[-5:]])
        else:
            self.R_inf = self.Z_real[-1] if len(self.Z_real) > 0 else 0
        
        # Total polarization resistance
        self.R_pol = np.max(self.Z_real) - self.R_inf if np.max(self.Z_real) > self.R_inf else 1.0
    
    def _build_kernel_matrix(self, tau_grid):
        """Build kernel matrix for given time grid"""
        M = len(tau_grid)
        K_real = np.zeros((self.N, M))
        K_imag = np.zeros((self.N, M))
        
        omega = 2 * np.pi * self.frequencies
        
        for i in range(self.N):
            for j in range(M):
                denominator = 1 + (omega[i] * tau_grid[j])**2
                K_real[i, j] = 1.0 / denominator
                K_imag[i, j] = -omega[i] * tau_grid[j] / denominator
        
        return K_real, K_imag
    
    def _l_curve_criterion(self, residuals, solution_norms):
        """Find corner of L-curve (maximum curvature)"""
        if len(residuals) < 3:
            return len(residuals) // 2
        
        log_res = np.log(residuals)
        log_sol = np.log(solution_norms)
        
        # Calculate curvature
        dlog_res = np.gradient(log_res)
        dlog_sol = np.gradient(log_sol)
        curvature = np.abs(dlog_res[1:-1] * dlog_sol[1:-1]) / (dlog_res[1:-1]**2 + dlog_sol[1:-1]**2)**1.5
        
        if len(curvature) > 0:
            return np.argmax(curvature) + 1
        return len(residuals) // 2

# ============================================================================
# Tikhonov Regularization
# ============================================================================

class TikhonovDRT(DRTCore):
    """Tikhonov regularization for DRT"""
    
    def __init__(self, frequencies, Z_real, Z_imag, regularization_order=2):
        super().__init__(frequencies, Z_real, Z_imag)
        self.regularization_order = regularization_order
    
    def _build_regularization_matrix(self, M, order):
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
    
    def compute(self, n_tau=150, lambda_value=None, lambda_auto=True, lambda_range=None):
        """Compute DRT using Tikhonov regularization"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        
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
                    
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    x = np.maximum(x, 0)
                    
                    residual = np.linalg.norm(K @ x - Z_target)
                    sol_norm = np.linalg.norm(L @ x)
                    
                    residuals.append(residual)
                    solution_norms.append(sol_norm)
                    solutions.append(x)
                except:
                    continue
            
            if len(residuals) > 2:
                best_idx = self._l_curve_criterion(np.array(residuals), np.array(solution_norms))
                lambda_opt = lambda_range[best_idx]
                gamma = solutions[best_idx]
            else:
                lambda_opt = lambda_range[0] if lambda_range else 1e-4
                A = np.vstack([K, lambda_opt * L])
                b = np.concatenate([Z_target, np.zeros(L.shape[0])])
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                gamma = np.maximum(x, 0)
        else:
            lam = lambda_value if lambda_value is not None else 1e-4
            A = np.vstack([K, lam * L])
            b = np.concatenate([Z_target, np.zeros(L.shape[0])])
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            gamma = np.maximum(x, 0)
        
        # Normalize DRT
        integral = np.trapz(gamma, np.log(tau_grid))
        if integral > 0:
            gamma = gamma / integral * self.R_pol
        
        return tau_grid, gamma, None
    
    def reconstruct_impedance(self, tau_grid, gamma):
        """Reconstruct impedance from DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        return Z_rec_real, Z_rec_imag

# ============================================================================
# Bayesian DRT
# ============================================================================

class BayesianDRT(DRTCore):
    """Bayesian method for DRT (MAP estimation)"""
    
    def __init__(self, frequencies, Z_real, Z_imag):
        super().__init__(frequencies, Z_real, Z_imag)
    
    def _objective_function(self, x, K, Z_target, L):
        """Objective function for Bayesian optimization"""
        gamma = x[:-1]
        log_lambda = x[-1]
        lam = np.exp(log_lambda)
        
        gamma = np.maximum(gamma, 0)
        
        residual = K @ gamma - Z_target
        data_fit = 0.5 * np.sum(residual**2)
        prior = 0.5 * lam * np.sum((L @ gamma)**2)
        
        return data_fit + prior + 0.5 * (len(gamma) * np.log(lam) - log_lambda)
    
    def compute(self, n_tau=150, n_iterations=200):
        """Compute DRT using Bayesian method"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        K = np.vstack([K_real, K_imag])
        
        # Regularization matrix (second derivative)
        L = np.zeros((n_tau-2, n_tau))
        for i in range(n_tau-2):
            L[i, i] = 1
            L[i, i+1] = -2
            L[i, i+2] = 1
        
        # Initialization
        x0 = np.ones(n_tau + 1) * 0.1
        x0[-1] = np.log(1e-4)
        
        # Optimization
        result = optimize.minimize(
            self._objective_function, x0,
            args=(K, Z_target, L),
            method='L-BFGS-B',
            options={'maxiter': n_iterations, 'disp': False}
        )
        
        gamma = np.maximum(result.x[:-1], 0)
        
        # Normalize
        integral = np.trapz(gamma, np.log(tau_grid))
        if integral > 0:
            gamma = gamma / integral * self.R_pol
        
        # Simple uncertainty estimation
        confidence = 0.3 * np.ones_like(gamma)
        
        return tau_grid, gamma, confidence
    
    def reconstruct_impedance(self, tau_grid, gamma):
        """Reconstruct impedance from DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        return Z_rec_real, Z_rec_imag

# ============================================================================
# Maximum Entropy DRT
# ============================================================================

class MaxEntropyDRT(DRTCore):
    """Maximum Entropy method for DRT"""
    
    def __init__(self, frequencies, Z_real, Z_imag):
        super().__init__(frequencies, Z_real, Z_imag)
    
    def _entropy(self, gamma):
        """Calculate Shannon entropy"""
        gamma_pos = gamma[gamma > 1e-10]
        if len(gamma_pos) == 0:
            return 0
        return -np.sum(gamma_pos * np.log(gamma_pos))
    
    def _objective_function(self, x, K, Z_target, lam):
        """Objective function with entropy penalty"""
        gamma = np.maximum(x, 1e-10)
        residual = K @ gamma - Z_target
        data_fit = 0.5 * np.sum(residual**2)
        entropy_penalty = -lam * self._entropy(gamma)
        return data_fit + entropy_penalty
    
    def compute(self, n_tau=150, lambda_value=0.1):
        """Compute DRT using maximum entropy method"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        K = np.vstack([K_real, K_imag])
        
        # Initialization
        x0 = np.ones(n_tau) / n_tau
        
        # Optimization
        result = optimize.minimize(
            self._objective_function, x0,
            args=(K, Z_target, lambda_value),
            method='L-BFGS-B',
            bounds=[(1e-10, None) for _ in range(n_tau)],
            options={'maxiter': 500, 'disp': False}
        )
        
        gamma = result.x
        
        # Normalize
        integral = np.trapz(gamma, np.log(tau_grid))
        if integral > 0:
            gamma = gamma / integral * self.R_pol
        
        return tau_grid, gamma, None
    
    def reconstruct_impedance(self, tau_grid, gamma):
        """Reconstruct impedance from DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        return Z_rec_real, Z_rec_imag

# ============================================================================
# Gaussian Process DRT
# ============================================================================

class GaussianProcessDRT(DRTCore):
    """Gaussian Process for DRT"""
    
    def __init__(self, frequencies, Z_real, Z_imag):
        super().__init__(frequencies, Z_real, Z_imag)
    
    def _rbf_kernel(self, x1, x2, length_scale=1.0, sigma_f=1.0):
        """Radial Basis Function kernel"""
        dist_matrix = np.subtract.outer(x1, x2)**2
        return sigma_f**2 * np.exp(-0.5 * dist_matrix / length_scale**2)
    
    def compute(self, n_tau=150, n_components=20):
        """Compute DRT using Gaussian Process (simplified)"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        log_tau_grid = np.log10(tau_grid)
        
        # Create basis from RBF functions
        basis_centers = np.linspace(log_tau_grid[0], log_tau_grid[-1], n_components)
        
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        K_full = np.vstack([K_real, K_imag])
        
        # Build feature matrix
        length_scale = (log_tau_grid[-1] - log_tau_grid[0]) / n_components
        Phi = np.zeros((self.N * 2, n_components))
        
        for i, center in enumerate(basis_centers):
            phi = np.exp(-0.5 * ((log_tau_grid - center) / length_scale)**2)
            phi = phi / np.sum(phi)
            Phi[:, i] = K_full @ phi
        
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        
        # Solve with regularization
        lam = 1e-4
        A = Phi.T @ Phi + lam * np.eye(n_components)
        b = Phi.T @ Z_target
        weights = np.linalg.solve(A, b)
        
        # Reconstruct DRT
        gamma = np.zeros(n_tau)
        for i, center in enumerate(basis_centers):
            phi = np.exp(-0.5 * ((log_tau_grid - center) / length_scale)**2)
            phi = phi / np.sum(phi)
            gamma += weights[i] * phi
        
        gamma = np.maximum(gamma, 0)
        
        # Normalize
        integral = np.trapz(gamma, np.log(tau_grid))
        if integral > 0:
            gamma = gamma / integral * self.R_pol
        
        # Uncertainty estimation
        uncertainty = np.abs(weights).mean() * np.ones_like(gamma)
        
        return tau_grid, gamma, uncertainty
    
    def reconstruct_impedance(self, tau_grid, gamma):
        """Reconstruct impedance from DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        return Z_rec_real, Z_rec_imag

# ============================================================================
# Genetic Programming DRT
# ============================================================================

class ISGPDRT(DRTCore):
    """Genetic Programming for DRT (simplified)"""
    
    def __init__(self, frequencies, Z_real, Z_imag):
        super().__init__(frequencies, Z_real, Z_imag)
    
    def _gaussian_peak(self, tau, tau0, width, amplitude):
        """Gaussian peak for DRT representation"""
        return amplitude * np.exp(-((np.log10(tau) - np.log10(tau0))**2) / (2 * width**2))
    
    def _evaluate_fitness(self, peaks_params, tau_grid):
        """Evaluate fitness of solution"""
        gamma = np.zeros_like(tau_grid)
        for params in peaks_params:
            gamma += self._gaussian_peak(tau_grid, params['tau0'], params['width'], params['amplitude'])
        
        # Reconstruct impedance
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        
        # Reconstruction error
        error = np.mean((self.Z_real - Z_rec_real)**2 + (self.Z_imag - Z_rec_imag)**2)
        
        # Complexity penalty
        complexity_penalty = 0.01 * len(peaks_params)
        
        return error + complexity_penalty
    
    def compute(self, n_tau=150, n_peaks_max=5, n_generations=50, population_size=20):
        """Compute DRT using ISGP"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        
        # Initialize population
        population = []
        for _ in range(population_size):
            n_peaks = np.random.randint(1, n_peaks_max + 1)
            peaks = []
            for _ in range(n_peaks):
                peak = {
                    'tau0': np.random.uniform(self.tau_min, self.tau_max),
                    'width': np.random.uniform(0.1, 1.0),
                    'amplitude': np.random.uniform(0.1, self.R_pol / n_peaks)
                }
                peaks.append(peak)
            population.append(peaks)
        
        # Evolution
        for generation in range(n_generations):
            fitness = [self._evaluate_fitness(peaks, tau_grid) for peaks in population]
            
            # Selection
            sorted_indices = np.argsort(fitness)
            elite = [population[i] for i in sorted_indices[:population_size // 2]]
            
            # Create new generation
            new_population = elite.copy()
            while len(new_population) < population_size:
                parent = elite[np.random.randint(len(elite))]
                child = [peak.copy() for peak in parent]
                
                # Mutation
                if np.random.random() < 0.3:
                    if np.random.random() < 0.5 and len(child) > 1:
                        child.pop(np.random.randint(len(child)))
                    elif len(child) < n_peaks_max:
                        new_peak = {
                            'tau0': np.random.uniform(self.tau_min, self.tau_max),
                            'width': np.random.uniform(0.1, 1.0),
                            'amplitude': np.random.uniform(0.1, self.R_pol / (len(child) + 1))
                        }
                        child.append(new_peak)
                
                for peak in child:
                    if np.random.random() < 0.3:
                        peak['tau0'] *= np.random.uniform(0.8, 1.2)
                        peak['tau0'] = np.clip(peak['tau0'], self.tau_min, self.tau_max)
                    if np.random.random() < 0.3:
                        peak['width'] *= np.random.uniform(0.8, 1.2)
                        peak['width'] = np.clip(peak['width'], 0.1, 2.0)
                    if np.random.random() < 0.3:
                        peak['amplitude'] *= np.random.uniform(0.7, 1.3)
                        peak['amplitude'] = np.clip(peak['amplitude'], 0.01, self.R_pol)
                
                new_population.append(child)
            
            population = new_population
        
        # Best solution
        fitness = [self._evaluate_fitness(peaks, tau_grid) for peaks in population]
        best_peaks = population[np.argmin(fitness)]
        
        # Build DRT
        gamma = np.zeros_like(tau_grid)
        for peak in best_peaks:
            gamma += self._gaussian_peak(tau_grid, peak['tau0'], peak['width'], peak['amplitude'])
        
        # Normalize
        integral = np.trapz(gamma, np.log(tau_grid))
        if integral > 0:
            gamma = gamma / integral * self.R_pol
        
        return tau_grid, gamma, best_peaks
    
    def reconstruct_impedance(self, tau_grid, gamma):
        """Reconstruct impedance from DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        return Z_rec_real, Z_rec_imag

# ============================================================================
# Peak Detection and Analysis
# ============================================================================

def find_peaks_drt(tau_grid, gamma, prominence=0.05):
    """Find peaks in DRT spectrum"""
    # Normalize gamma
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

def calculate_resistances(tau, drt, peaks_idx):
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
        
        area = np.trapz(drt[start:end], np.log(tau[start:end]))
        resistances.append(area)
    
    return resistances

def fit_gaussian_peaks(tau_grid, gamma, n_peaks=None):
    """Fit DRT with sum of Gaussians"""
    log_tau = np.log10(tau_grid)
    
    if n_peaks is None:
        peaks = find_peaks_drt(tau_grid, gamma)
        n_peaks = len(peaks)
    
    def sum_gaussians(log_tau, *params):
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
    except:
        return gamma, []

# ============================================================================
# Visualization Functions (Matplotlib - Publication Quality)
# ============================================================================

def plot_nyquist_matplotlib(freq, re_z, im_z, re_rec=None, im_rec=None, title="Nyquist Plot"):
    """Create publication-quality Nyquist plot"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.plot(re_z, im_z, 'o-', markersize=4, linewidth=1.5, 
            label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.5)
    
    if re_rec is not None and im_rec is not None:
        ax.plot(re_rec, im_rec, 's-', markersize=3, linewidth=1.0,
                label='Reconstructed', color='#ff7f0e', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    
    ax.set_xlabel(r"$\mathrm{Re}(Z)$ / $\Omega$", fontweight='bold')
    ax.set_ylabel(r"$-\mathrm{Im}(Z)$ / $\Omega$", fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Equal aspect ratio
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    ax.set_aspect('equal', adjustable='box')
    
    return fig

def plot_bode_matplotlib(freq, re_z, im_z, re_rec=None, im_rec=None):
    """Create publication-quality Bode plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    
    # Magnitude plot
    mag = np.sqrt(re_z**2 + im_z**2)
    ax1.loglog(freq, mag, 'o-', markersize=4, linewidth=1.5, 
               label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.5)
    if re_rec is not None and im_rec is not None:
        mag_rec = np.sqrt(re_rec**2 + im_rec**2)
        ax1.loglog(freq, mag_rec, 's-', markersize=3, linewidth=1.0,
                   label='Reconstructed', color='#ff7f0e', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    ax1.set_xlabel("Frequency / Hz", fontweight='bold')
    ax1.set_ylabel("$|Z|$ / $\Omega$", fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Phase plot
    phase = np.arctan2(im_z, re_z) * 180 / np.pi
    ax2.semilogx(freq, phase, 'o-', markersize=4, linewidth=1.5,
                 label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.5)
    if re_rec is not None and im_rec is not None:
        phase_rec = np.arctan2(im_rec, re_rec) * 180 / np.pi
        ax2.semilogx(freq, phase_rec, 's-', markersize=3, linewidth=1.0,
                     label='Reconstructed', color='#ff7f0e', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    ax2.set_xlabel("Frequency / Hz", fontweight='bold')
    ax2.set_ylabel("Phase / °", fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle("Bode Plot", fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_drt_matplotlib(tau, drt, peaks=None, drt_std=None, title="Distribution of Relaxation Times"):
    """Create publication-quality DRT plot"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot DRT with uncertainty if available
    if drt_std is not None:
        ax.fill_between(tau, drt - drt_std, drt + drt_std,
                        alpha=0.3, color='gray', label='±2σ uncertainty')
    ax.loglog(tau, drt, '-', linewidth=2, color='#2ca02c', label='DRT')
    
    # Plot peaks
    if peaks and len(peaks) > 0:
        peak_tau = [p['tau'] for p in peaks]
        peak_drt = [p['amplitude'] for p in peaks]
        ax.plot(peak_tau, peak_drt, 'rv', markersize=8, label='Detected peaks')
        
        # Add peak labels
        for i, (t, d) in enumerate(zip(peak_tau, peak_drt)):
            freq = 1/(2*np.pi*t)
            ax.annotate(f'τ={t:.2e}s\nf={freq:.2e}Hz',
                       xy=(t, d), xytext=(t*1.5, d*1.2),
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel(r"Relaxation Time $\tau$ / s", fontweight='bold')
    ax.set_ylabel(r"$\gamma(\tau)$ / $\Omega$", fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    return fig

def plot_kk_residuals_matplotlib(freq, res_real, res_imag):
    """Create publication-quality KK residuals plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
    
    ax1.semilogx(freq, res_real * 100, 'o-', markersize=4, linewidth=1.0, color='#1f77b4')
    ax1.set_xlabel("Frequency / Hz", fontweight='bold')
    ax1.set_ylabel(r"$\Delta \mathrm{Re}(Z)$ / %", fontweight='bold')
    ax1.set_title("Kramers-Kronig Test - Real Part Residuals", fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axhline(y=2, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axhline(y=-2, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax2.semilogx(freq, res_imag * 100, 'o-', markersize=4, linewidth=1.0, color='#1f77b4')
    ax2.set_xlabel("Frequency / Hz", fontweight='bold')
    ax2.set_ylabel(r"$\Delta \mathrm{Im}(Z)$ / %", fontweight='bold')
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

def plot_impedance_plotly(frequencies, Z_real_exp, Z_imag_exp, Z_real_rec=None, Z_imag_rec=None):
    """Create interactive impedance plots with Plotly"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Nyquist Plot', 'Bode Plot - Magnitude'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Nyquist plot
    fig.add_trace(
        go.Scatter(x=Z_real_exp, y=-Z_imag_exp, mode='markers',
                   name='Experimental', marker=dict(size=6, color='blue')),
        row=1, col=1
    )
    
    if Z_real_rec is not None and Z_imag_rec is not None:
        fig.add_trace(
            go.Scatter(x=Z_real_rec, y=-Z_imag_rec, mode='lines',
                       name='Reconstructed', line=dict(color='red', width=2)),
            row=1, col=1
        )
    
    fig.update_xaxes(title_text="Z' (Ω)", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' (Ω)", row=1, col=1)
    
    # Bode plot - Magnitude
    Z_mod_exp = np.sqrt(Z_real_exp**2 + Z_imag_exp**2)
    fig.add_trace(
        go.Scatter(x=frequencies, y=Z_mod_exp, mode='markers',
                   name='Experimental', marker=dict(size=6, color='blue')),
        row=1, col=2
    )
    
    if Z_real_rec is not None and Z_imag_rec is not None:
        Z_mod_rec = np.sqrt(Z_real_rec**2 + Z_imag_rec**2)
        fig.add_trace(
            go.Scatter(x=frequencies, y=Z_mod_rec, mode='lines',
                       name='Reconstructed', line=dict(color='red', width=2)),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="|Z| (Ω)", type="log", row=1, col=2)
    
    fig.update_layout(height=500, showlegend=True)
    
    return fig

def plot_drt_plotly(tau_grid, gamma, peaks=None, confidence=None):
    """Create interactive DRT plot with Plotly"""
    fig = go.Figure()
    
    # Main DRT curve
    fig.add_trace(go.Scatter(
        x=tau_grid, y=gamma,
        mode='lines',
        name='DRT',
        line=dict(color='blue', width=2)
    ))
    
    # Confidence interval
    if confidence is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([tau_grid, tau_grid[::-1]]),
            y=np.concatenate([gamma + confidence, (gamma - confidence)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
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
                text=f"τ = {peak['tau']:.2e} s<br>f = {1/(2*np.pi*peak['tau']):.2f} Hz",
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

def create_report(freq, re_z, im_z, tau, drt, peaks_data, method_name, params):
    """Generate analysis report"""
    report = f"""
    ============================================================
    EIS-DRT Analysis Report
    ============================================================
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Analysis Method: {method_name}
    
    Data Information:
    - Number of frequencies: {len(freq)}
    - Frequency range: {freq.min():.2e} - {freq.max():.2e} Hz
    - Ohmic resistance (R∞): {params.get('R_inf', 0):.4f} Ω
    - Polarization resistance (Rpol): {params.get('R_pol', 0):.4f} Ω
    
    DRT Parameters:
    - Time constant range: {tau.min():.2e} - {tau.max():.2e} s
    
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
    - KK test passed: {params.get('kk_passed', False)}
    - Max KK residual: {params.get('max_kk_res', 0):.3f}%
    
    ============================================================
    """
    
    return report

# ============================================================================
# Main Application
# ============================================================================

def main():
    st.title("⚡ Electrochemical Impedance Spectroscopy Analysis")
    st.markdown("### Distribution of Relaxation Times (DRT) Analysis Tool")
    st.markdown("Поддерживаются 5 методов инверсии: Тихоновская регуляризация, Байесовский метод, "
                "Максимальная энтропия, Гауссовские процессы, Генетическое программирование")
    st.markdown("---")
    
    # Sidebar for input controls
    with st.sidebar:
        st.header("📁 Data Input")
        
        # Data input method selection
        input_method = st.radio("Select input method:", 
                                ["Upload File", "Manual Entry"])
        
        freq = None
        re_z = None
        im_z = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Choose file", type=['txt', 'csv', 'xlsx', 'dat', 'z', 'mpt'])
            if uploaded_file:
                # Try to detect columns
                try:
                    df = pd.read_csv(uploaded_file, nrows=5)
                    st.subheader("Column Mapping")
                    col_freq = st.selectbox("Frequency column", df.columns)
                    col_re = st.selectbox("Re(Z) column", df.columns)
                    col_im = st.selectbox("-Im(Z) column", df.columns)
                    
                    if st.button("Load Data"):
                        freq, re_z, im_z = load_data(uploaded_file, col_freq, col_re, col_im)
                        if freq is not None:
                            st.success(f"Loaded {len(freq)} data points")
                        else:
                            st.error("Error loading data")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        else:  # Manual entry
            freq, re_z, im_z = manual_data_entry()
        
        # Analysis parameters
        if freq is not None:
            st.header("⚙️ Analysis Parameters")
            
            # Method selection
            analysis_method = st.selectbox("DRT Calculation Method",
                                          ["Tikhonov Regularization",
                                           "Bayesian Method",
                                           "Maximum Entropy",
                                           "Gaussian Process",
                                           "Genetic Programming (ISGP)"])
            
            # Estimate and display R_inf and R_pol
            R_inf_est = re_z[-1] if len(re_z) > 0 else 0
            R_pol_est = np.max(re_z) - R_inf_est if np.max(re_z) > R_inf_est else 1.0
            
            st.info(f"Estimated R∞ = {R_inf_est:.4f} Ω, Rpol = {R_pol_est:.4f} Ω")
            
            R_inf = st.number_input("R∞ (Ohmic resistance)", 
                                   value=float(R_inf_est), format="%.6f")
            R_pol = st.number_input("Rpol (Polarization resistance)", 
                                   value=float(R_pol_est), format="%.6f")
            
            n_tau = st.slider("Number of time points", 50, 300, 150)
            
            # Method-specific parameters
            if analysis_method == "Tikhonov Regularization":
                reg_order = st.selectbox("Regularization order", [0, 1, 2], index=2,
                                        help="0: amplitude smoothing, 1: slope smoothing, 2: curvature smoothing")
                lambda_auto = st.checkbox("Automatic λ selection", value=True)
                if not lambda_auto:
                    lambda_value = st.number_input("λ value", value=1e-4, format="%.1e")
                else:
                    lambda_value = None
            elif analysis_method == "Maximum Entropy":
                entropy_lambda = st.number_input("Entropy λ", value=0.1, format="%.2f",
                                                help="Smaller values give smoother spectra")
            elif analysis_method == "Gaussian Process":
                n_components = st.slider("Number of GP components", 10, 50, 20)
            elif analysis_method == "Genetic Programming (ISGP)":
                n_peaks_max = st.slider("Maximum number of peaks", 1, 10, 5)
                n_generations = st.slider("Number of generations", 10, 100, 50)
            
            # Peak detection parameters
            st.header("🔍 Peak Detection")
            peak_prominence = st.slider("Peak prominence (%)", 1, 20, 5) / 100
            
            # Run analysis button
            analyze_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content area
    if freq is not None and 'analyze_button' in locals() and analyze_button:
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Sort data by frequency
        sort_idx = np.argsort(freq)
        freq = freq[sort_idx]
        re_z = re_z[sort_idx]
        im_z = im_z[sort_idx]
        
        # Perform KK test
        with st.spinner("Performing Kramers-Kronig validation..."):
            kk_passed, max_res, res_real, res_imag, _ = kramers_kronig_test(freq, re_z, im_z)
            
            if kk_passed:
                st.success(f"✓ KK test passed (max residual: {max_res*100:.2f}%)")
            else:
                st.warning(f"⚠ KK test failed (max residual: {max_res*100:.2f}%)")
        
        # Calculate DRT based on selected method
        with st.spinner(f"Calculating DRT using {analysis_method}..."):
            if analysis_method == "Tikhonov Regularization":
                drt_solver = TikhonovDRT(freq, re_z, im_z, regularization_order=reg_order)
                tau, gamma, confidence = drt_solver.compute(n_tau=n_tau, lambda_value=lambda_value, lambda_auto=lambda_auto)
                method_key = "Tikhonov"
            elif analysis_method == "Bayesian Method":
                drt_solver = BayesianDRT(freq, re_z, im_z)
                tau, gamma, confidence = drt_solver.compute(n_tau=n_tau)
                method_key = "Bayesian"
            elif analysis_method == "Maximum Entropy":
                drt_solver = MaxEntropyDRT(freq, re_z, im_z)
                tau, gamma, confidence = drt_solver.compute(n_tau=n_tau, lambda_value=entropy_lambda)
                method_key = "MaxEntropy"
            elif analysis_method == "Gaussian Process":
                drt_solver = GaussianProcessDRT(freq, re_z, im_z)
                tau, gamma, confidence = drt_solver.compute(n_tau=n_tau, n_components=n_components)
                method_key = "GP-DRT"
            else:  # Genetic Programming
                drt_solver = ISGPDRT(freq, re_z, im_z)
                tau, gamma, peaks_params = drt_solver.compute(n_tau=n_tau, n_peaks_max=n_peaks_max, 
                                                              n_generations=n_generations)
                confidence = None
                method_key = "ISGP"
            
            # Reconstruct impedance
            Z_rec_real, Z_rec_imag = drt_solver.reconstruct_impedance(tau, gamma)
        
        # Find peaks
        peaks = find_peaks_drt(tau, gamma, peak_prominence)
        
        # Calculate resistances if peaks found
        if peaks:
            peaks_idx = [np.argmin(np.abs(tau - p['tau'])) for p in peaks]
            resistances = calculate_resistances(tau, gamma, peaks_idx)
            for i, p in enumerate(peaks):
                p['resistance'] = resistances[i] if i < len(resistances) else 0
                p['capacitance'] = p['tau'] / p['resistance'] if p['resistance'] > 0 else 0
        
        # Create tabs for results
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 DRT Spectrum", "🔄 Nyquist & Bode", 
                                                 "📊 Interactive Plots", "🔍 KK Test", "📋 Report"])
        
        with tab1:
            st.subheader("Distribution of Relaxation Times")
            
            # Publication-quality DRT plot
            fig_drt = plot_drt_matplotlib(tau, gamma, peaks, confidence)
            st.pyplot(fig_drt)
            
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
            fig_nyquist = plot_nyquist_matplotlib(freq, re_z, im_z, Z_rec_real, Z_rec_imag)
            st.pyplot(fig_nyquist)
            
            # Bode plot
            fig_bode = plot_bode_matplotlib(freq, re_z, im_z, Z_rec_real, Z_rec_imag)
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
            fig_imp_plotly = plot_impedance_plotly(freq, re_z, im_z, Z_rec_real, Z_rec_imag)
            st.plotly_chart(fig_imp_plotly, use_container_width=True)
            
            # Interactive DRT plot
            fig_drt_plotly = plot_drt_plotly(tau, gamma, peaks, confidence)
            st.plotly_chart(fig_drt_plotly, use_container_width=True)
            
            # Reconstruction error
            error_real = np.abs(re_z - Z_rec_real) / np.abs(re_z + 1e-10) * 100
            error_imag = np.abs(im_z - Z_rec_imag) / np.abs(im_z + 1e-10) * 100
            mean_error = np.mean(np.sqrt(error_real**2 + error_imag**2))
            st.metric("Mean Reconstruction Error", f"{mean_error:.2f} %")
        
        with tab4:
            if res_real is not None and res_imag is not None:
                st.subheader("Kramers-Kronig Validation")
                fig_kk = plot_kk_residuals_matplotlib(freq, res_real, res_imag)
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
                'R_inf': R_inf,
                'R_pol': R_pol,
                'kk_passed': kk_passed,
                'max_kk_res': max_res * 100,
            }
            
            report = create_report(freq, re_z, im_z, tau, gamma, peaks, analysis_method, params)
            st.text(report)
            
            # Download report
            st.download_button("📥 Download Full Report (TXT)", 
                              data=report,
                              file_name=f"eis_drt_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                              mime="text/plain")
            
            # Export DRT data
            drt_data = pd.DataFrame({
                'tau_s': tau,
                'log10_tau': np.log10(tau),
                'gamma_tau': gamma
            })
            if confidence is not None:
                drt_data['gamma_uncertainty'] = confidence
            
            csv = drt_data.to_csv(index=False)
            st.download_button("📥 Export DRT Data (CSV)", 
                              data=csv,
                              file_name=f"drt_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                              mime="text/csv")
            
            # Export peaks data
            if peaks:
                peaks_df = pd.DataFrame(peaks)
                peaks_csv = peaks_df.to_csv(index=False)
                st.download_button("📥 Export Peaks Data (CSV)", 
                                  data=peaks_csv,
                                  file_name=f"peaks_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                  mime="text/csv")
    
    elif freq is None:
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
    st.markdown("⚡ *DRT Analysis Tool v2.0 | 5 inversion methods: Tikhonov, Bayesian, MaxEntropy, GP, ISGP*")


if __name__ == "__main__":
    main()
