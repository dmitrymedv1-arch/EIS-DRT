"""
Electrochemical Impedance Spectroscopy (EIS) Analysis Tool
Distribution of Relaxation Times (DRT) Analysis with Gaussian Deconvolution

Поддерживаемые методы:
- Тихоновская регуляризация (Tikhonov) с NNLS
- Байесовский метод с MCMC (PyMC)
- Метод максимальной энтропии (Maximum Entropy) с авто-выбором λ
- Гауссовские процессы (fGP-DRT) с non-negativity constraints
- Loewner Framework (RLF) - data-driven метод
- Generalized DRT для обработки индуктивных петель

Дополнительно:
- Gaussian Deconvolution of DRT Peaks
- Multi-stage workflow with state preservation
- Peak editing and area distribution analysis

Author: DRT Analysis Tool
Version: 4.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, AutoMinorLocator
from matplotlib.patches import Patch
from scipy import optimize, linalg, interpolate, integrate
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_func
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, least_squares
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
import logging
from typing import Tuple, Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
import time

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
    page_title="EIS-DRT Analysis Tool v4.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Modern Scientific Styling
# ============================================================================

def apply_modern_scientific_style():
    """Apply modern scientific plotting style with enhanced aesthetics"""
    plt.style.use('default')
    plt.rcParams.update({
        # Font settings
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        
        # Axes settings
        'axes.labelsize': 14,
        'axes.labelweight': 'bold',
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.facecolor': '#f8f9fa',
        'axes.edgecolor': '#2c3e50',
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        
        # Tick settings
        'xtick.color': '#2c3e50',
        'ytick.color': '#2c3e50',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'xtick.minor.size': 4,
        'ytick.major.size': 6,
        'ytick.minor.size': 4,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        
        # Legend settings
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#2c3e50',
        'legend.fancybox': True,
        'legend.shadow': True,
        
        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',
        'figure.figsize': [12, 8],
        
        # Lines and markers
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'lines.markeredgewidth': 1,
        'errorbar.capsize': 3,
        
        # Color cycles - modern scientific palette
        'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', 
                                              '#d62728', '#9467bd', '#8c564b', 
                                              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    })

apply_modern_scientific_style()

# Custom CSS for modern matte effects
st.markdown("""
<style>
    /* Modern button styling with matte effect */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46a0 100%);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Primary button special styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* Metric cards */
    .stMetric {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border: 1px solid #e2e8f0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 10px;
        border-left-width: 4px;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Navigation step indicators */
    .step-indicator {
        background: white;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .step-completed {
        color: #48bb78;
        font-weight: bold;
    }
    .step-current {
        color: #667eea;
        font-weight: bold;
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 4px 8px;
        border-radius: 6px;
    }
    .step-pending {
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)


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
# Data Classes for Gaussian Deconvolution Results
# ============================================================================

@dataclass
class GaussianPeak:
    """Container for Gaussian peak parameters"""
    id: int
    center: float          # Center in linear space
    center_log: float      # Center in log space
    amplitude: float       # Amplitude in original scale
    amplitude_norm: float  # Normalized amplitude
    sigma_log: float       # Sigma in log space
    fwhm: float           # Full width at half maximum
    area: float           # Area under peak
    fraction: float       # Fraction of total area
    fraction_percent: float  # Percentage fraction
    source: str = 'auto'  # Source: 'auto', 'manual', 'residuals'
    y_norm: np.ndarray = None  # Normalized y values for plotting


@dataclass
class DeconvolutionResult:
    """Container for Gaussian deconvolution results"""
    peaks: List[GaussianPeak]
    fit_y_norm: np.ndarray
    x: np.ndarray
    y_norm: np.ndarray
    y_original: np.ndarray
    x_linear: np.ndarray
    use_log_x: bool
    use_log_y: bool
    quality_metrics: Dict[str, Any]
    baseline_params: Optional[List[float]] = None
    baseline_method: str = 'none'
    total_area: float = 0.0
    max_amplitude: float = 0.0


# ============================================================================
# Application State Management
# ============================================================================

@dataclass
class AppState:
    """Centralized state management for the entire application"""
    # Current step
    current_step: int = 1
    
    # Step 1: Data loading
    impedance_data: Optional[ImpedanceData] = None
    data_loaded: bool = False
    
    # Step 2: DRT calculation
    drt_result: Optional[DRTResult] = None
    drt_method: str = "Tikhonov Regularization (NNLS)"
    drt_parameters: Dict[str, Any] = field(default_factory=dict)
    drt_calculated: bool = False
    
    # Step 3: Gaussian deconvolution
    deconvolver: Optional[Any] = None
    derivatives: Optional[Tuple] = None
    deconv_result: Optional[DeconvolutionResult] = None
    peak_info: Optional[List[Dict]] = None
    initial_peak_params: Optional[List[float]] = None
    manual_peaks: List[Dict] = field(default_factory=list)
    residuals_peaks: List[Dict] = field(default_factory=list)
    pending_remove: Optional[int] = None
    pending_split: Optional[Tuple[int, float]] = None
    manual_peak_position: Optional[float] = None
    deconv_parameters: Dict[str, Any] = field(default_factory=dict)
    deconv_calculated: bool = False
    
    # Step 4: Results
    results_ready: bool = False
    
    # General settings
    use_log_x: bool = True
    use_log_y: bool = False
    clip_negative: bool = True
    show_warnings: bool = True
    smoothing_level: str = 'none'
    baseline_method: str = 'none'
    fitting_method: str = 'trf'
    fit_quality: str = 'balanced'
    max_nfev: int = 5000
    preview_mode: bool = False
    
    # Peak detection settings
    sensitivity: float = 0.03
    min_distance: int = 5
    
    # Temporary storage for preview
    preview_fit: Optional[np.ndarray] = None
    last_popt: Optional[np.ndarray] = None


# Initialize session state
if 'app_state' not in st.session_state:
    st.session_state.app_state = AppState()


# ============================================================================
# Data Loading and Validation (from EIS code)
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
    st.subheader("📝 Ручной ввод данных")
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
    
    if st.button("📥 Загрузить данные", type="primary", use_container_width=True):
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
                
                with st.expander("📊 Просмотр загруженных данных"):
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
        from scipy.signal import hilbert
        analytic = hilbert(interp_im)
        re_predicted = np.real(analytic)
        
        # Interpolate back to original frequencies
        re_pred_original = interp1d(interp_omega, re_predicted, kind='cubic', fill_value='extrapolate')(omega)
        
        # Calculate residuals
        residuals = (re_z - re_pred_original) / np.abs(re_z + 1e-10)
        max_residual = np.max(np.abs(residuals))
        is_valid = max_residual < 0.05
        
        return is_valid, max_residual, residuals, np.zeros_like(residuals)
    except Exception as e:
        logging.warning(f"KK Hilbert transform failed: {e}")
        return False, 1.0, None, None


# ============================================================================
# Base DRT Class with Generalized Support (from EIS code)
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
        
        # Determine if inductive behavior is present
        high_freq_idx = np.where(self.frequencies > 0.1 * np.max(self.frequencies))[0]
        if len(high_freq_idx) > 0:
            self.has_inductive_loop = np.any(self.Z_imag[high_freq_idx] > 0)
        else:
            self.has_inductive_loop = False
        
        # Automatic determination of relaxation time range
        self.tau_min = 1.0 / (2 * np.pi * np.max(self.frequencies)) * 0.1
        self.tau_max = 1.0 / (2 * np.pi * np.min(self.frequencies)) * 10
        
        # Estimate ohmic resistance
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
# Tikhonov Regularization with NNLS (from EIS code)
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
        
        lambda_opt = None
        
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
            lambda_opt = lambda_value if lambda_value is not None else 1e-4
            A = np.vstack([K, lambda_opt * L])
            b = np.concatenate([Z_target, np.zeros(L.shape[0])])
            gamma = self._solve_nnls(A, b)
        
        gamma_std = np.abs(np.gradient(np.gradient(gamma))) * 0.1
        
        return DRTResult(
            tau_grid=tau_grid,
            gamma=gamma,
            gamma_std=gamma_std,
            method="Tikhonov Regularization (NNLS)",
            R_inf=self.R_inf,
            R_pol=self.R_pol,
            metadata={
                'lambda': lambda_opt,
                'order': self.regularization_order,
                'lambda_auto': lambda_auto
            }
        )
    
    def reconstruct_impedance(self, tau_grid: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct impedance from DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=self.include_inductive)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        return Z_rec_real, Z_rec_imag


# ============================================================================
# Bayesian DRT with MCMC (from EIS code)
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
        
        L = np.zeros((n_tau-2, n_tau))
        for i in range(n_tau-2):
            L[i, i] = 1
            L[i, i+1] = -2
            L[i, i+2] = 1
        
        with pm.Model() as model:
            gamma_raw = pm.HalfNormal('gamma_raw', sigma=1.0, shape=n_tau)
            
            smoothness = pm.HalfCauchy('smoothness', beta=0.1)
            reg_penalty = smoothness * pm.math.sum(pm.math.abs(L @ gamma_raw))
            
            sigma = pm.HalfCauchy('sigma', beta=0.1)
            Z_pred = pm.math.dot(K, gamma_raw)
            likelihood = pm.Normal('likelihood', mu=Z_pred, sigma=sigma, observed=Z_target)
            
            pm.Potential('reg', -reg_penalty)
            
            trace = pm.sample(draws=n_samples, tune=n_tune, chains=n_chains, 
                             return_inferencedata=True, progressbar=False)
        
        gamma_samples = trace.posterior['gamma_raw'].values.reshape(-1, n_tau)
        gamma_mean = np.mean(gamma_samples, axis=0)
        gamma_std = np.std(gamma_samples, axis=0)
        
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
# Maximum Entropy DRT with Automatic Lambda Selection (from EIS code)
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
# Finite Gaussian Process DRT (fGP-DRT) (from EIS code)
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
        
        basis_centers = np.linspace(log_tau_grid[0], log_tau_grid[-1], n_components)
        length_scale = (log_tau_grid[-1] - log_tau_grid[0]) / n_components
        
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=self.include_inductive)
        K_full = np.vstack([K_real, K_imag])
        
        Phi = np.zeros((self.N * 2, n_components))
        for i, center in enumerate(basis_centers):
            phi = np.exp(-0.5 * ((log_tau_grid - center) / length_scale)**2)
            phi = phi / np.sum(phi)
            Phi[:, i] = K_full @ phi
        
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        
        lam = 1e-4
        A = Phi.T @ Phi + lam * np.eye(n_components)
        b = Phi.T @ Z_target
        weights_init = np.linalg.solve(A, b)
        weights_init = np.maximum(weights_init, 0)
        
        if PYMC_AVAILABLE:
            with pm.Model() as model:
                weights = pm.TruncatedNormal('weights', mu=weights_init, sigma=1.0, 
                                             lower=0, shape=n_components)
                
                sigma = pm.HalfCauchy('sigma', beta=0.1)
                Z_pred = pm.math.dot(Phi, weights)
                likelihood = pm.Normal('likelihood', mu=Z_pred, sigma=sigma, observed=Z_target)
                
                trace = pm.sample(draws=n_samples, tune=n_samples//2, 
                                 chains=2, progressbar=False)
                
                weights_samples = trace.posterior['weights'].values.reshape(-1, n_components)
                weights_mean = np.mean(weights_samples, axis=0)
                weights_std = np.std(weights_samples, axis=0)
        else:
            weights_mean = weights_init
            weights_std = np.ones_like(weights_init) * 0.1
        
        gamma = np.zeros(n_tau)
        gamma_std = np.zeros(n_tau)
        for i, center in enumerate(basis_centers):
            phi = np.exp(-0.5 * ((log_tau_grid - center) / length_scale)**2)
            phi = phi / np.sum(phi)
            gamma += weights_mean[i] * phi
            gamma_std += weights_std[i] * phi
        
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
# Loewner Framework (RLF) - Data-Driven DRT (from EIS code)
# ============================================================================

class LoewnerFrameworkDRT(DRTCore):
    """Loewner Framework for data-driven DRT extraction"""
    
    def __init__(self, data: ImpedanceData):
        super().__init__(data, include_inductive=False)
    
    def _build_loewner_matrices(self, omega: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build Loewner and shifted Loewner matrices"""
        n = len(omega)
        n_left = n // 2
        n_right = n - n_left
        
        left_omega = omega[:n_left]
        right_omega = omega[n_left:]
        left_Z = Z[:n_left]
        right_Z = Z[n_left:]
        
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
        
        diffs = np.diff(singular_values)
        diffs2 = np.diff(diffs)
        
        if len(diffs2) > 0:
            knee_idx = np.argmax(np.abs(diffs2)) + 1
            return min(knee_idx + 1, n)
        
        return n // 2
    
    def compute(self, n_tau: int = 150, model_order: Optional[int] = None) -> DRTResult:
        """Compute DRT using Loewner Framework"""
        
        omega = 2 * np.pi * self.frequencies
        Z = self.Z
        
        L, Ls, left_Z, right_Z = self._build_loewner_matrices(omega, Z)
        
        U, S, Vh = np.linalg.svd(L, full_matrices=False)
        
        if model_order is None:
            model_order = self._scree_not_threshold(S)
        
        model_order = max(1, min(model_order, len(S) - 1))
        
        U_r = U[:, :model_order]
        S_r = np.diag(S[:model_order])
        V_r = Vh[:model_order, :]
        
        E_r = -U_r.conj().T @ L @ V_r.conj().T
        A_r = -U_r.conj().T @ Ls @ V_r.conj().T
        B_r = U_r.conj().T @ left_Z
        C_r = right_Z @ V_r.conj().T
        
        try:
            eigvals, eigvecs = linalg.eig(A_r, E_r)
            
            tau_loewner = -1.0 / eigvals
            valid = (np.real(tau_loewner) > 0) & (np.imag(tau_loewner) < 1e-6)
            tau_loewner = np.real(tau_loewner[valid])
            
            R_loewner = np.zeros(len(tau_loewner))
            for i in range(len(tau_loewner)):
                R_loewner[i] = np.abs(C_r @ eigvecs[:, i] * (eigvecs[:, i].conj().T @ B_r))
        except:
            tau_loewner = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
            R_loewner = np.ones(n_tau) / n_tau
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        gamma = np.zeros(n_tau)
        
        if len(tau_loewner) > 1:
            idx_sorted = np.argsort(tau_loewner)
            tau_sorted = tau_loewner[idx_sorted]
            R_sorted = R_loewner[idx_sorted]
            
            interp_func = interpolate.interp1d(np.log10(tau_sorted), R_sorted, 
                                               kind='linear', fill_value=0, bounds_error=False)
            gamma = interp_func(np.log10(tau_grid))
            gamma = np.maximum(gamma, 0)
        
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
# Peak Detection and Analysis (from EIS code)
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

class GaussianModel:
    """Model for sum of Gaussians with baseline correction"""
    
    @staticmethod
    def gaussian(x, amp, cen, sigma):
        """Gaussian function with safe sigma"""
        return amp * np.exp(-(x - cen)**2 / (2 * max(sigma, np.finfo(float).eps)**2))
    
    @staticmethod
    def multi_gaussian(x, *params):
        """Sum of multiple Gaussians"""
        n = len(params) // 3
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            amp = params[3*i]
            cen = params[3*i + 1]
            sigma = abs(params[3*i + 2])
            y += GaussianModel.gaussian(x, amp, cen, sigma)
        return y
    
    @staticmethod
    def multi_gaussian_with_baseline(x, n_peaks, peak_params, baseline_params, baseline_method):
        """Sum of Gaussians with baseline correction"""
        # Calculate peaks
        y_peaks = np.zeros_like(x, dtype=float)
        for i in range(n_peaks):
            amp = peak_params[3*i]
            cen = peak_params[3*i + 1]
            sigma = abs(peak_params[3*i + 2])
            y_peaks += GaussianModel.gaussian(x, amp, cen, sigma)
        
        # Calculate baseline
        if baseline_method == "constant" and len(baseline_params) >= 1:
            y_baseline = baseline_params[0]
        elif baseline_method == "linear" and len(baseline_params) >= 2:
            y_baseline = baseline_params[0] + baseline_params[1] * x
        elif baseline_method == "quadratic" and len(baseline_params) >= 3:
            y_baseline = baseline_params[0] + baseline_params[1] * x + baseline_params[2] * x**2
        else:
            y_baseline = 0
        
        return y_peaks + y_baseline
    
    @staticmethod
    def calculate_area(amp, sigma):
        """Area under Gaussian"""
        return amp * sigma * np.sqrt(2 * np.pi)
    
    @staticmethod
    def calculate_fwhm(sigma):
        """Full width at half maximum"""
        return 2 * np.sqrt(2 * np.log(2)) * sigma
    
    @staticmethod
    def estimate_sigma_from_peak(x, y, peak_idx):
        """Estimate sigma with fallback methods"""
        try:
            widths, width_heights, left_ips, right_ips = peak_widths(
                y, [peak_idx], rel_height=0.5
            )
            fwhm = widths[0] * np.mean(np.diff(x))
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            return sigma
        except Exception as e:
            # Fallback: estimate from distance to nearest minimum
            left_min = peak_idx
            right_min = peak_idx
            
            # Find left minimum
            for i in range(peak_idx - 1, 0, -1):
                if y[i] < y[i-1] and y[i] < y[i+1]:
                    left_min = i
                    break
            
            # Find right minimum
            for i in range(peak_idx + 1, len(y) - 1):
                if y[i] < y[i-1] and y[i] < y[i+1]:
                    right_min = i
                    break
            
            # Estimate sigma as 1/3 of the width to nearest minima
            width = (right_min - left_min) * np.mean(np.diff(x))
            sigma = width / 3.0
            return max(sigma, 0.01 * (np.max(x) - np.min(x)) / 10)


# ============================================================================
# Gaussian Model for Deconvolution (from second code)
# ============================================================================

class GaussianModelDeconv:
    """Model for sum of Gaussians with baseline correction"""
    
    @staticmethod
    def gaussian(x, amp, cen, sigma):
        """Gaussian function with safe sigma"""
        return amp * np.exp(-(x - cen)**2 / (2 * max(sigma, np.finfo(float).eps)**2))
    
    @staticmethod
    def multi_gaussian(x, *params):
        """Sum of multiple Gaussians"""
        n = len(params) // 3
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            amp = params[3*i]
            cen = params[3*i + 1]
            sigma = abs(params[3*i + 2])
            y += GaussianModelDeconv.gaussian(x, amp, cen, sigma)
        return y
    
    @staticmethod
    def multi_gaussian_with_baseline(x, n_peaks, peak_params, baseline_params, baseline_method):
        """Sum of Gaussians with baseline correction"""
        y_peaks = np.zeros_like(x, dtype=float)
        for i in range(n_peaks):
            amp = peak_params[3*i]
            cen = peak_params[3*i + 1]
            sigma = abs(peak_params[3*i + 2])
            y_peaks += GaussianModelDeconv.gaussian(x, amp, cen, sigma)
        
        if baseline_method == "constant" and len(baseline_params) >= 1:
            y_baseline = baseline_params[0]
        elif baseline_method == "linear" and len(baseline_params) >= 2:
            y_baseline = baseline_params[0] + baseline_params[1] * x
        elif baseline_method == "quadratic" and len(baseline_params) >= 3:
            y_baseline = baseline_params[0] + baseline_params[1] * x + baseline_params[2] * x**2
        else:
            y_baseline = 0
        
        return y_peaks + y_baseline
    
    @staticmethod
    def calculate_area(amp, sigma):
        """Area under Gaussian"""
        return amp * sigma * np.sqrt(2 * np.pi)
    
    @staticmethod
    def calculate_fwhm(sigma):
        """Full width at half maximum"""
        return 2 * np.sqrt(2 * np.log(2)) * sigma


# ============================================================================
# Fit Quality Analyzer (from second code)
# ============================================================================

class FitQualityAnalyzer:
    """Fit quality analysis"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, n_params):
        """Calculate quality metrics"""
        residuals = y_true - y_pred
        n = len(y_true)
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        rss = ss_res
        aic = n * np.log(rss/n) + 2 * n_params if rss > 0 else -np.inf
        bic = n * np.log(rss/n) + n_params * np.log(n) if rss > 0 else -np.inf
        
        chi_squared = rss / (n - n_params) if n > n_params else np.inf
        max_error = np.max(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        
        return {
            'R²': r_squared,
            'AIC': aic,
            'BIC': bic,
            'χ²': chi_squared,
            'Max Error': max_error,
            'RMSE': rmse,
            'Residuals': residuals
        }


# ============================================================================
# Gaussian Fitter (from second code)
# ============================================================================

class GaussianFitter:
    """Handles Gaussian fitting with multiple optimization methods and baseline"""
    
    def __init__(self, method='trf', max_nfev=5000, baseline_method='none', 
                 fit_quality='balanced', last_popt=None):
        self.method = method
        self.max_nfev = max_nfev
        self.baseline_method = baseline_method
        self.fit_quality = fit_quality
        self.last_popt = last_popt
        self.convergence_history = []
        self.fit_progress = 0
        
        if fit_quality == 'fast':
            self.xtol = 1e-3
            self.ftol = 1e-3
            self.gtol = 1e-3
        elif fit_quality == 'balanced':
            self.xtol = 1e-5
            self.ftol = 1e-5
            self.gtol = 1e-5
        else:
            self.xtol = 1e-8
            self.ftol = 1e-8
            self.gtol = 1e-8
    
    def get_n_baseline_params(self):
        """Get number of baseline parameters"""
        return {
            'none': 0,
            'constant': 1,
            'linear': 2,
            'quadratic': 3
        }.get(self.baseline_method, 0)
    
    def fit(self, x, y_norm, initial_peak_params, y_max, 
            progress_callback=None, fixed_params=None):
        """Perform fitting with progress tracking"""
        n_peaks = len(initial_peak_params) // 3
        n_baseline = self.get_n_baseline_params()
        
        if self.last_popt is not None:
            expected_len = n_peaks * 3 + n_baseline
            if len(self.last_popt) == expected_len:
                initial_params = self.last_popt.copy()
                if progress_callback:
                    progress_callback(0.1, "Using cached parameters...")
            else:
                initial_params = np.array(initial_peak_params)
                if n_baseline > 0:
                    if self.baseline_method == 'constant':
                        baseline_init = [np.percentile(y_norm, 5)]
                    elif self.baseline_method == 'linear':
                        baseline_init = [np.percentile(y_norm, 5), 0]
                    else:
                        baseline_init = [np.percentile(y_norm, 5), 0, 0]
                    initial_params = np.concatenate([initial_params, baseline_init])
        else:
            initial_params = np.array(initial_peak_params)
            if n_baseline > 0:
                if self.baseline_method == 'constant':
                    baseline_init = [np.percentile(y_norm, 5)]
                elif self.baseline_method == 'linear':
                    baseline_init = [np.percentile(y_norm, 5), 0]
                else:
                    baseline_init = [np.percentile(y_norm, 5), 0, 0]
                initial_params = np.concatenate([initial_params, baseline_init])
        
        if len(initial_params) == 0:
            return False, None, None, None
        
        lower_bounds, upper_bounds = self._create_bounds(x, y_norm, n_peaks, n_baseline)
        
        for i in range(len(initial_params)):
            initial_params[i] = np.clip(initial_params[i], lower_bounds[i], upper_bounds[i])
        
        try:
            if progress_callback:
                progress_callback(0.3, "Initializing fit...")
            
            def model_func(x, *params):
                return multi_gaussian_with_baseline_flat(
                    x, *params, n_peaks=n_peaks, baseline_method=self.baseline_method
                )
            
            popt, pcov = curve_fit(
                model_func,
                x,
                y_norm,
                p0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                method=self.method,
                maxfev=self.max_nfev,
                xtol=self.xtol,
                ftol=self.ftol,
                gtol=self.gtol
            )
            
            if progress_callback:
                progress_callback(0.8, "Calculating components...")
            
            fit_y_norm = model_func(x, *popt)
            
            peak_params = popt[:n_peaks*3]
            baseline_params = popt[n_peaks*3:] if n_baseline > 0 else []
            
            components = []
            for i in range(n_peaks):
                amp_norm = peak_params[3*i]
                cen = peak_params[3*i + 1]
                sigma = abs(peak_params[3*i + 2])
                
                amp = amp_norm * y_max
                area = GaussianModelDeconv.calculate_area(amp_norm, sigma) * y_max
                
                component_y_norm = GaussianModelDeconv.gaussian(x, amp_norm, cen, sigma)
                
                cen_linear = 10**cen if np.any(x < 0) else cen
                
                components.append({
                    'id': i + 1,
                    'amp_norm': amp_norm,
                    'amp': amp,
                    'cen_log': cen,
                    'cen_linear': cen_linear,
                    'sigma_log': sigma,
                    'fwhm': GaussianModelDeconv.calculate_fwhm(sigma),
                    'area': area,
                    'fraction': 0,
                    'y_norm': component_y_norm,
                    'source': 'auto'
                })
            
            total_area = sum([c['area'] for c in components])
            for c in components:
                c['fraction'] = c['area'] / total_area if total_area > 0 else 0
                c['fraction_percent'] = c['fraction'] * 100
            
            if progress_callback:
                progress_callback(1.0, "Fit complete!")
            
            return True, popt, components, baseline_params
            
        except Exception as e:
            if progress_callback:
                progress_callback(1.0, f"Fit failed: {e}")
            return False, None, None, None
    
    def _create_bounds(self, x, y_norm, n_peaks, n_baseline):
        """Create bounds for fitting"""
        lower_bounds = []
        upper_bounds = []
        x_range = np.max(x) - np.min(x)
        
        for i in range(n_peaks):
            lower_bounds.extend([0, np.min(x), x_range * 0.001])
            upper_bounds.extend([2 * np.max(y_norm), np.max(x), x_range * 0.5])
        
        if n_baseline >= 1:
            lower_bounds.append(-np.max(y_norm))
            upper_bounds.append(np.max(y_norm))
        if n_baseline >= 2:
            lower_bounds.append(-x_range)
            upper_bounds.append(x_range)
        if n_baseline >= 3:
            lower_bounds.append(-x_range**2)
            upper_bounds.append(x_range**2)
        
        return lower_bounds, upper_bounds
    
    def preview_fit(self, x, peak_params, y_max, baseline_params=None):
        """Preview fit without optimization (fast)"""
        n_peaks = len(peak_params) // 3
        n_baseline = self.get_n_baseline_params()
        
        if baseline_params is None and n_baseline > 0:
            if self.baseline_method == 'constant':
                baseline_params = [0]
            elif self.baseline_method == 'linear':
                baseline_params = [0, 0]
            else:
                baseline_params = [0, 0, 0]
        
        fit_y_norm = multi_gaussian_with_baseline_flat(
            x, *peak_params, *baseline_params if baseline_params else [], 
            n_peaks=n_peaks, baseline_method=self.baseline_method
        )
        
        return fit_y_norm


# Add flat version for curve_fit compatibility
def multi_gaussian_with_baseline_flat(x, *params, n_peaks, baseline_method):
    """Flat version for curve_fit"""
    if baseline_method == "none":
        return GaussianModelDeconv.multi_gaussian(x, *params)
    
    n_baseline_params = {
        'none': 0,
        'constant': 1,
        'linear': 2,
        'quadratic': 3
    }.get(baseline_method, 0)
    
    peak_params = params[:n_peaks*3]
    baseline_params = params[n_peaks*3:] if n_baseline_params > 0 else []
    
    return GaussianModelDeconv.multi_gaussian_with_baseline(
        x, n_peaks, peak_params, baseline_params, baseline_method
    )

GaussianModelDeconv.multi_gaussian_with_baseline_flat = multi_gaussian_with_baseline_flat


# ============================================================================
# Derivative Analyzer (from second code)
# ============================================================================

class DerivativeAnalyzer:
    """Analysis of first and second derivatives for peak detection"""
    
    @staticmethod
    def calculate_derivatives(x, y, window_length=11, polyorder=3):
        """Calculate smoothed derivatives with fallback for small datasets"""
        if len(x) < window_length:
            window_length = len(x) if len(x) % 2 == 1 else len(x) - 1
        
        if window_length < polyorder + 2:
            dy = np.gradient(y, x)
            d2y = np.gradient(dy, x)
            return dy, d2y, y
        
        try:
            y_smooth = savgol_filter(y, window_length, polyorder)
            dy = savgol_filter(y, window_length, polyorder, deriv=1, delta=np.mean(np.diff(x)))
            d2y = savgol_filter(y, window_length, polyorder, deriv=2, delta=np.mean(np.diff(x)))
        except Exception as e:
            warnings.warn(f"Savitzky-Golay failed, using simple gradient: {e}")
            y_smooth = y
            dy = np.gradient(y, x)
            d2y = np.gradient(dy, x)
        
        return dy, d2y, y_smooth
    
    @staticmethod
    def find_peaks_by_derivatives(x, y, dy, d2y, threshold=0.01):
        """Find peaks by zero crossing of first derivative and negative second derivative"""
        peaks = []
        for i in range(1, len(x) - 1):
            if (dy[i-1] > 0 and dy[i] <= 0) or (dy[i-1] >= 0 and dy[i] < 0):
                if d2y[i] < 0:
                    if y[i] > threshold * np.max(y):
                        peaks.append(i)
        return peaks


# ============================================================================
# Data Preprocessor (from second code)
# ============================================================================

class DataPreprocessor:
    """Handles data preprocessing including clipping and log transformations"""
    
    def __init__(self, clip_negative=True, show_warnings=True):
        self.clip_negative = clip_negative
        self.show_warnings = show_warnings
        self.clipped_points = 0
        self.small_values_warning = False
    
    def smooth_data(self, x, y, method='savgol', level='none', x_log=False):
        """Smooth data with various methods and levels"""
        if level == 'none' or len(y) < 5:
            return y
        
        n_points = len(y)
        if level == 'light':
            window = min(5, n_points - 1 if n_points % 2 == 0 else n_points)
        elif level == 'medium':
            window = min(11, n_points - 1 if n_points % 2 == 0 else n_points)
        elif level == 'strong':
            window = min(21, n_points - 1 if n_points % 2 == 0 else n_points)
        elif level == 'adaptive':
            noise_estimate = np.std(np.diff(y)) / np.mean(np.abs(y)) if np.mean(np.abs(y)) > 0 else 1
            if noise_estimate > 0.5:
                window = min(21, n_points - 1 if n_points % 2 == 0 else n_points)
            elif noise_estimate > 0.2:
                window = min(11, n_points - 1 if n_points % 2 == 0 else n_points)
            else:
                window = min(5, n_points - 1 if n_points % 2 == 0 else n_points)
        else:
            return y
        
        if window % 2 == 0:
            window += 1
        
        try:
            if method == 'savgol':
                polyorder = min(3, window - 1)
                return savgol_filter(y, window, polyorder)
            elif method == 'gaussian':
                sigma = window / 5
                return gaussian_filter1d(y, sigma)
        except Exception as e:
            if self.show_warnings:
                warnings.warn(f"Smoothing failed: {e}")
            return y
    
    def preprocess_for_fitting(self, x_linear, y_original, use_log_x, use_log_y, smoothing_level='none'):
        """Preprocess data for fitting with proper handling of edge cases"""
        # Sort by X to ensure monotonic increasing X
        sort_idx = np.argsort(x_linear)
        x_sorted = x_linear[sort_idx]
        y_sorted = y_original[sort_idx]
        
        # Handle negative values
        if self.clip_negative:
            negative_mask = y_sorted < 0
            self.clipped_points = np.sum(negative_mask)
            if self.clipped_points > 0 and self.show_warnings:
                warnings.warn(f"Clipped {self.clipped_points} negative values to 0")
            y_for_fitting = np.maximum(y_sorted, 0)
        else:
            y_for_fitting = y_sorted
        
        # Apply smoothing if requested
        if smoothing_level != 'none':
            y_for_fitting = self.smooth_data(x_sorted, y_for_fitting, 'savgol', smoothing_level, use_log_x)
        
        # Small epsilon for log transformations
        eps = np.finfo(float).eps
        
        # Check for very small values when using log
        if use_log_y and np.any(y_for_fitting < eps * 100):
            self.small_values_warning = True
            if self.show_warnings:
                warnings.warn("Very small Y values detected. Log transformation may cause artifacts.")
        
        # Apply logarithmic transformations
        if use_log_x:
            x_pos = np.maximum(x_sorted, eps)
            x = np.log10(x_pos)
            x_label = 'log₁₀(τ)'
        else:
            x = x_sorted
            x_label = 'τ (s)'
        
        if use_log_y:
            y_pos = np.maximum(y_for_fitting, eps)
            y = np.log10(y_pos)
            y_label = 'log₁₀(γ(τ))'
        else:
            y = y_for_fitting
            y_label = 'γ(τ) (Ω)'
        
        return {
            'x_sorted': x_sorted,
            'y_sorted': y_sorted,
            'x': x,
            'y': y,
            'y_for_fitting': y_for_fitting,
            'x_label': x_label,
            'y_label': y_label,
            'clipped_points': self.clipped_points,
            'small_values_warning': self.small_values_warning
        }


# ============================================================================
# Gaussian Deconvolver (from second code)
# ============================================================================
class GaussianDeconvolver:
    """Main class for spectral deconvolution with baseline correction"""
    
    def __init__(self, x_linear, y_original, use_log_x=True, use_log_y=False,
                 clip_negative=True, show_warnings=True, baseline_method='none',
                 smoothing_level='none'):
        # Store original data WITHOUT ANY MODIFICATIONS for display purposes
        self.x_original = np.array(x_linear).copy()
        self.y_original_raw = np.array(y_original).copy()
        
        # Working arrays that may be modified
        self.x_linear = np.array(x_linear)
        self.y_original = np.array(y_original)
        self.use_log_x = use_log_x
        self.use_log_y = use_log_y
        self.baseline_method = baseline_method
        self.smoothing_level = smoothing_level
        
        # Sort by X to ensure monotonic increasing X
        sort_idx = np.argsort(self.x_linear)
        self.x_linear = self.x_linear[sort_idx]
        self.y_original = self.y_original[sort_idx]
        
        # Store sorted original data for display
        self.x_sorted = self.x_linear.copy()
        self.y_sorted = self.y_original.copy()
        
        # Preprocess data
        self.preprocessor = DataPreprocessor(clip_negative, show_warnings)
        preprocessed = self.preprocessor.preprocess_for_fitting(
            self.x_linear, self.y_original, use_log_x, use_log_y, smoothing_level
        )
        
        # Update with preprocessed data
        self.x_sorted = preprocessed['x_sorted']
        self.y_sorted = preprocessed['y_sorted']
        self.x = preprocessed['x']
        self.y = preprocessed['y']
        self.y_for_fitting = preprocessed['y_for_fitting']
        self.x_label = preprocessed['x_label']
        self.y_label = preprocessed['y_label']
        self.clipped_points = preprocessed['clipped_points']
        self.small_values_warning = preprocessed['small_values_warning']
        
        # Normalization - use 95th percentile instead of max for robustness
        self.y_max = np.percentile(self.y_for_fitting, 95) if np.any(self.y_for_fitting > 0) else 1.0
        
        # For fitting, we normalize but keep track for denormalization
        if self.y_max > 0:
            self.y_norm = self.y / self.y_max
        else:
            self.y_norm = self.y
        
        # Results containers
        self.components = []
        self.fit_y_norm = None
        self.popt = None
        self.baseline_params = None
        self.quality_metrics = {}
        self.convergence_history = []
        self.total_area = 0
        
        # Fitter
        self.fitter = None
        
        # For compatibility with existing code
        self.multi_gaussian = GaussianModel.multi_gaussian
        self.gaussian = GaussianModel.gaussian
    
    def _get_n_baseline_params(self):
        """Get number of baseline parameters"""
        return {
            'none': 0,
            'constant': 1,
            'linear': 2,
            'quadratic': 3
        }.get(self.baseline_method, 0)
    
    def _prepare_initial_params(self, peak_params, n_baseline):
        """Prepare initial parameters with baseline if needed"""
        params = list(peak_params)
        if n_baseline > 0:
            if self.baseline_method == 'constant':
                baseline_init = [np.percentile(self.y_norm, 5)]
            elif self.baseline_method == 'linear':
                baseline_init = [np.percentile(self.y_norm, 5), 0]
            else:  # quadratic
                baseline_init = [np.percentile(self.y_norm, 5), 0, 0]
            params.extend(baseline_init[:n_baseline])
        return params
    
    def auto_detect_peaks(self, sensitivity=0.03, min_distance=5):
        """Automatic peak detection using derivatives"""
        # Smoothing
        window_length = min(11, len(self.y_norm) // 5 * 2 + 1)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 5:
            y_smooth = savgol_filter(self.y_norm, window_length, 3)
        else:
            y_smooth = self.y_norm
        
        # Calculate derivatives
        dy, d2y, y_smooth = DerivativeAnalyzer.calculate_derivatives(self.x, y_smooth)
        
        # Peak search with different methods
        height_threshold = sensitivity * np.max(y_smooth)
        peaks1, _ = find_peaks(y_smooth, height=height_threshold, distance=min_distance)
        peaks2 = DerivativeAnalyzer.find_peaks_by_derivatives(self.x, y_smooth, dy, d2y, sensitivity)
        
        # Combine results
        all_peaks = sorted(set(np.concatenate([peaks1, peaks2])))
        
        # Filter close peaks
        filtered_peaks = []
        for peak in all_peaks:
            if not filtered_peaks or abs(self.x[peak] - self.x[filtered_peaks[-1]]) > min_distance * np.mean(np.diff(self.x)):
                filtered_peaks.append(peak)
        
        # Estimate parameters
        peak_info = []
        initial_params = []
        
        for peak_idx in filtered_peaks:
            cen = self.x[peak_idx]
            amp = y_smooth[peak_idx]
            
            # Estimate sigma with fallback
            sigma = GaussianModel.estimate_sigma_from_peak(self.x, y_smooth, peak_idx)
            sigma = max(sigma, 0.01 * (np.max(self.x) - np.min(self.x)) / max(len(filtered_peaks), 1))
            
            # Get original Y value for display
            if self.use_log_x:
                x_linear = 10**self.x[peak_idx]
            else:
                x_linear = self.x[peak_idx]
            
            # Find closest index in original data - always in linear space
            idx = np.argmin(np.abs(self.x_sorted - x_linear))
            y_original_value = self.y_sorted[idx]
            
            peak_info.append({
                'index': peak_idx,
                'x': self.x[peak_idx],
                'x_linear': x_linear,
                'y': self.y[peak_idx],
                'y_original': y_original_value,
                'amp_est': amp,
                'cen_est': cen,
                'sigma_est': sigma,
                'dy': dy[peak_idx],
                'd2y': d2y[peak_idx],
                'source': 'auto'
            })
            
            initial_params.extend([amp, cen, sigma])
        
        return filtered_peaks, peak_info, initial_params, (dy, d2y, y_smooth)
    
    def add_manual_peak(self, x_position_linear, amplitude=None, sigma_est=None):
        """Add a peak manually at specified linear X position"""
        # Convert to log space if needed
        if self.use_log_x:
            x_position = np.log10(x_position_linear)
        else:
            x_position = x_position_linear
        
        # Find index for amplitude estimation
        idx = np.argmin(np.abs(self.x_sorted - x_position_linear))
        
        # Estimate amplitude if not provided
        if amplitude is None:
            # Get normalized amplitude at this position
            if self.use_log_x:
                # Find closest index in log space
                log_idx = np.argmin(np.abs(self.x - x_position))
                amplitude = self.y_norm[log_idx] if log_idx < len(self.y_norm) else 0.1
            else:
                amplitude = self.y_norm[idx] if idx < len(self.y_norm) else 0.1
        
        # Estimate sigma if not provided
        if sigma_est is None:
            # Estimate based on distance to nearest minimum
            if self.use_log_x:
                x_search = self.x
                y_search = self.y_norm
            else:
                x_search = self.x_linear
                y_search = self.y_original / self.y_max
            
            # Find nearest minima to left and right
            left_idx = idx
            right_idx = idx
            for i in range(idx - 1, 0, -1):
                if i < len(y_search) - 1 and y_search[i] < y_search[i-1] and y_search[i] < y_search[i+1]:
                    left_idx = i
                    break
            for i in range(idx + 1, len(y_search) - 1):
                if y_search[i] < y_search[i-1] and y_search[i] < y_search[i+1]:
                    right_idx = i
                    break
            
            # Estimate sigma
            width = (x_search[right_idx] - x_search[left_idx]) if right_idx > left_idx else 0.1
            sigma_est = max(width / 3.0, 0.01 * (np.max(x_search) - np.min(x_search)) / 20)
        
        # Add peak info
        peak_info_entry = {
            'index': idx,
            'x': x_position,
            'x_linear': x_position_linear,
            'y': amplitude,
            'y_original': self.y_sorted[idx],
            'amp_est': amplitude,
            'cen_est': x_position,
            'sigma_est': sigma_est,
            'dy': 0,
            'd2y': 0,
            'source': 'manual'
        }
        
        return peak_info_entry, [amplitude, x_position, sigma_est]
    
    def find_missing_peaks_by_residuals(self, peak_info, sensitivity=0.02, min_distance=5):
        """Find missing peaks by analyzing residuals after initial fit"""
        if not peak_info:
            return [], []
        
        # Build initial model with current peaks
        n_peaks = len(peak_info)
        if n_peaks == 0:
            return [], []
        
        peak_params = []
        for info in peak_info:
            peak_params.extend([info['amp_est'], info['cen_est'], info['sigma_est']])
        
        # Calculate initial fit
        y_initial_fit = GaussianModel.multi_gaussian(self.x, *peak_params)
        
        # Calculate residuals
        residuals = self.y_norm - y_initial_fit
        
        # Detect peaks in residuals
        height_threshold = sensitivity * np.max(np.abs(residuals))
        
        # Smooth residuals for better peak detection
        window_length = min(11, len(residuals) // 5 * 2 + 1)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 5:
            residuals_smooth = savgol_filter(residuals, window_length, 3)
        else:
            residuals_smooth = residuals
        
        # Find positive peaks (where data exceeds fit)
        positive_peaks, _ = find_peaks(residuals_smooth, height=height_threshold, distance=min_distance)
        
        # Find negative peaks (where fit exceeds data - potential shoulders)
        negative_peaks, _ = find_peaks(-residuals_smooth, height=height_threshold, distance=min_distance)
        
        # Combine and filter
        all_candidate_indices = sorted(set(positive_peaks) | set(negative_peaks))
        
        missing_peaks = []
        missing_params = []
        
        for idx in all_candidate_indices:
            # Skip if too close to existing peaks
            too_close = False
            for info in peak_info:
                if abs(self.x[idx] - info['cen_est']) < min_distance * np.mean(np.diff(self.x)):
                    too_close = True
                    break
            
            if too_close:
                continue
            
            cen = self.x[idx]
            amp = abs(residuals_smooth[idx])
            sigma = GaussianModel.estimate_sigma_from_peak(self.x, residuals_smooth, idx)
            sigma = max(sigma, 0.01 * (np.max(self.x) - np.min(self.x)) / 20)
            
            # Get original Y value for display
            if self.use_log_x:
                x_linear = 10**cen
            else:
                x_linear = cen
            
            # Find closest index in original data
            orig_idx = np.argmin(np.abs(self.x_sorted - x_linear))
            y_original_value = self.y_sorted[orig_idx]
            
            missing_peaks.append({
                'index': idx,
                'x': cen,
                'x_linear': x_linear,
                'y': amp,
                'y_original': y_original_value,
                'amp_est': amp,
                'cen_est': cen,
                'sigma_est': sigma,
                'dy': 0,
                'd2y': 0,
                'source': 'residuals'
            })
            
            missing_params.extend([amp, cen, sigma])
        
        return missing_peaks, missing_params
    
    def _create_bounds(self, x, y_norm, n_peaks, n_baseline):
        """Create bounds for fitting"""
        lower_bounds = []
        upper_bounds = []
        x_range = np.max(x) - np.min(x)
        
        # Peak bounds
        for i in range(n_peaks):
            lower_bounds.extend([0, np.min(x), x_range * 0.001])
            upper_bounds.extend([2 * np.max(y_norm), np.max(x), x_range * 0.5])
        
        # Baseline bounds
        if n_baseline >= 1:  # constant
            lower_bounds.append(-np.max(y_norm))
            upper_bounds.append(np.max(y_norm))
        if n_baseline >= 2:  # linear term
            lower_bounds.append(-x_range)
            upper_bounds.append(x_range)
        if n_baseline >= 3:  # quadratic term
            lower_bounds.append(-x_range**2)
            upper_bounds.append(x_range**2)
        
        return lower_bounds, upper_bounds
    
    def fit(self, initial_params=None, method='trf', maxfev=5000, 
            fit_quality='balanced', last_popt=None, progress_callback=None):
        """Perform fitting with selected method and baseline"""
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks()
        
        if len(initial_params) == 0:
            if progress_callback:
                progress_callback(1.0, "No peaks detected!")
            return False
        
        n_peaks = len(initial_params) // 3
        n_baseline = self._get_n_baseline_params()
        
        # Use last good parameters if available
        if last_popt is not None:
            expected_len = n_peaks * 3 + n_baseline
            if len(last_popt) == expected_len:
                initial_params = last_popt.copy()
            else:
                initial_params = self._prepare_initial_params(initial_params, n_baseline)
        else:
            initial_params = self._prepare_initial_params(initial_params, n_baseline)
        
        # Create bounds
        lower_bounds, upper_bounds = self._create_bounds(self.x, self.y_norm, n_peaks, n_baseline)
        
        # Ensure initial_params are within bounds
        for i in range(len(initial_params)):
            initial_params[i] = np.clip(initial_params[i], lower_bounds[i], upper_bounds[i])
        
        try:
            if progress_callback:
                progress_callback(0.3, "Initializing fit...")
            
            # Define the model function for curve_fit
            def model_func(x, *params):
                if n_baseline == 0:
                    return GaussianModel.multi_gaussian(x, *params)
                else:
                    peak_params = params[:n_peaks*3]
                    baseline_params = params[n_peaks*3:]
                    return GaussianModel.multi_gaussian_with_baseline(
                        x, n_peaks, peak_params, baseline_params, self.baseline_method
                    )
            
            # Set tolerances based on fit quality
            if fit_quality == 'fast':
                xtol, ftol, gtol = 1e-3, 1e-3, 1e-3
                maxfev = min(maxfev, 2000)
            elif fit_quality == 'balanced':
                xtol, ftol, gtol = 1e-5, 1e-5, 1e-5
                maxfev = min(maxfev, 5000)
            else:  # precise
                xtol, ftol, gtol = 1e-8, 1e-8, 1e-8
                maxfev = min(maxfev, 10000)
            
            if progress_callback:
                progress_callback(0.5, "Running curve_fit...")
            
            # Perform fit
            popt, pcov = curve_fit(
                model_func,
                self.x,
                self.y_norm,
                p0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                method=method,
                maxfev=maxfev,
                xtol=xtol,
                ftol=ftol,
                gtol=gtol
            )
            
            if progress_callback:
                progress_callback(0.8, "Calculating components...")
            
            # Calculate fit
            fit_y_norm = model_func(self.x, *popt)
            
            # Extract components
            components = []
            peak_params = popt[:n_peaks*3]
            baseline_params = popt[n_peaks*3:] if n_baseline > 0 else []
            
            for i in range(n_peaks):
                amp_norm = peak_params[3*i]
                cen = peak_params[3*i + 1]
                sigma = abs(peak_params[3*i + 2])
                
                amp = amp_norm * self.y_max
                # Используем правильный метод расчета площади
                area = GaussianModelDeconv.calculate_area(amp_norm, sigma) * self.y_max
                
                component_y_norm = GaussianModelDeconv.gaussian(self.x, amp_norm, cen, sigma)
                
                # Calculate center in linear space
                if self.use_log_x:
                    cen_linear = 10**cen
                else:
                    cen_linear = cen
                
                components.append({
                    'id': i + 1,
                    'amp_norm': amp_norm,
                    'amp': amp,
                    'cen_log': cen,
                    'cen_linear': cen_linear,
                    'sigma_log': sigma,
                    'fwhm': GaussianModel.calculate_fwhm(sigma),
                    'area': area,
                    'fraction': 0,
                    'y_norm': component_y_norm,
                    'source': 'auto'
                })
            
            # Calculate fractions
            total_area = sum([c['area'] for c in components])
            for c in components:
                c['fraction'] = c['area'] / total_area if total_area > 0 else 0
                c['fraction_percent'] = c['fraction'] * 100
            
            # Store results
            self.popt = popt
            self.components = components
            self.baseline_params = baseline_params
            self.fit_y_norm = fit_y_norm
            self.total_area = total_area
            
            # Calculate quality metrics
            self.quality_metrics = FitQualityAnalyzer.calculate_metrics(
                self.y_norm, self.fit_y_norm, len(popt)
            )
            
            if progress_callback:
                progress_callback(1.0, "Fit complete!")
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(1.0, f"Fit failed: {e}")
            print(f"Error in fit: {e}")  # For debugging
            return False
    
    def preview_fit(self, initial_params=None):
        """Preview fit without optimization (fast)"""
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks()
        
        if len(initial_params) == 0:
            return None
        
        n_peaks = len(initial_params) // 3
        n_baseline = self._get_n_baseline_params()
        
        # Prepare parameters with baseline
        full_params = self._prepare_initial_params(initial_params, n_baseline)
        
        # Calculate fit
        if n_baseline == 0:
            fit_y_norm = GaussianModel.multi_gaussian(self.x, *full_params)
        else:
            peak_params = full_params[:n_peaks*3]
            baseline_params = full_params[n_peaks*3:]
            fit_y_norm = GaussianModel.multi_gaussian_with_baseline(
                self.x, n_peaks, peak_params, baseline_params, self.baseline_method
            )
        
        return fit_y_norm
    
    def remove_peak(self, peak_id):
        """Remove a peak (does NOT perform fit, just marks for removal)"""
        if peak_id > len(self.components):
            return False
        
        # Store the operation
        st.session_state.app_state.pending_remove = peak_id
        return True
    
    def split_peak(self, peak_id, split_position):
        """Split a peak into two (does NOT perform fit, just marks for splitting)"""
        if peak_id > len(self.components):
            return False
        
        # Store the operation
        st.session_state.app_state.pending_split = (peak_id, split_position)
        return True
    
    def apply_pending_operations(self, fit_quality='balanced', progress_callback=None):
        """Apply all pending operations and perform fit"""
        # Get current parameters
        if self.components:
            current_params = []
            for c in self.components:
                current_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
        else:
            return False
        
        # Apply pending remove
        if st.session_state.app_state.pending_remove is not None:
            remove_id = st.session_state.app_state.pending_remove
            new_params = []
            for i, c in enumerate(self.components):
                if i != remove_id - 1:
                    new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
            current_params = new_params
            st.session_state.app_state.pending_remove = None
        
        # Apply pending split
        if st.session_state.app_state.pending_split is not None:
            peak_id, split_position = st.session_state.app_state.pending_split
            peak = self.components[peak_id - 1]
            
            new_params = []
            for i, c in enumerate(self.components):
                if i == peak_id - 1:
                    amp1 = c['amp_norm'] * 0.6
                    amp2 = c['amp_norm'] * 0.4
                    
                    cen1 = split_position - c['sigma_log'] * 0.3
                    cen2 = split_position + c['sigma_log'] * 0.3
                    
                    cen1 = np.clip(cen1, np.min(self.x), np.max(self.x))
                    cen2 = np.clip(cen2, np.min(self.x), np.max(self.x))
                    
                    sigma1 = c['sigma_log'] * 0.7
                    sigma2 = c['sigma_log'] * 0.7
                    
                    new_params.extend([amp1, cen1, sigma1])
                    new_params.extend([amp2, cen2, sigma2])
                else:
                    new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
            
            current_params = new_params
            st.session_state.app_state.pending_split = None
        
        # Perform fit
        return self.fit(
            initial_params=current_params,
            method=st.session_state.app_state.fitting_method,
            maxfev=st.session_state.app_state.max_nfev,
            fit_quality=fit_quality,
            last_popt=st.session_state.app_state.last_popt,
            progress_callback=progress_callback
        )
    
    def remove_peak_by_id(self, peak_id):
        """Remove a peak by its ID from peak_info and initial_params"""
        if st.session_state.app_state.peak_info is None:
            return False
        
        # Find the peak in peak_info
        peak_to_remove = None
        for i, info in enumerate(st.session_state.app_state.peak_info):
            if info.get('id', i+1) == peak_id or i+1 == peak_id:
                peak_to_remove = i
                break
        
        if peak_to_remove is None:
            return False
        
        # Remove from peak_info
        removed_peak = st.session_state.app_state.peak_info.pop(peak_to_remove)
        
        # Remove from initial_params (3 parameters per peak)
        if st.session_state.app_state.initial_peak_params is not None:
            start_idx = peak_to_remove * 3
            del st.session_state.app_state.initial_peak_params[start_idx:start_idx + 3]
        
        # Also remove from manual_peaks or residuals_peaks if present
        if removed_peak.get('source') == 'manual':
            st.session_state.app_state.manual_peaks = [p for p in st.session_state.app_state.manual_peaks 
                                                        if p.get('x_linear') != removed_peak.get('x_linear')]
        elif removed_peak.get('source') == 'residuals':
            st.session_state.app_state.residuals_peaks = [p for p in st.session_state.app_state.residuals_peaks 
                                                           if p.get('x_linear') != removed_peak.get('x_linear')]
        
        return True
    
    def create_deconvolution_result(self) -> DeconvolutionResult:
        """Create result container from current components"""
        peaks = []
        for c in self.components:
            peak = GaussianPeak(
                id=c['id'],
                center=c['cen_linear'],
                center_log=c['cen_log'],
                amplitude=c['amp'],
                amplitude_norm=c['amp_norm'],
                sigma_log=c['sigma_log'],
                fwhm=c['fwhm'],
                area=c['area'],
                fraction=c['fraction'],
                fraction_percent=c['fraction_percent'],
                source=c.get('source', 'auto'),
                y_norm=c['y_norm']
            )
            peaks.append(peak)
        
        # Восстанавливаем оригинальный масштаб y_original
        y_original_restored = self.y_original * self.y_max if self.y_max > 0 else self.y_original
        
        return DeconvolutionResult(
            peaks=peaks,
            fit_y_norm=self.fit_y_norm if self.fit_y_norm is not None else np.zeros_like(self.x),
            x=self.x,
            y_norm=self.y_norm,
            y_original=y_original_restored,
            x_linear=self.x_linear,
            use_log_x=self.use_log_x,
            use_log_y=self.use_log_y,
            quality_metrics=self.quality_metrics,
            baseline_params=self.baseline_params,
            baseline_method=self.baseline_method,
            total_area=self.total_area,
            max_amplitude=max([c['amp'] for c in self.components]) if self.components else 0
        )


# ============================================================================
# Visualization Functions (from both codes)
# ============================================================================

def plot_nyquist_matplotlib(data: ImpedanceData, re_rec: Optional[np.ndarray] = None, 
                           im_rec: Optional[np.ndarray] = None, title: str = "Nyquist Plot",
                           highlight_idx: Optional[int] = None) -> plt.Figure:
    """Create publication-quality Nyquist plot"""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    ax.plot(data.re_z, data.im_z, 'o-', markersize=5, linewidth=1.8, 
            label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.8)
    
    if highlight_idx is not None and 0 <= highlight_idx < data.n_points:
        ax.plot(data.re_z[highlight_idx], data.im_z[highlight_idx], 'ro', 
                markersize=12, markeredgecolor='red', markerfacecolor='none', linewidth=2.5,
                label='Selected Point')
    
    if re_rec is not None and im_rec is not None:
        ax.plot(re_rec, im_rec, 's-', markersize=4, linewidth=1.2,
                label='Reconstructed', color='#ff7f0e', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    
    ax.set_xlabel("Re(Z) / Ohm", fontweight='bold', fontsize=14)
    ax.set_ylabel("-Im(Z) / Ohm", fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='black', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Scientific formatting
    ax.ticklabel_format(style='scientific', scilimits=(-2, 2))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    ax.set_aspect('equal', adjustable='box')
    
    return fig


def plot_bode_matplotlib(data: ImpedanceData, re_rec: Optional[np.ndarray] = None, 
                         im_rec: Optional[np.ndarray] = None,
                         highlight_idx: Optional[int] = None) -> plt.Figure:
    """Create publication-quality Bode plot with highlight point"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    
    mag = data.Z_mod
    ax1.loglog(data.freq, mag, 'o-', markersize=5, linewidth=1.8, 
               label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.8)
    
    if highlight_idx is not None and 0 <= highlight_idx < data.n_points:
        ax1.loglog(data.freq[highlight_idx], mag[highlight_idx], 'ro', 
                   markersize=12, markeredgecolor='red', markerfacecolor='none', linewidth=2.5)
    
    if re_rec is not None and im_rec is not None:
        mag_rec = np.sqrt(re_rec**2 + im_rec**2)
        ax1.loglog(data.freq, mag_rec, 's-', markersize=4, linewidth=1.2,
                   label='Reconstructed', color='#ff7f0e', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    
    ax1.set_xlabel("Frequency / Hz", fontweight='bold', fontsize=14)
    ax1.set_ylabel("|Z| / Ohm", fontweight='bold', fontsize=14)
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Scientific formatting for log axes
    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    phase = data.phase
    ax2.semilogx(data.freq, phase, 'o-', markersize=5, linewidth=1.8,
                 label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.8)
    
    if highlight_idx is not None and 0 <= highlight_idx < data.n_points:
        ax2.semilogx(data.freq[highlight_idx], phase[highlight_idx], 'ro', 
                     markersize=12, markeredgecolor='red', markerfacecolor='none', linewidth=2.5)
    
    if re_rec is not None and im_rec is not None:
        phase_rec = np.arctan2(im_rec, re_rec) * 180 / np.pi
        ax2.semilogx(data.freq, phase_rec, 's-', markersize=4, linewidth=1.2,
                     label='Reconstructed', color='#ff7f0e', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    
    ax2.set_xlabel("Frequency / Hz", fontweight='bold', fontsize=14)
    ax2.set_ylabel("Phase / deg", fontweight='bold', fontsize=14)
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Scientific formatting
    ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    fig.suptitle("Bode Plot", fontweight='bold', fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_drt_matplotlib(result: DRTResult, peaks: Optional[List[Dict[str, Any]]] = None,
                       title: str = "Distribution of Relaxation Times") -> plt.Figure:
    """Create publication-quality DRT plot with both tau and frequency axes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    if result.gamma_std is not None:
        ax1.fill_between(result.tau_grid, result.gamma - 2*result.gamma_std, 
                        result.gamma + 2*result.gamma_std,
                        alpha=0.3, color='gray', label='±2σ uncertainty')
    ax1.semilogx(result.tau_grid, result.gamma, '-', linewidth=2, color='#2ca02c', label='DRT')
    
    if peaks and len(peaks) > 0:
        peak_tau = [p['tau'] for p in peaks]
        peak_drt = [p['amplitude'] for p in peaks]
        ax1.plot(peak_tau, peak_drt, 'rv', markersize=8, label='Detected peaks')
        
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
    
    frequencies = 1 / (2 * np.pi * result.tau_grid)
    sort_idx = np.argsort(frequencies)[::-1]
    freqs_sorted = frequencies[sort_idx]
    gamma_sorted = result.gamma[sort_idx]
    
    if result.gamma_std is not None:
        gamma_std_sorted = result.gamma_std[sort_idx]
        ax2.fill_between(freqs_sorted, gamma_sorted - 2*gamma_std_sorted, 
                        gamma_sorted + 2*gamma_std_sorted,
                        alpha=0.3, color='gray', label='±2σ uncertainty')
    ax2.semilogx(freqs_sorted, gamma_sorted, '-', linewidth=2, color='#2ca02c', label='DRT')
    
    if peaks and len(peaks) > 0:
        peak_freqs = [p['frequency'] for p in peaks]
        peak_amplitudes = [p['amplitude'] for p in peaks]
        peak_pairs = sorted(zip(peak_freqs, peak_amplitudes), key=lambda x: x[0], reverse=True)
        peak_freqs_sorted, peak_amplitudes_sorted = zip(*peak_pairs) if peak_pairs else ([], [])
        
        ax2.plot(peak_freqs_sorted, peak_amplitudes_sorted, 'rv', markersize=8, label='Detected peaks')
        
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
    ax2.set_xscale('log')
    ax2.invert_xaxis()
    ax2.set_xlim(freqs_sorted[0], freqs_sorted[-1])
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    
    fig.suptitle(title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_deconvolution_result(deconv_result: DeconvolutionResult, show_components: bool = True,
                              show_baseline: bool = True, title: str = "Gaussian Deconvolution Result",
                              preview_mode: bool = False, preview_fit: Optional[np.ndarray] = None) -> plt.Figure:
    """Plot deconvolution result with components and baseline"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Main deconvolution
    if deconv_result.use_log_x:
        ax1.set_xscale('log')
    
    ax1.scatter(deconv_result.x_linear, deconv_result.y_original, 
                s=15, alpha=0.5, color='black', label='Data', zorder=1)
    
    x_dense = np.linspace(np.min(deconv_result.x_linear), np.max(deconv_result.x_linear), 2000)
    if deconv_result.use_log_x:
        x_dense_log = np.log10(x_dense)
    else:
        x_dense_log = x_dense
    
    if show_components and deconv_result.peaks:
        colors = plt.cm.Set3(np.linspace(0, 1, len(deconv_result.peaks)))
        for peak, color in zip(deconv_result.peaks, colors):
            y_component = GaussianModelDeconv.gaussian(x_dense_log, peak.amplitude_norm, 
                                                       peak.center_log, peak.sigma_log) * deconv_result.y_original.max()
            ax1.fill_between(x_dense, 0, y_component, color=color, alpha=0.3, linewidth=0)
            ax1.plot(x_dense, y_component, '-', color=color, linewidth=2,
                    label=f'Peak {peak.id}: {peak.fraction_percent:.1f}%', zorder=2)
    
    if show_baseline and deconv_result.baseline_params and deconv_result.baseline_method != 'none':
        if deconv_result.baseline_method == 'constant':
            y_baseline = deconv_result.baseline_params[0] * deconv_result.y_original.max()
            ax1.axhline(y=y_baseline, color='gray', linestyle=':', linewidth=1.5, label='Baseline', zorder=1)
        elif deconv_result.baseline_method == 'linear':
            y_baseline = (deconv_result.baseline_params[0] + 
                         deconv_result.baseline_params[1] * x_dense_log) * deconv_result.y_original.max()
            ax1.plot(x_dense, y_baseline, 'gray', linestyle=':', linewidth=1.5, label='Baseline', zorder=1)
        elif deconv_result.baseline_method == 'quadratic':
            y_baseline = (deconv_result.baseline_params[0] + 
                         deconv_result.baseline_params[1] * x_dense_log +
                         deconv_result.baseline_params[2] * x_dense_log**2) * deconv_result.y_original.max()
            ax1.plot(x_dense, y_baseline, 'gray', linestyle=':', linewidth=1.5, label='Baseline', zorder=1)
    
    if preview_mode and preview_fit is not None:
        y_total = preview_fit * deconv_result.y_original.max()
        ax1.plot(x_dense, y_total, 'b--', linewidth=2, label='Preview (no fit)', zorder=3, alpha=0.7)
    elif deconv_result.fit_y_norm is not None:
        n_peaks = len(deconv_result.peaks)
        peak_params = []
        for peak in deconv_result.peaks:
            peak_params.extend([peak.amplitude_norm, peak.center_log, peak.sigma_log])
        
        y_total = GaussianModelDeconv.multi_gaussian_with_baseline(
            x_dense_log, n_peaks, peak_params, 
            deconv_result.baseline_params or [], deconv_result.baseline_method
        ) * deconv_result.y_original.max()
        
        ax1.plot(x_dense, y_total, 'r--', linewidth=2, label='Total Fit', zorder=3)
    
    ax1.set_xlabel('X' + (' (log scale)' if deconv_result.use_log_x else ''), fontweight='bold')
    ax1.set_ylabel('Intensity', fontweight='bold')
    ax1.set_title(title, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, frameon=True, edgecolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right plot: Area distribution
    if deconv_result.peaks:
        peaks_ids = [f'Peak {p.id}' for p in deconv_result.peaks]
        fractions = [p.fraction_percent for p in deconv_result.peaks]
        colors = plt.cm.Set3(np.linspace(0, 1, len(deconv_result.peaks)))
        
        bars = ax2.bar(peaks_ids, fractions, color=colors, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Peak', fontweight='bold')
        ax2.set_ylabel('Fraction (%)', fontweight='bold')
        ax2.set_title('Peak Area Distribution', fontweight='bold')
        ax2.set_ylim(0, max(fractions) * 1.2 if fractions else 100)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, frac in zip(bars, fractions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{frac:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.tick_params(axis='x', rotation=45)
    
    # Add quality metrics text
    if deconv_result.quality_metrics and not preview_mode:
        metrics_text = f"R² = {deconv_result.quality_metrics.get('R²', 0):.4f}\n"
        metrics_text += f"RMSE = {deconv_result.quality_metrics.get('RMSE', 0):.2e}"
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    return fig


def plot_deconvolution_components_comparison(deconv_result: DeconvolutionResult) -> plt.Figure:
    """Plot normalized components comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if deconv_result.use_log_x:
        ax.set_xscale('log')
    
    x_dense = np.linspace(np.min(deconv_result.x_linear), np.max(deconv_result.x_linear), 2000)
    if deconv_result.use_log_x:
        x_dense_log = np.log10(x_dense)
    else:
        x_dense_log = x_dense
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(deconv_result.peaks)))
    for peak, color in zip(deconv_result.peaks, colors):
        y_component = GaussianModelDeconv.gaussian(x_dense_log, peak.amplitude_norm, 
                                                   peak.center_log, peak.sigma_log)
        y_component_norm = y_component / max(peak.amplitude_norm, 1e-10)
        
        ax.plot(x_dense, y_component_norm, '-', color=color, linewidth=2,
               label=f'Peak {peak.id} (center: {peak.center:.3e})')
        ax.axvline(x=peak.center, color=color, linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('X' + (' (log scale)' if deconv_result.use_log_x else ''), fontweight='bold')
    ax.set_ylabel('Normalized Intensity', fontweight='bold')
    ax.set_title('Normalized Components Comparison', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


# ============================================================================
# Step 1: Data Loading and Preprocessing
# ============================================================================

def step1_data_loading():
    """Step 1: Load and preprocess impedance data"""
    st.header("📁 Step 1: Data Loading and Preprocessing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_method = st.radio("Select input method:", ["Upload File", "Manual Entry"], horizontal=True)
        
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
                    
                    if st.button("📥 Load Data", type="primary", use_container_width=True):
                        freq, re_z, im_z = load_data(uploaded_file, col_freq, col_re, col_im)
                        if freq is not None:
                            st.session_state.app_state.impedance_data = ImpedanceData(freq, re_z, im_z)
                            st.session_state.app_state.data_loaded = True
                            # Initialize selection state
                            if 'selected_point_idx' not in st.session_state:
                                st.session_state.selected_point_idx = 0
                            if 'freq_range_idx' not in st.session_state:
                                st.session_state.freq_range_idx = (0, len(freq) - 1)
                            st.success(f"✅ Loaded {len(freq)} data points")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        else:
            freq, re_z, im_z = manual_data_entry()
            if freq is not None:
                st.session_state.app_state.impedance_data = ImpedanceData(freq, re_z, im_z)
                st.session_state.app_state.data_loaded = True
                if 'selected_point_idx' not in st.session_state:
                    st.session_state.selected_point_idx = 0
                if 'freq_range_idx' not in st.session_state:
                    st.session_state.freq_range_idx = (0, len(freq) - 1)
                st.success(f"✅ Loaded {len(freq)} data points")
                st.rerun()
    
    with col2:
        st.subheader("📊 Data Format")
        st.info("""
        **Expected format:**
        - Frequency (Hz)
        - Re(Z) (Ω)
        - -Im(Z) (Ω)
        
        **Supported separators:**
        - Space
        - Comma
        - Tab
        """)
    
    if st.session_state.app_state.data_loaded and st.session_state.app_state.impedance_data:
        st.markdown("---")
        st.subheader("📈 Data Preview")
        
        data = st.session_state.app_state.impedance_data
        
        # Initialize session state variables if not exists
        if 'selected_point_idx' not in st.session_state:
            st.session_state.selected_point_idx = 0
        if 'freq_range_idx' not in st.session_state:
            st.session_state.freq_range_idx = (0, data.n_points - 1)
        
        # Compact Data Preprocessing section
        with st.container():
            st.markdown("**✂️ Data Preprocessing**")
            
            col_a, col_b, col_c = st.columns([2, 1.5, 1.5])
            
            with col_a:
                # Frequency range by point indices
                f_min_idx, f_max_idx = st.slider(
                    "Frequency range (by point index)",
                    min_value=0,
                    max_value=data.n_points - 1,
                    value=st.session_state.freq_range_idx,
                    step=1,
                    key="freq_range_slider"
                )
                
                # Update session state
                if (f_min_idx, f_max_idx) != st.session_state.freq_range_idx:
                    st.session_state.freq_range_idx = (f_min_idx, f_max_idx)
                
                # Display actual frequency values
                f_min_actual = data.freq[f_min_idx]
                f_max_actual = data.freq[f_max_idx]
                st.caption(f"Range: {f_min_actual:.2e} Hz - {f_max_actual:.2e} Hz")
            
            with col_b:
                # Point selection for removal
                point_idx = st.slider(
                    "Point index to remove",
                    min_value=0,
                    max_value=data.n_points - 1,
                    value=min(st.session_state.selected_point_idx, data.n_points - 1),
                    step=1,
                    key="point_selector_step1"
                )
                st.session_state.selected_point_idx = point_idx
                
                # Display point info
                if point_idx < data.n_points:
                    st.caption(f"f = {data.freq[point_idx]:.2e} Hz")
                    st.caption(f"Z = {data.Z_mod[point_idx]:.2e} Ω")
            
            with col_c:
                st.write("")
                st.write("")
                if st.button("🗑️ Remove Point", use_container_width=True, key="remove_point_btn"):
                    if point_idx < data.n_points:
                        data.remove_point(point_idx)
                        # Adjust indices after removal
                        new_max = data.n_points - 1
                        st.session_state.selected_point_idx = min(point_idx, new_max)
                        st.session_state.freq_range_idx = (min(st.session_state.freq_range_idx[0], new_max),
                                                           min(st.session_state.freq_range_idx[1], new_max))
                        st.success(f"Removed point {point_idx}")
                        st.rerun()
                
                if st.button("🔄 Reset All", use_container_width=True, key="reset_data_btn"):
                    data.reset()
                    st.session_state.selected_point_idx = 0
                    st.session_state.freq_range_idx = (0, data.n_points - 1)
                    st.success("Data reset to original")
                    st.rerun()
            
            # Apply range button
            col_d, col_e = st.columns([1, 3])
            with col_d:
                if st.button("📊 Apply Range", type="primary", use_container_width=True):
                    f_min = data.freq[f_min_idx]
                    f_max = data.freq[f_max_idx]
                    data.apply_frequency_range(f_min, f_max)
                    st.session_state.selected_point_idx = 0
                    st.session_state.freq_range_idx = (0, data.n_points - 1)
                    st.success(f"Applied range: {f_min:.2e} - {f_max:.2e} Hz")
                    st.rerun()
            
            st.caption(f"📊 Points: {data.n_points} / {len(data.original_freq)}")
        
        st.markdown("---")
        
        # Create plots with caching to prevent rerender on slider move
        @st.cache_data(ttl=60, show_spinner=False)
        def cached_plots(freq_tuple, re_tuple, im_tuple, point_idx, n_points, original_len):
            """Cache plots to prevent rerender on slider movement"""
            temp_data = ImpedanceData(np.array(freq_tuple), np.array(re_tuple), np.array(im_tuple))
            fig_nyquist = plot_nyquist_matplotlib(temp_data, highlight_idx=point_idx)
            fig_bode = plot_bode_matplotlib(temp_data, highlight_idx=point_idx)
            return fig_nyquist, fig_bode
        
        # Convert data to tuples for caching
        freq_tuple = tuple(data.freq)
        re_tuple = tuple(data.re_z)
        im_tuple = tuple(data.im_z)
        
        # Get cached plots
        fig_nyquist, fig_bode = cached_plots(
            freq_tuple, re_tuple, im_tuple, 
            st.session_state.selected_point_idx,
            data.n_points, len(data.original_freq)
        )
        
        # Display plots
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_nyquist)
        with col2:
            st.pyplot(fig_bode)
        plt.close('all')
        
        # Navigation buttons
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.app_state.current_step = 1
                st.rerun()
        with col_next:
            if st.button("Next: DRT Analysis ➡️", type="primary", use_container_width=True):
                st.session_state.app_state.current_step = 2
                st.rerun()


# ============================================================================
# Step 2: DRT Analysis
# ============================================================================

def step2_drt_analysis():
    """Step 2: Select DRT method and perform analysis"""
    st.header("⚡ Step 2: DRT Analysis")
    
    data = st.session_state.app_state.impedance_data
    
    if data is None:
        st.error("No data loaded. Please go back to Step 1.")
        if st.button("⬅️ Back to Step 1"):
            st.session_state.app_state.current_step = 1
            st.rerun()
        return
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("⚙️ Analysis Parameters")
        
        analysis_method = st.selectbox(
            "DRT Calculation Method",
            ["Tikhonov Regularization (NNLS)",
             "Bayesian MCMC",
             "Maximum Entropy (auto-λ)",
             "Finite Gaussian Process (fGP-DRT)",
             "Loewner Framework (RLF)",
             "Generalized DRT (with inductive loops)"]
        )
        
        n_tau = st.slider("Number of time points", 50, 300, 150)
        
        include_inductive = False
        if analysis_method == "Generalized DRT (with inductive loops)":
            include_inductive = st.checkbox("Include inductive loops", value=True)
        
        # Method-specific parameters
        lambda_auto = True
        lambda_value = None
        reg_order = 2
        
        if analysis_method == "Tikhonov Regularization (NNLS)":
            reg_order = st.selectbox("Regularization order", [0, 1, 2], index=2)
            lambda_auto = st.checkbox("Automatic λ selection", value=True)
            if not lambda_auto:
                lambda_value = st.number_input("λ value", value=1e-4, format="%.1e")
        
        elif analysis_method == "Bayesian MCMC":
            if not PYMC_AVAILABLE:
                st.warning("⚠️ PyMC not installed. Bayesian MCMC may not work.")
            n_samples = st.slider("MCMC samples", 500, 5000, 2000)
            n_tune = st.slider("Tuning samples", 500, 2000, 1000)
            st.session_state.app_state.drt_parameters['n_samples'] = n_samples
            st.session_state.app_state.drt_parameters['n_tune'] = n_tune
        
        elif analysis_method == "Maximum Entropy (auto-λ)":
            entropy_lambda_auto = st.checkbox("Auto-select λ", value=True)
            if not entropy_lambda_auto:
                st.session_state.app_state.drt_parameters['entropy_lambda'] = st.number_input("Entropy λ", value=0.1, format="%.2f")
            st.session_state.app_state.drt_parameters['lambda_auto'] = entropy_lambda_auto
        
        elif analysis_method == "Finite Gaussian Process (fGP-DRT)":
            n_components = st.slider("GP components", 10, 50, 30)
            st.session_state.app_state.drt_parameters['n_components'] = n_components
        
        elif analysis_method == "Loewner Framework (RLF)":
            model_order = st.number_input("Model order (0=auto)", min_value=0, max_value=100, value=0)
            st.session_state.app_state.drt_parameters['model_order'] = model_order if model_order > 0 else None
        
        # Store parameters
        st.session_state.app_state.drt_method = analysis_method
        st.session_state.app_state.drt_parameters.update({
            'n_tau': n_tau,
            'include_inductive': include_inductive,
            'reg_order': reg_order,
            'lambda_auto': lambda_auto,
            'lambda_value': lambda_value
        })
        
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Back to Data", use_container_width=True):
                st.session_state.app_state.current_step = 1
                st.rerun()
        with col_next:
            if st.button("🚀 Run DRT Analysis", type="primary", use_container_width=True):
                with st.spinner("Calculating DRT..."):
                    try:
                        if analysis_method == "Tikhonov Regularization (NNLS)":
                            drt_solver = TikhonovDRT(data, regularization_order=reg_order, 
                                                     include_inductive=include_inductive)
                            result = drt_solver.compute(n_tau=n_tau, lambda_value=lambda_value, 
                                                        lambda_auto=lambda_auto)
                        
                        elif analysis_method == "Bayesian MCMC":
                            drt_solver = BayesianDRT(data, include_inductive=include_inductive)
                            if PYMC_AVAILABLE:
                                result = drt_solver.compute(n_tau=n_tau, 
                                                           n_samples=st.session_state.app_state.drt_parameters.get('n_samples', 2000),
                                                           n_tune=st.session_state.app_state.drt_parameters.get('n_tune', 1000))
                            else:
                                result = drt_solver.compute(n_tau=n_tau, n_samples=500)
                        
                        elif analysis_method == "Maximum Entropy (auto-λ)":
                            drt_solver = MaxEntropyDRT(data, include_inductive=include_inductive)
                            lambda_auto_val = st.session_state.app_state.drt_parameters.get('lambda_auto', True)
                            lambda_val = st.session_state.app_state.drt_parameters.get('entropy_lambda', None)
                            result = drt_solver.compute(n_tau=n_tau, lambda_value=lambda_val, 
                                                        lambda_auto=lambda_auto_val)
                        
                        elif analysis_method == "Finite Gaussian Process (fGP-DRT)":
                            drt_solver = FiniteGaussianProcessDRT(data, include_inductive=include_inductive)
                            n_comp = st.session_state.app_state.drt_parameters.get('n_components', 30)
                            result = drt_solver.compute(n_tau=n_tau, n_components=n_comp)
                        
                        elif analysis_method == "Loewner Framework (RLF)":
                            drt_solver = LoewnerFrameworkDRT(data)
                            model_order_val = st.session_state.app_state.drt_parameters.get('model_order', None)
                            result = drt_solver.compute(n_tau=n_tau, model_order=model_order_val)
                        
                        else:  # Generalized DRT
                            drt_solver = TikhonovDRT(data, regularization_order=2, include_inductive=include_inductive)
                            result = drt_solver.compute(n_tau=n_tau, lambda_auto=True)
                        
                        # Store results
                        st.session_state.app_state.drt_result = result
                        st.session_state.app_state.drt_calculated = True
                        
                        # Also store for reconstruction
                        st.session_state.app_state.drt_solver = drt_solver
                        
                        st.success("✅ DRT calculation complete!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"DRT calculation failed: {e}")
    
    with col2:
        if st.session_state.app_state.drt_calculated and st.session_state.app_state.drt_result:
            st.subheader("📊 DRT Results")
            
            result = st.session_state.app_state.drt_result
            
            st.info(f"""
            **Method:** {result.method}
            **R∞:** {result.R_inf:.4f} Ω
            **Rpol:** {result.R_pol:.4f} Ω
            **τ range:** {result.tau_grid.min():.2e} - {result.tau_grid.max():.2e} s
            """)
            
            if 'lambda' in result.metadata:
                st.info(f"**λ:** {result.metadata['lambda']:.3e}")
            
            peaks = find_peaks_drt(result.tau_grid, result.gamma, prominence=0.05)
            
            fig_drt = plot_drt_matplotlib(result, peaks)
            st.pyplot(fig_drt)
            plt.close()
            
            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("⬅️ Back to Parameters", use_container_width=True):
                    pass
            with col_next:
                if st.button("Next: Peak Deconvolution ➡️", type="primary", use_container_width=True):
                    st.session_state.app_state.current_step = 3
                    st.rerun()
        else:
            st.info("👈 Configure parameters and click 'Run DRT Analysis' to begin")


# ============================================================================
# Step 3: Gaussian Deconvolution of DRT Peaks
# ============================================================================

def step3_gaussian_deconvolution():
    """Step 3: Perform Gaussian deconvolution on DRT peaks"""
    st.header("📈 Step 3: Gaussian Deconvolution of DRT Peaks")
    
    if st.session_state.app_state.drt_result is None:
        st.error("No DRT results found. Please complete Step 2 first.")
        if st.button("⬅️ Back to DRT Analysis"):
            st.session_state.app_state.current_step = 2
            st.rerun()
        return
    
    drt_result = st.session_state.app_state.drt_result
    
    # Prepare data for deconvolution
    # Convert DRT (γ vs τ) to format suitable for Gaussian deconvolution
    log_tau = np.log10(drt_result.tau_grid)
    gamma_norm = drt_result.gamma / np.max(drt_result.gamma) if np.max(drt_result.gamma) > 0 else drt_result.gamma
    
    # Create deconvolver if not exists
    if st.session_state.app_state.deconvolver is None:
        deconvolver = GaussianDeconvolver(
            x_linear=drt_result.tau_grid,
            y_original=gamma_norm,
            use_log_x=True,
            use_log_y=False,
            clip_negative=st.session_state.app_state.clip_negative,
            show_warnings=st.session_state.app_state.show_warnings,
            baseline_method=st.session_state.app_state.baseline_method,
            smoothing_level=st.session_state.app_state.smoothing_level
        )
        st.session_state.app_state.deconvolver = deconvolver
        
        # Auto-detect peaks immediately on entry
        with st.spinner("Auto-detecting peaks..."):
            peaks, peak_info, initial_params, derivatives = deconvolver.auto_detect_peaks(
                sensitivity=st.session_state.app_state.sensitivity,
                min_distance=st.session_state.app_state.min_distance
            )
            st.session_state.app_state.peak_info = peak_info
            st.session_state.app_state.derivatives = derivatives
            st.session_state.app_state.initial_peak_params = initial_params
            st.session_state.app_state.manual_peaks = []
            st.session_state.app_state.residuals_peaks = []
    
    deconvolver = st.session_state.app_state.deconvolver
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("🔍 Peak Detection Parameters")
        
        sensitivity = st.slider("Sensitivity", 0.001, 0.1, 
                               value=st.session_state.app_state.sensitivity,
                               step=0.001, format="%.3f")
        min_distance = st.slider("Minimum distance between peaks", 1, 20,
                                value=st.session_state.app_state.min_distance, step=1)
        
        if sensitivity != st.session_state.app_state.sensitivity or min_distance != st.session_state.app_state.min_distance:
            st.session_state.app_state.sensitivity = sensitivity
            st.session_state.app_state.min_distance = min_distance
            # Redetect peaks
            with st.spinner("Re-detecting peaks..."):
                peaks, peak_info, initial_params, derivatives = deconvolver.auto_detect_peaks(
                    sensitivity=sensitivity, min_distance=min_distance
                )
                st.session_state.app_state.peak_info = peak_info
                st.session_state.app_state.derivatives = derivatives
                st.session_state.app_state.initial_peak_params = initial_params
                st.session_state.app_state.manual_peaks = []
                st.session_state.app_state.residuals_peaks = []
            st.rerun()
        
        st.markdown("---")
        st.subheader("Manual Peak Addition")
        
        # Get number of data points for point selection
        n_points = len(deconvolver.x_linear)
        
        # Create slider by point index (1-based for user-friendly)
        point_index = st.slider("Select peak by point index:",
                               min_value=1,
                               max_value=n_points,
                               value=n_points // 2,
                               step=1,
                               help="Select point index (1 to {}) to add peak at that position".format(n_points))
        
        # Convert index to actual τ value
        manual_position = deconvolver.x_linear[point_index - 1]
        
        # Display the τ value for reference
        st.info(f"Selected position: τ = {manual_position:.3e} s (point {point_index}/{n_points})")
        
        st.session_state.app_state.manual_peak_position = manual_position
        
        col_add1, col_add2 = st.columns(2)
        with col_add1:
            if st.button("➕ Add Manual Peak", use_container_width=True):
                new_peak, new_params = deconvolver.add_manual_peak(manual_position)
                if st.session_state.app_state.peak_info is None:
                    st.session_state.app_state.peak_info = []
                # Assign ID for tracking
                new_peak['id'] = len(st.session_state.app_state.peak_info) + 1
                st.session_state.app_state.peak_info.append(new_peak)
                if st.session_state.app_state.initial_peak_params is None:
                    st.session_state.app_state.initial_peak_params = []
                st.session_state.app_state.initial_peak_params.extend(new_params)
                st.session_state.app_state.manual_peaks.append(new_peak)
                st.success(f"Manual peak added at τ = {manual_position:.3e} s")
                st.rerun()
        
        with col_add2:
            if st.button("🔍 Find Missing Peaks", use_container_width=True):
                with st.spinner("Analyzing residuals..."):
                    if st.session_state.app_state.peak_info is None:
                        _, st.session_state.app_state.peak_info, st.session_state.app_state.initial_peak_params, _ = deconvolver.auto_detect_peaks(
                            sensitivity=sensitivity, min_distance=min_distance
                        )
                    
                    missing_peaks, missing_params = deconvolver.find_missing_peaks_by_residuals(
                        st.session_state.app_state.peak_info,
                        sensitivity=sensitivity * 0.5,
                        min_distance=min_distance
                    )
                    
                    if missing_peaks:
                        for p in missing_peaks:
                            p['id'] = len(st.session_state.app_state.peak_info) + 1
                            st.session_state.app_state.peak_info.append(p)
                            st.session_state.app_state.initial_peak_params.extend([p['amp_est'], p['cen_est'], p['sigma_est']])
                            st.session_state.app_state.residuals_peaks.append(p)
                        st.success(f"Added {len(missing_peaks)} peaks from residuals")
                        st.rerun()
                    else:
                        st.info("No additional peaks found in residuals")
        
        st.markdown("---")
        
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Back to DRT", use_container_width=True):
                st.session_state.app_state.current_step = 2
                st.rerun()
        with col_next:
            if st.button("🎯 Perform Deconvolution", type="primary", use_container_width=True):
                with st.spinner("Performing Gaussian deconvolution..."):
                    # Ensure peak_info is initialized
                    if st.session_state.app_state.peak_info is None or len(st.session_state.app_state.peak_info) == 0:
                        peaks, peak_info, initial_params, _ = deconvolver.auto_detect_peaks(
                            sensitivity=sensitivity, min_distance=min_distance
                        )
                        st.session_state.app_state.peak_info = peak_info
                        st.session_state.app_state.initial_peak_params = initial_params
                        st.success(f"Auto-detected {len(peak_info)} peaks")
                    
                    # Check if we have peaks to fit
                    if st.session_state.app_state.initial_peak_params is None or len(st.session_state.app_state.initial_peak_params) == 0:
                        st.error("No peaks detected. Please adjust sensitivity or add manual peaks.")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(progress, message):
                            progress_bar.progress(progress)
                            status_text.text(message)
                        
                        # Ensure initial_params is a list, not None
                        initial_params = st.session_state.app_state.initial_peak_params
                        if isinstance(initial_params, np.ndarray):
                            initial_params = initial_params.tolist()
                        
                        success = deconvolver.fit(
                            initial_params=initial_params,
                            method=st.session_state.app_state.fitting_method,
                            maxfev=st.session_state.app_state.max_nfev,
                            fit_quality=st.session_state.app_state.fit_quality,
                            last_popt=st.session_state.app_state.last_popt,
                            progress_callback=update_progress
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if success:
                            st.session_state.app_state.last_popt = deconvolver.popt
                            st.session_state.app_state.deconv_result = deconvolver.create_deconvolution_result()
                            st.session_state.app_state.deconv_calculated = True
                            st.session_state.app_state.current_step = 4  # ADD THIS LINE - переход на шаг 4
                            st.success("✅ Deconvolution complete!")
                            st.rerun()
                        else:
                            st.error("Deconvolution failed. Try adjusting parameters.")
    
    with col2:
        st.subheader("📊 Peak Detection Preview")
        
        if st.session_state.app_state.peak_info is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if deconvolver.use_log_x:
                ax.set_xscale('log')
            
            ax.plot(deconvolver.x_linear, deconvolver.y_original, 
                   'o-', markersize=3, linewidth=1, alpha=0.7, 
                   label='DRT Data', color='black', zorder=1)
            
            source_colors = {'auto': '#2ca02c', 'manual': '#ff7f0e', 'residuals': '#1f77b4'}
            for idx, info in enumerate(st.session_state.app_state.peak_info):
                source = info.get('source', 'auto')
                color = source_colors.get(source, '#2ca02c')
                ax.plot(info['x_linear'], info['y_original'], 'o', 
                       markersize=8, markeredgecolor='darkred', 
                       markerfacecolor=color, zorder=3)
                ax.text(info['x_linear'], info['y_original'] * 1.05, 
                       f'τ={info["x_linear"]:.2e}s', ha='center', 
                       fontsize=8, rotation=45)
            
            if st.session_state.app_state.manual_peak_position is not None:
                idx = np.argmin(np.abs(deconvolver.x_linear - st.session_state.app_state.manual_peak_position))
                y_at_position = deconvolver.y_original[idx]
                ax.axvline(x=st.session_state.app_state.manual_peak_position, 
                          color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.plot(st.session_state.app_state.manual_peak_position, y_at_position, 
                       'ro', markersize=10)
            
            ax.set_xlabel('Relaxation Time τ (s)', fontweight='bold')
            ax.set_ylabel('γ(τ) (norm.)', fontweight='bold')
            ax.set_title(f'Detected Peaks ({len(st.session_state.app_state.peak_info)} peaks)', fontweight='bold')
            ax.legend(['DRT Data', 'Detected Peaks'], loc='upper left')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            st.pyplot(fig)
            plt.close()
            
            # Peak info table with delete buttons
            if st.session_state.app_state.peak_info:
                st.subheader("Peak List")
                
                for i, info in enumerate(st.session_state.app_state.peak_info):
                    col_a, col_b, col_c, col_d, col_e = st.columns([0.5, 2, 2, 2, 1])
                    source = info.get('source', 'auto')
                    source_icon = "🟢" if source == 'auto' else "🟠" if source == 'manual' else "🔵"
                    
                    with col_a:
                        st.write(f"{i+1}")
                    with col_b:
                        st.write(f"{source_icon} {source}")
                    with col_c:
                        st.write(f"τ = {info['x_linear']:.4e} s")
                    with col_d:
                        st.write(f"γ = {info['y_original']:.4e}")
                    with col_e:
                        if st.button("🗑️", key=f"delete_peak_{i}", help=f"Delete peak {i+1}"):
                            if deconvolver.remove_peak_by_id(i+1):
                                st.success(f"Peak {i+1} removed")
                                st.rerun()
                
                st.markdown("---")
                
                # Additional statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    auto_count = sum(1 for p in st.session_state.app_state.peak_info if p.get('source', 'auto') == 'auto')
                    st.metric("Auto-detected", auto_count)
                with col2:
                    manual_count = sum(1 for p in st.session_state.app_state.peak_info if p.get('source', '') == 'manual')
                    st.metric("Manually added", manual_count)
                with col3:
                    residual_count = sum(1 for p in st.session_state.app_state.peak_info if p.get('source', '') == 'residuals')
                    st.metric("From residuals", residual_count)
        
        elif st.session_state.app_state.deconv_calculated and st.session_state.app_state.deconv_result:
            st.success("✅ Deconvolution completed!")
            result = st.session_state.app_state.deconv_result
            st.metric("R²", f"{result.quality_metrics.get('R²', 0):.4f}")
            st.metric("Number of Peaks", len(result.peaks))
            st.metric("Total Area", f"{result.total_area:.4e}")


# ============================================================================
# Step 4: Results and Export
# ============================================================================

def step4_results():
    """Step 4: Display results, tables, and export options"""
    st.header("📊 Step 4: Results and Export")
    
    if st.session_state.app_state.deconv_result is None:
        st.error("No deconvolution results found. Please complete Step 3 first.")
        if st.button("⬅️ Back to Deconvolution"):
            st.session_state.app_state.current_step = 3
            st.rerun()
        return
    
    deconv_result = st.session_state.app_state.deconv_result
    
    # Navigation buttons
    col_prev, col_next = st.columns([1, 5])
    with col_prev:
        if st.button("⬅️ Back to Deconvolution", use_container_width=True):
            st.session_state.app_state.current_step = 3
            st.rerun()
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Deconvolution Plot", "📊 Area Distribution", "📋 Complete Dataset", 
                                             "📈 Normalized View", "📥 Export"])
    
    with tab1:
        st.subheader("Gaussian Deconvolution Result")
        
        fig = plot_deconvolution_result(deconv_result, show_components=True, show_baseline=True)
        st.pyplot(fig)
        plt.close()
        
        # Download plot button
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button("📥 Download Deconvolution Plot (PNG)", 
                          data=buf,
                          file_name=f"deconvolution_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                          mime="image/png")
    
    with tab2:
        st.subheader("Area Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of fractions
            fig, ax = plt.subplots(figsize=(8, 6))
            peaks_ids = [f'Peak {p.id}' for p in deconv_result.peaks]
            fractions = [p.fraction_percent for p in deconv_result.peaks]
            colors = plt.cm.Set3(np.linspace(0, 1, len(deconv_result.peaks)))
            
            bars = ax.bar(peaks_ids, fractions, color=colors, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Peak', fontweight='bold')
            ax.set_ylabel('Fraction (%)', fontweight='bold')
            ax.set_title('Peak Area Distribution', fontweight='bold')
            ax.set_ylim(0, max(fractions) * 1.2 if fractions else 100)
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, frac in zip(bars, fractions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{frac:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            wedges, texts, autotexts = ax.pie(fractions, labels=peaks_ids, autopct='%1.1f%%',
                                               colors=colors, startangle=90,
                                               textprops={'fontweight': 'bold'})
            ax.set_title('Area Distribution - Pie Chart', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Summary statistics
        st.markdown("---")
        st.subheader("Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Area", f"{deconv_result.total_area:.4e}")
        with col2:
            st.metric("Number of Peaks", len(deconv_result.peaks))
        with col3:
            max_peak = max(deconv_result.peaks, key=lambda p: p.fraction)
            st.metric("Dominant Peak", f"Peak {max_peak.id} ({max_peak.fraction_percent:.1f}%)")
        with col4:
            avg_area = deconv_result.total_area / len(deconv_result.peaks) if deconv_result.peaks else 0
            st.metric("Average Area", f"{avg_area:.4e}")
    
    with tab3:
        st.subheader("Complete Dataset - Peak Parameters")
        
        # Create detailed table
        data = []
        for peak in deconv_result.peaks:
            data.append({
                'Peak ID': peak.id,
                'Center (τ, s)': f"{peak.center:.4e}",
                'Center (log τ)': f"{peak.center_log:.4f}",
                'Amplitude': f"{peak.amplitude:.4e}",
                'Amplitude (norm)': f"{peak.amplitude_norm:.4f}",
                'Sigma (log)': f"{peak.sigma_log:.4f}",
                'FWHM': f"{peak.fwhm:.4f}",
                'Area': f"{peak.area:.4e}",
                'Fraction (%)': f"{peak.fraction_percent:.2f}"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Quality Metrics")
        
        metrics = deconv_result.quality_metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{metrics.get('R²', 0):.6f}")
        with col2:
            st.metric("RMSE", f"{metrics.get('RMSE', 0):.2e}")
        with col3:
            st.metric("AIC", f"{metrics.get('AIC', 0):.2f}")
        with col4:
            st.metric("BIC", f"{metrics.get('BIC', 0):.2f}")
        
        if deconv_result.baseline_method != 'none' and deconv_result.baseline_params:
            st.markdown("---")
            st.subheader("Baseline Parameters")
            baseline_df = pd.DataFrame([{
                'Method': deconv_result.baseline_method,
                'Parameters': ', '.join([f"{p:.4e}" for p in deconv_result.baseline_params])
            }])
            st.dataframe(baseline_df, use_container_width=True)
    
    with tab4:
        st.subheader("Normalized View (Max Peak = 1)")
        
        fig = plot_deconvolution_components_comparison(deconv_result)
        st.pyplot(fig)
        plt.close()
        
        # Normalized parameters table
        st.markdown("---")
        st.subheader("Normalized Parameters")
        
        norm_data = []
        max_amp = deconv_result.max_amplitude
        for peak in deconv_result.peaks:
            norm_data.append({
                'Peak': peak.id,
                'Center (τ, s)': f"{peak.center:.4e}",
                'Normalized Amplitude': f"{peak.amplitude / max_amp:.4f}",
                'Original Amplitude': f"{peak.amplitude:.4e}",
                'Fraction (%)': f"{peak.fraction_percent:.2f}"
            })
        
        df_norm = pd.DataFrame(norm_data)
        st.dataframe(df_norm, use_container_width=True)
    
    with tab5:
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Export Peak Data**")
            
            # Create peaks DataFrame
            peaks_df = pd.DataFrame([{
                'Peak_ID': p.id,
                'Center_tau_s': p.center,
                'Center_log_tau': p.center_log,
                'Amplitude': p.amplitude,
                'Amplitude_Normalized': p.amplitude_norm,
                'Sigma_log': p.sigma_log,
                'FWHM': p.fwhm,
                'Area': p.area,
                'Fraction': p.fraction,
                'Fraction_Percent': p.fraction_percent,
                'Source': getattr(p, 'source', 'auto')
            } for p in deconv_result.peaks])
            
            csv_peaks = peaks_df.to_csv(index=False)
            st.download_button(
                "📥 Export Peaks as CSV",
                data=csv_peaks,
                file_name=f"deconvolution_peaks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("**Export Fitting Data**")
            
            # Create fitting data DataFrame
            fit_data = pd.DataFrame({
                'tau_s': deconv_result.x_linear,
                'gamma_tau': deconv_result.y_original,
                'gamma_fit': deconv_result.fit_y_norm * deconv_result.y_original.max(),
                'Residuals': deconv_result.quality_metrics.get('Residuals', np.zeros_like(deconv_result.x_linear)) * deconv_result.y_original.max()
            })
            
            csv_fit = fit_data.to_csv(index=False)
            st.download_button(
                "📥 Export Fitting Data as CSV",
                data=csv_fit,
                file_name=f"deconvolution_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("**Generate Report**")
            
            if st.button("📄 Generate Detailed Report", use_container_width=True):
                report = f"""GAUSSIAN DECONVOLUTION REPORT
{"="*80}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

QUALITY METRICS:
{"-"*40}
R²: {deconv_result.quality_metrics.get('R²', 0):.6f}
AIC: {deconv_result.quality_metrics.get('AIC', 0):.2f}
BIC: {deconv_result.quality_metrics.get('BIC', 0):.2f}
χ²: {deconv_result.quality_metrics.get('χ²', 0):.2e}
RMSE: {deconv_result.quality_metrics.get('RMSE', 0):.2e}
Max Error: {deconv_result.quality_metrics.get('Max Error', 0):.2e}

"""
                if deconv_result.baseline_method != 'none':
                    report += f"""BASELINE PARAMETERS:
{"-"*40}
Method: {deconv_result.baseline_method}
Parameters: {', '.join([f'{p:.4e}' for p in deconv_result.baseline_params])}

"""
                
                report += f"""PEAK PARAMETERS:
{"-"*80}
ID    Center (s)      Amplitude       Area           Fraction(%)
{"-"*80}"""
                
                for p in deconv_result.peaks:
                    report += f"\n{p.id:<4} {p.center:.4e}   {p.amplitude:.4e}   {p.area:.4e}   {p.fraction_percent:.2f}"
                
                report += f"""
{"="*80}
Total Area: {deconv_result.total_area:.6e}
Number of Peaks: {len(deconv_result.peaks)}
{"="*80}"""
                
                st.download_button(
                    "📥 Download Report",
                    data=report,
                    file_name=f"deconvolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            st.markdown("---")
            st.markdown("**Start New Analysis**")
            
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.app_state = AppState()
                st.rerun()
    
    # Additional: Compare with original DRT
    if st.session_state.app_state.drt_result:
        st.markdown("---")
        st.subheader("Comparison with Original DRT")
        
        drt_result = st.session_state.app_state.drt_result
        peaks_drt = find_peaks_drt(drt_result.tau_grid, drt_result.gamma)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogx(drt_result.tau_grid, drt_result.gamma, 'b-', linewidth=2, label='Original DRT')
            
            for peak in deconv_result.peaks:
                ax.axvline(x=peak.center, color='red', linestyle='--', alpha=0.5)
                ax.text(peak.center, np.max(drt_result.gamma) * 0.9, 
                       f'Peak {peak.id}', ha='center', fontsize=9)
            
            ax.set_xlabel('Relaxation Time τ (s)', fontweight='bold')
            ax.set_ylabel('γ(τ)', fontweight='bold')
            ax.set_title('DRT with Deconvolved Peak Positions', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, linestyle='--')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Comparison table
            comparison_data = []
            for i, peak in enumerate(deconv_result.peaks):
                matching_drt = None
                for p in peaks_drt:
                    if abs(np.log10(p['tau']) - peak.center_log) < 0.3:
                        matching_drt = p
                        break
                
                comparison_data.append({
                    'Deconvolved Peak': i + 1,
                    'τ (s)': f"{peak.center:.4e}",
                    'DRT τ (s)': f"{matching_drt['tau']:.4e}" if matching_drt else "N/A",
                    'Match': "✓" if matching_drt else "✗"
                })
            
            df_comp = pd.DataFrame(comparison_data)
            st.dataframe(df_comp, use_container_width=True)


# ============================================================================
# Navigation and Main Application
# ============================================================================

def show_step_indicator():
    """Display step indicator in sidebar"""
    st.sidebar.markdown("### 📍 Analysis Steps")
    
    steps = {
        1: "1. Data Loading",
        2: "2. DRT Analysis",
        3: "3. Peak Deconvolution",
        4: "4. Results & Export"
    }
    
    current = st.session_state.app_state.current_step
    
    for step_num, step_name in steps.items():
        if step_num < current:
            st.sidebar.markdown(f"✅ **{step_name}**")
        elif step_num == current:
            st.sidebar.markdown(f"🔵 **{step_name}**")
        else:
            st.sidebar.markdown(f"⏳ {step_name}")
    
    st.sidebar.markdown("---")
    
    # Settings expander
    with st.sidebar.expander("⚙️ Global Settings", expanded=False):
        st.session_state.app_state.clip_negative = st.checkbox(
            "Clip negative values", 
            value=st.session_state.app_state.clip_negative
        )
        st.session_state.app_state.show_warnings = st.checkbox(
            "Show warnings", 
            value=st.session_state.app_state.show_warnings
        )
        st.session_state.app_state.smoothing_level = st.selectbox(
            "Smoothing",
            options=['none', 'light', 'medium', 'strong', 'adaptive'],
            index=0
        )
        st.session_state.app_state.baseline_method = st.selectbox(
            "Baseline correction",
            options=['none', 'constant', 'linear', 'quadratic'],
            index=0
        )
        st.session_state.app_state.fitting_method = st.selectbox(
            "Fitting method",
            options=['trf', 'dogbox', 'lm'],
            index=0
        )
        st.session_state.app_state.fit_quality = st.selectbox(
            "Fit quality",
            options=['fast', 'balanced', 'precise'],
            index=1
        )
        st.session_state.app_state.preview_mode = st.checkbox(
            "Preview mode (no fitting)",
            value=st.session_state.app_state.preview_mode
        )
    
    # Reset button
    if st.sidebar.button("🔄 Reset All", use_container_width=True):
        st.session_state.app_state = AppState()
        st.rerun()


def main():
    """Main application entry point"""
    show_step_indicator()
    
    # Route to appropriate step
    current_step = st.session_state.app_state.current_step
    
    if current_step == 1:
        step1_data_loading()
    elif current_step == 2:
        step2_drt_analysis()
    elif current_step == 3:
        step3_gaussian_deconvolution()
    elif current_step == 4:
        step4_results()
    else:
        step1_data_loading()
    
    # Footer
    st.markdown("---")
    st.markdown("⚡ **EIS-DRT Analysis Tool v4.0** | Multi-stage workflow: DRT Analysis → Gaussian Deconvolution → Area Distribution Analysis")


if __name__ == "__main__":
    main()
