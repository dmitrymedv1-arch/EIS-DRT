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
from scipy.optimize import curve_fit, least_squares, nnls
from scipy.stats import linregress
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
import cvxopt
from cvxopt import matrix, solvers

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="EIS-DRT Analysis Tool v5.0",
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
        
        # Tick settings - MAKE TICK LABELS BLACK
        'xtick.color': 'black',
        'ytick.color': 'black',
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


def get_plot_style() -> Dict[str, Any]:
    """Return current publication plot style settings from session_state."""
    default_style = {
        "font_family": "serif",
        "fs_label": 12,
        "fs_tick": 10,
        "fs_legend": 10,
        "lw": 2.0,
        "ms": 5,
        "show_grid": True,
        "tick_dir": "in",
        "axes_bg": "white",
        "cmap": "Set3",
        "dpi": 600,
    }
    if "plot_style" not in st.session_state:
        st.session_state.plot_style = default_style.copy()
    return st.session_state.plot_style


def apply_publication_plot_style(style: Dict[str, Any]):
    """Apply user-selected publication plot settings to matplotlib rcParams."""
    font_family = style.get("font_family", "serif")
    plt.rcParams.update({
        "font.family": font_family,
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": style["fs_tick"],
        "axes.labelsize": style["fs_label"],
        "axes.labelweight": "bold",
        "axes.titlesize": style["fs_label"] + 1,
        "axes.titleweight": "bold",
        "axes.facecolor": style["axes_bg"],
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "axes.grid": style["show_grid"],
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": style["fs_tick"],
        "ytick.labelsize": style["fs_tick"],
        "xtick.direction": style["tick_dir"],
        "ytick.direction": style["tick_dir"],
        "legend.fontsize": style["fs_legend"],
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "black",
        "lines.linewidth": style["lw"],
        "lines.markersize": style["ms"],
        "figure.dpi": 150,
        "savefig.dpi": style["dpi"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def render_plot_with_download(fig: plt.Figure, filename_base: str, key_prefix: str = ""):
    """Display matplotlib figure with PNG (high-DPI) and SVG vector download buttons."""
    style = get_plot_style()
    dpi = style["dpi"]
    if not style["show_grid"]:
        for axis in fig.axes:
            axis.grid(False, which="both")
    st.pyplot(fig)

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=dpi, bbox_inches="tight")
    png_bytes = buf_png.getvalue()

    buf_svg = io.BytesIO()
    fig.savefig(buf_svg, format="svg", bbox_inches="tight")
    svg_bytes = buf_svg.getvalue()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    c1, c2 = st.columns(2)
    c1.download_button(
        f"⬇️ PNG ({dpi} DPI)",
        data=png_bytes,
        file_name=f"{filename_base}_{ts}.png",
        mime="image/png",
        key=f"png_{key_prefix}_{filename_base}",
        use_container_width=True,
    )
    c2.download_button(
        "⬇️ SVG (Vector)",
        data=svg_bytes,
        file_name=f"{filename_base}_{ts}.svg",
        mime="image/svg+xml",
        key=f"svg_{key_prefix}_{filename_base}",
        use_container_width=True,
    )
    plt.close(fig)

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
    L: float = 0.0  # Added: inductance value
    lambda_opt: float = None  # Added: optimal regularization parameter
    convergence: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def log_tau(self) -> np.ndarray:
        """Log10 of relaxation times"""
        return np.log10(self.tau_grid)
    
    def get_integral(self) -> float:
        """
        Calculate total polarization resistance from DRT.
        R_pol = ∫ γ(τ) d(ln τ)
        
        This should equal self.R_pol (within regularization error)
        """
        return np.trapezoid(self.gamma, np.log(self.tau_grid))
    
    def get_integral_linear(self) -> float:
        """
        Calculate integral over linear τ (for reference only).
        This is NOT the correct way to get R_pol.
        """
        return np.trapezoid(self.gamma, self.tau_grid)
    
    def verify_integral(self) -> Tuple[float, float]:
        """
        Verify that DRT integral matches R_pol.
        
        Returns:
            Tuple[float, float]: (integral_value, ratio_to_R_pol)
        """
        integral = self.get_integral()
        ratio = integral / self.R_pol if self.R_pol > 0 else 0
        return integral, ratio


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
    auto_convention: bool = True
    
    def __post_init__(self):
        # Приводим мнимую часть к конвенции -Im(Z) > 0 для емкостной дуги:
        # если медиана основной части спектра отрицательна (пришел сырой Im(Z)), инвертируем знак;
        # если данные уже в конвенции -Im(Z) > 0 (или auto_convention=False), оставляем как есть.
        if self.auto_convention and np.median(self.im_z) < 0:
            im_converted = -self.im_z.copy()
        else:
            im_converted = self.im_z.copy()
            
        self.original_freq = self.freq.copy()
        self.original_re_z = self.re_z.copy()
        self.original_im_z = im_converted.copy()
        self.im_z = im_converted
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
    
    def detect_inductive_behavior(self) -> bool:
        """
        Automatically detect if the spectrum shows inductive behavior.
        Inductive behavior is indicated by NEGATIVE -Im(Z) at high frequencies
        or decreasing trend in -Im(Z) towards high frequencies.
        
        Returns:
            bool: True if inductive behavior is detected
        """
        # Check for negative -Im(Z) at high frequencies (inductive behavior)
        high_freq_mask = self.freq > 0.1 * np.max(self.freq)
        if np.any(high_freq_mask):
            # Inductive behavior: -Im(Z) is negative (below x-axis)
            if np.any(self.im_z[high_freq_mask] < 0):
                return True
        
        # Check for decreasing trend in -Im(Z) at high frequencies (going more negative)
        if len(self.freq) > 10:
            high_idx = np.argsort(self.freq)[-10:]  # Top 10 highest frequencies
            imag_high = self.im_z[high_idx]
            if len(imag_high) > 2 and np.all(np.diff(imag_high) < 0):
                return True
        
        return False


# ============================================================================
# Data Classes for Gaussian Deconvolution Results
# ============================================================================

@dataclass
class GaussianPeak:
    """Container for Gaussian peak parameters"""
    id: int
    center: float          # Center in linear space (τ in seconds)
    center_log: float      # Center in log10 space
    amplitude: float       # Amplitude in original scale (Ω)
    amplitude_norm: float  # Normalized amplitude
    sigma_log: float       # Sigma in log10 space
    fwhm: float           # Full width at half maximum in log10 space
    area: float           # Area under peak (Ω) - this equals resistance contribution
    fraction: float       # Fraction of total area (0-1)
    fraction_percent: float  # Percentage fraction (0-100)
    source: str = 'auto'  # Source: 'auto', 'manual', 'residuals'
    y_norm: np.ndarray = None  # Normalized y values for plotting
    characteristic_frequency: float = None  # Added for sorting
    
    def __post_init__(self):
        if self.characteristic_frequency is None:
            self.characteristic_frequency = 1.0 / (2 * np.pi * self.center)
    
    def get_characteristic_frequency(self) -> float:
        """
        Get characteristic frequency of this peak.
        f = 1/(2πτ)
        
        Returns:
            float: Characteristic frequency in Hz
        """
        return self.characteristic_frequency
    
    def get_resistance_contribution(self) -> float:
        """
        Get resistance contribution of this peak.
        This equals the area under the peak.
        
        Returns:
            float: Resistance contribution in Ω
        """
        return self.area


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
    fit_y_original: Optional[np.ndarray] = None  # Added for original scale fit
    baseline_params: Optional[List[float]] = None
    baseline_method: str = 'none'
    total_area: float = 0.0
    max_amplitude: float = 0.0
    
    def verify_resistance_conservation(self) -> Tuple[float, float]:
        """
        Verify that sum of peak areas equals total DRT integral (R_pol).
        
        Returns:
            Tuple[float, float]: (sum_peak_areas, ratio_to_total)
        """
        sum_areas = sum([p.area for p in self.peaks])
        ratio = sum_areas / self.total_area if self.total_area > 0 else 0
        return sum_areas, ratio
    
    def get_peak_resistances(self) -> List[float]:
        """
        Get resistance contribution of each peak in Ω.
        
        Returns:
            List[float]: Peak resistances in Ω
        """
        return [p.area for p in self.peaks]
    
    def get_peak_frequencies(self) -> List[float]:
        """
        Get characteristic frequencies of peaks.
        f = 1/(2πτ)
        
        Returns:
            List[float]: Characteristic frequencies in Hz
        """
        return [p.get_characteristic_frequency() for p in self.peaks]


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
    drt_solver: Optional[Any] = None  # Added to store solver for reconstruction
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
    deconv_tau_range_idx: Optional[Tuple[int, int]] = None
    deconv_parameters: Dict[str, Any] = field(default_factory=dict)
    deconv_calculated: bool = False
    
    # Step 4: Results
    results_ready: bool = False
    refined_rq_params: Optional[Dict[str, Any]] = None
    
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

def parse_uploaded_dataframe(file, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Universally read uploaded EIS file (.txt, .csv, .dat, .z, .mpt, .xlsx)
    with automatic separator and header detection.
    """
    file.seek(0)
    filename = getattr(file, 'name', '').lower()
    
    # Handle Excel files
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df_raw = pd.read_excel(file, header=None, nrows=nrows)
    else:
        raw_bytes = file.read()
        file.seek(0)
        try:
            text = raw_bytes.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = raw_bytes.decode('latin-1', errors='ignore')
        
        # Filter out empty lines and comment lines starting with #
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith('#')]
        if not lines:
            raise ValueError("Uploaded file is empty.")
        
        if nrows is not None:
            lines = lines[:nrows + 1]
        
        # Detect delimiter from first data line
        sample_line = lines[-1]
        if '\t' in sample_line:
            sep = r'\t+'
        elif ';' in sample_line:
            sep = ';'
        elif ',' in sample_line:
            # Check if comma is used as column separator (at least 2 commas) vs decimal comma
            if sample_line.count(',') >= 2 and len(sample_line.split()) == 1:
                sep = ','
            elif sample_line.count(',') >= 2 and ('\t' not in sample_line):
                sep = ','
            else:
                sep = r'\s+'
                lines = [l.replace(',', '.') for l in lines]
        else:
            sep = r'\s+'
        
        df_raw = pd.read_csv(io.StringIO('\n'.join(lines)), sep=sep, header=None, engine='python')
    
    # Drop completely empty columns
    df_raw = df_raw.dropna(axis=1, how='all')
    
    # Auto-detect whether the first row is a text header or numeric data
    first_row = df_raw.iloc[0]
    is_header = False
    for val in first_row:
        try:
            float(str(val).replace(',', '.'))
        except ValueError:
            is_header = True
            break
    
    if is_header:
        headers = [str(col).strip() for col in df_raw.iloc[0]]
        # Deduplicate column names if needed
        seen = {}
        unique_headers = []
        for h in headers:
            if h in seen:
                seen[h] += 1
                unique_headers.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                unique_headers.append(h)
        df = df_raw.iloc[1:].copy()
        df.columns = unique_headers
    else:
        # Assign clean default column names so the first data point is NOT lost
        default_names = ['Column 1 (Freq)', 'Column 2 (Re Z)', 'Column 3 (Im Z)']
        cols = []
        for idx in range(df_raw.shape[1]):
            cols.append(default_names[idx] if idx < 3 else f"Column {idx + 1}")
        df = df_raw.copy()
        df.columns = cols
    
    df = df.reset_index(drop=True)
    if nrows is not None:
        df = df.head(nrows)
    return df


def load_data(file, freq_col, re_col, im_col) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load impedance data from uploaded file."""
    if file is not None:
        try:
            df = parse_uploaded_dataframe(file)
            if freq_col in df.columns and re_col in df.columns and im_col in df.columns:
                # Convert safely to numeric and drop non-numeric rows if any exist in the middle
                sub = df[[freq_col, re_col, im_col]].apply(
                    lambda s: pd.to_numeric(s.astype(str).str.replace(',', '.', regex=False), errors='coerce')
                ).dropna()
                freq = sub[freq_col].values.astype(float)
                re_z = sub[re_col].values.astype(float)
                im_z = sub[im_col].values.astype(float)
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
                        # НЕ используем abs() - сохраняем знак!
                        # Отрицательное значение = индуктивное поведение
                        im_val = float(parts[2])
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
        
        # Robust estimation of ohmic resistance
        self.R_inf = self._estimate_R_inf_robust()
        
        # Total polarization resistance (difference between low and high frequency real parts)
        low_freq_idx = np.where(self.frequencies < 0.1 * np.max(self.frequencies))[0]
        if len(low_freq_idx) > 3:
            R_total = np.mean(self.Z_real[low_freq_idx[:5]])
        else:
            R_total = self.Z_real[0] if len(self.Z_real) > 0 else 0
        self.R_pol = R_total - self.R_inf if R_total > self.R_inf else 1.0
    
    def _estimate_R_inf_robust(self) -> float:
        """
        Robust estimation of ohmic resistance R∞.
        For spectra with inductive behavior, extrapolation is used.
        
        Returns:
            float: Estimated R∞ in Ω
        """
        # Take highest frequency points (top 20% or at least 3 points)
        n_high = max(3, len(self.frequencies) // 5)
        high_freq_indices = np.argsort(self.frequencies)[-n_high:]
        
        # Check if inductive behavior is present
        has_inductive = self.data.detect_inductive_behavior()
        
        if has_inductive and len(high_freq_indices) >= 3:
            # For spectra with inductance, extrapolate R∞ using 1/ω²
            omega = 2 * np.pi * self.frequencies[high_freq_indices]
            re_z_high = self.Z_real[high_freq_indices]
            
            # Avoid division by zero
            inv_omega2 = 1 / (omega**2 + 1e-10)
            
            try:
                # Linear regression: Re(Z) = R∞ + slope * (1/ω²)
                slope, intercept, r_value, p_value, std_err = linregress(inv_omega2, re_z_high)
                return intercept
            except Exception as e:
                logging.warning(f"Extrapolation failed: {e}, using mean of high frequencies")
                return np.mean(self.Z_real[high_freq_indices])
        else:
            # Simple mean of high frequency points
            return np.mean(self.Z_real[high_freq_indices])
    
    def _detect_inductive_behavior(self) -> bool:
        """
        Automatically detect if the spectrum shows inductive behavior.
        
        Returns:
            bool: True if inductive behavior is detected
        """
        return self.data.detect_inductive_behavior()
    
    def _l_curve_criterion(self, residuals: np.ndarray, solution_norms: np.ndarray) -> int:
        """
        Find corner of L-curve (maximum curvature) for automatic λ selection.
        
        Args:
            residuals: Array of residuals for different λ values
            solution_norms: Array of solution norms for different λ values
        
        Returns:
            int: Index of optimal λ in the λ array
        """
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
    
    def _build_kernel_matrix(self, tau_grid: np.ndarray, include_rl: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build kernel matrix for given time grid.
        IMPORTANT: Kernels are scaled for integration over d(ln τ)
        
        Args:
            tau_grid: Array of relaxation times
            include_rl: If True, includes RL contribution for inductive loops
                        (Note: In DRTtools, L is handled separately, not in kernel)
        
        Returns:
            Tuple of (K_real, K_imag) matrices
        """
        M = len(tau_grid)
        K_real = np.zeros((self.N, M))
        K_imag = np.zeros((self.N, M))
        
        omega = 2 * np.pi * self.frequencies
        
        # Calculate step in natural logarithm (ln) for proper integration
        # d(ln τ) = d(log10 τ) * ln(10)
        dln_tau = np.mean(np.diff(np.log(tau_grid)))  # This is the step in ln(τ)
        
        for i in range(self.N):
            for j in range(M):
                # Standard DRT kernel for RC elements
                denominator = 1 + (omega[i] * tau_grid[j])**2
                
                # K_real = 1 / (1 + ω²τ²) * d(ln τ)
                K_real[i, j] = 1.0 / denominator * dln_tau
                
                # K_imag = -ωτ / (1 + ω²τ²) * d(ln τ)
                K_imag[i, j] = -omega[i] * tau_grid[j] / denominator * dln_tau
        
        return K_real, K_imag
    
    def _build_kernel_matrix_with_inductance(self, tau_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build kernel matrix for DRT and also return the column for inductance.
        This follows the DRTtools approach where L is handled as a separate parameter.
        
        Args:
            tau_grid: Array of relaxation times
        
        Returns:
            Tuple of (K_extended, K_real, K_imag) where:
            - K_extended: Full matrix for [γ; L] with shape (2N, M+1)
            - K_real: Real part kernel (without L)
            - K_imag: Imag part kernel (without L)
        """
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=False)
        
        # Build the combined kernel matrix
        # For Real part: contribution from γ only (L has no real part)
        # For Imag part: contribution from γ AND from L: -ωL
        omega = 2 * np.pi * self.frequencies
        
        # Create extended matrix: [K_real; K_imag] with extra column for L
        K_base = np.vstack([K_real, K_imag])
        L_column = np.concatenate([np.zeros(self.N), -omega])
        K_extended = np.column_stack([K_base, L_column])
        
        return K_extended, K_real, K_imag
    
    def get_drt_integral(self, gamma: np.ndarray, tau_grid: np.ndarray) -> float:
        """
        Calculate integral of DRT over ln(τ) which equals polarization resistance.
        R_pol = ∫ γ(τ) d(ln τ)
        """
        return np.trapezoid(gamma, np.log(tau_grid))
    
    def verify_reconstruction(self, gamma: np.ndarray, L: float, tau_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Verify reconstruction quality by calculating fitted impedance and error.
        
        Args:
            gamma: DRT array
            L: Inductance value (0 if not used)
            tau_grid: Relaxation time grid
        
        Returns:
            Tuple of (Z_rec_real, Z_rec_imag, relative_error)
        """
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=False)
        
        # Reconstruct impedance
        Z_rec_real = self.R_inf + (K_real @ gamma) / np.log(10)
        Z_rec_imag = -(K_imag @ gamma) / np.log(10) + 2 * np.pi * self.frequencies * L
        
        # Calculate relative error
        Z_original = self.Z_real + 1j * self.Z_imag
        Z_reconstructed = Z_rec_real + 1j * Z_rec_imag
        error_percent = np.abs((Z_original - Z_reconstructed) / (Z_original + 1e-10)) * 100
        mean_error = np.mean(error_percent)
        
        return Z_rec_real, Z_rec_imag, mean_error


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
        """Solve non-negative least squares problem"""
        from scipy.optimize import nnls
        x, resid = nnls(A, b)
        if resid > 1e-6 * np.linalg.norm(b):
            logging.warning(f"NNLS residual: {resid}")
        return x
    
    def compute(self, n_tau: int = 150, lambda_value: Optional[float] = None, 
                lambda_auto: bool = True, lambda_range: Optional[np.ndarray] = None) -> DRTResult:
        """
        Compute DRT using Tikhonov regularization with NNLS.
        
        The DRT gamma(τ) satisfies:
        Z(ω) = R∞ + ∫ γ(τ) / (1 + iωτ) d(ln τ)
        
        Therefore: R_pol = ∫ γ(τ) d(ln τ)
        """
        
        # Create logarithmic grid of relaxation times
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        
        # Build kernel matrices (already scaled with d(ln τ))
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=False)
        
        # Target vector: subtract ohmic resistance from real part
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        K = np.vstack([K_real, K_imag])
        
        # Build regularization matrix
        L = self._build_regularization_matrix(n_tau, self.regularization_order)
        
        lambda_opt = None
        gamma = None
        
        if lambda_auto:
            if lambda_range is None:
                lambda_range = np.logspace(-8, 2, 30)
            
            residuals = []
            solution_norms = []
            solutions = []
            
            for lam in lambda_range:
                try:
                    # Augmented system: [K; λL] * gamma = [Z_target; 0]
                    A = np.vstack([K, lam * L])
                    b = np.concatenate([Z_target, np.zeros(L.shape[0])])
                    
                    x = self._solve_nnls(A, b)
                    
                    # Calculate residual and solution norm
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
        
        # Calculate uncertainty estimate from second derivative
        gamma_std = np.abs(np.gradient(np.gradient(gamma))) * 0.1
        
        # Verify integral of DRT equals R_pol
        drt_integral = self.get_drt_integral(gamma, tau_grid)
        
        # Log verification
        print(f"R_pol from EIS: {self.R_pol:.6f} Ω")
        print(f"R_pol from DRT integral: {drt_integral:.6f} Ω")
        print(f"Ratio (should be ~1.0): {drt_integral/self.R_pol:.4f}")
        
        if drt_integral / self.R_pol < 0.8 or drt_integral / self.R_pol > 1.2:
            warnings.warn(f"DRT integral ({drt_integral:.4f}) does not match R_pol ({self.R_pol:.4f}). Ratio: {drt_integral/self.R_pol:.3f}")
        
        gamma_corrected = gamma * np.log(10)  # ln(10) ≈ 2.302585
        
        # Пересчитываем интеграл с corrected gamma
        drt_integral_corrected = self.get_drt_integral(gamma_corrected, tau_grid)
        
        return DRTResult(
            tau_grid=tau_grid,
            gamma=gamma_corrected,
            gamma_std=gamma_std * np.log(10),  # также корректируем стандартное отклонение
            method="Tikhonov Regularization (NNLS)",
            R_inf=self.R_inf,
            R_pol=self.R_pol,
            L=0.0,
            lambda_opt=lambda_opt,
            metadata={
                'lambda': lambda_opt,
                'order': self.regularization_order,
                'lambda_auto': lambda_auto,
                'drt_integral': drt_integral_corrected,
                'integral_ratio': drt_integral_corrected / self.R_pol
            }
        )
    
    def compute_with_inductance(self, n_tau: int = 150, lambda_value: Optional[float] = None,
                                 lambda_auto: bool = True, lambda_range: Optional[np.ndarray] = None) -> DRTResult:
        """
        Compute DRT with inductance using the DRTtools approach.
        
        In this method, L is fitted as a separate parameter along with γ(τ).
        The model is: Z(ω) = R∞ + i·2πf·L + ∫ γ(τ)/(1 + iωτ) d(ln τ)
        
        The extended matrix is: [K_real; K_imag] with an extra column [0; -ω] for L
        """
        
        # Create logarithmic grid of relaxation times
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        
        # Build extended kernel matrix with L column
        K_extended, K_real, K_imag = self._build_kernel_matrix_with_inductance(tau_grid)
        
        # Target vector: subtract ohmic resistance from real part
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        
        # Build regularization matrix for γ only (L is not regularized)
        L_reg = self._build_regularization_matrix(n_tau, self.regularization_order)
        
        # Number of unknowns: n_tau (γ) + 1 (L)
        n_total = n_tau + 1
        
        # Build extended regularization matrix: zeros for L
        L_extended = np.zeros((L_reg.shape[0], n_total))
        L_extended[:, :n_tau] = L_reg
        
        lambda_opt = None
        gamma = None
        L_value = None
        
        if lambda_auto:
            if lambda_range is None:
                lambda_range = np.logspace(-8, 2, 30)
            
            residuals = []
            solution_norms = []
            solutions = []
            
            for lam in lambda_range:
                try:
                    # Augmented system: [K_extended; λL_extended] * x = [Z_target; 0]
                    A = np.vstack([K_extended, lam * L_extended])
                    b = np.concatenate([Z_target, np.zeros(L_extended.shape[0])])
                    
                    x = self._solve_nnls(A, b)
                    
                    # Calculate residual and solution norm
                    residual = np.linalg.norm(K_extended @ x - Z_target)
                    sol_norm = np.linalg.norm(L_extended @ x)
                    
                    residuals.append(residual)
                    solution_norms.append(sol_norm)
                    solutions.append(x)
                except Exception as e:
                    logging.warning(f"Lambda {lam} failed: {e}")
                    continue
            
            if len(residuals) > 2:
                best_idx = self._l_curve_criterion(np.array(residuals), np.array(solution_norms))
                lambda_opt = lambda_range[best_idx]
                x_opt = solutions[best_idx]
                gamma = x_opt[:n_tau]
                L_value = x_opt[-1]
            else:
                lambda_opt = lambda_range[0] if len(lambda_range) > 0 else 1e-4
                A = np.vstack([K_extended, lambda_opt * L_extended])
                b = np.concatenate([Z_target, np.zeros(L_extended.shape[0])])
                x_opt = self._solve_nnls(A, b)
                gamma = x_opt[:n_tau]
                L_value = x_opt[-1]
        else:
            lambda_opt = lambda_value if lambda_value is not None else 1e-4
            A = np.vstack([K_extended, lambda_opt * L_extended])
            b = np.concatenate([Z_target, np.zeros(L_extended.shape[0])])
            x_opt = self._solve_nnls(A, b)
            gamma = x_opt[:n_tau]
            L_value = x_opt[-1]
        
        # Calculate uncertainty estimate from second derivative
        gamma_std = np.abs(np.gradient(np.gradient(gamma))) * 0.1
        
        # Verify integral of DRT equals R_pol
        drt_integral = self.get_drt_integral(gamma, tau_grid)
        
        # Log verification
        print(f"R_pol from EIS: {self.R_pol:.6f} Ω")
        print(f"R_pol from DRT integral: {drt_integral:.6f} Ω")
        print(f"Ratio (should be ~1.0): {drt_integral/self.R_pol:.4f}")
        print(f"Fitted inductance L: {L_value:.6e} H")
        
        if drt_integral / self.R_pol < 0.8 or drt_integral / self.R_pol > 1.2:
            warnings.warn(f"DRT integral ({drt_integral:.4f}) does not match R_pol ({self.R_pol:.4f}). Ratio: {drt_integral/self.R_pol:.3f}")
        
        gamma_corrected = gamma * np.log(10)  # ln(10) ≈ 2.302585
        
        # Recalculate integral with corrected gamma
        drt_integral_corrected = self.get_drt_integral(gamma_corrected, tau_grid)
        
        return DRTResult(
            tau_grid=tau_grid,
            gamma=gamma_corrected,
            gamma_std=gamma_std * np.log(10),
            method="Tikhonov Regularization with Inductance",
            R_inf=self.R_inf,
            R_pol=self.R_pol,
            L=L_value,
            lambda_opt=lambda_opt,
            metadata={
                'lambda': lambda_opt,
                'order': self.regularization_order,
                'lambda_auto': lambda_auto,
                'drt_integral': drt_integral_corrected,
                'integral_ratio': drt_integral_corrected / self.R_pol,
                'inductance': L_value
            }
        )
    
    def reconstruct_impedance(self, tau_grid: np.ndarray, gamma: np.ndarray, L: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct impedance from DRT and optional inductance.
        
        Args:
            tau_grid: Relaxation time grid
            gamma: DRT values
            L: Inductance value (default 0)
        
        Returns:
            Tuple of (Z_rec_real, Z_rec_imag)
        """
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=False)
        Z_rec_real = self.R_inf + (K_real @ gamma) / np.log(10)
        Z_rec_imag = -(K_imag @ gamma) / np.log(10) + 2 * np.pi * self.frequencies * L
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
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=False)
        
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
        
        # Calculate uncertainty estimate from second derivative
        gamma_std = np.abs(np.gradient(np.gradient(gamma))) * 0.15
        
        # Apply ln(10) correction for consistency with Tikhonov method
        gamma_corrected = gamma * np.log(10)  # ln(10) ≈ 2.302585
        
        # Recalculate integral with corrected gamma
        drt_integral_corrected = self.get_drt_integral(gamma_corrected, tau_grid)
        
        print(f"MaxEntropy DRT - R_pol from EIS: {self.R_pol:.6f} Ω")
        print(f"MaxEntropy DRT - Integral (before correction): {self.get_drt_integral(gamma, tau_grid):.6f} Ω")
        print(f"MaxEntropy DRT - Integral (after correction): {drt_integral_corrected:.6f} Ω")
        print(f"MaxEntropy DRT - Ratio (after correction): {drt_integral_corrected/self.R_pol:.4f}")
        
        return DRTResult(
            tau_grid=tau_grid,
            gamma=gamma_corrected,
            gamma_std=gamma_std * np.log(10),  # также корректируем стандартное отклонение
            method="Maximum Entropy",
            R_inf=self.R_inf,
            R_pol=self.R_pol,
            L=0.0,
            lambda_opt=lambda_opt,
            metadata={
                'lambda': lambda_opt,
                'drt_integral': drt_integral_corrected,
                'integral_ratio': drt_integral_corrected / self.R_pol
            }
        )
    
    def compute_with_inductance(self, n_tau: int = 150, lambda_value: Optional[float] = None,
                                 lambda_auto: bool = True) -> DRTResult:
        """
        Compute DRT with inductance using maximum entropy method.
        Note: Full implementation would require extending the entropy framework.
        For now, this is a placeholder that uses the standard method.
        """
        # For maximum entropy, the extension to include L is non-trivial
        # This is a simplified version that just adds L after the fact
        result = self.compute(n_tau=n_tau, lambda_value=lambda_value, lambda_auto=lambda_auto)
        
        # Estimate L from high frequency data
        has_inductive = self.data.detect_inductive_behavior()
        if has_inductive:
            high_freq_mask = self.frequencies > 0.1 * np.max(self.frequencies)
            if np.any(high_freq_mask):
                omega_high = 2 * np.pi * self.frequencies[high_freq_mask]
                imag_high = -self.Z_imag[high_freq_mask]
                # Estimate L from slope of -Im(Z) vs ω
                if len(omega_high) > 2:
                    slope, _, _, _, _ = linregress(omega_high, imag_high)
                    L_est = -slope  # Since -Im(Z) = -ωL + RC part, slope ≈ -L
                    result.L = max(0, L_est)
        
        result.method = "Maximum Entropy with Inductance (estimated)"
        return result
    
    def reconstruct_impedance(self, tau_grid: np.ndarray, gamma: np.ndarray, L: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct impedance from DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid, include_rl=False)
        Z_rec_real = self.R_inf + (K_real @ gamma) / np.log(10)
        Z_rec_imag = -(K_imag @ gamma) / np.log(10) + 2 * np.pi * self.frequencies * L
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

    def _create_bounds_with_fixed_centers(self, x, y_norm, n_peaks, n_baseline, fixed_centers):
        """
        Create bounds with fixed centers for stage 1 optimization.
        Centers are fixed to initial positions, only amplitudes and sigmas vary.
        
        Args:
            x: x data array
            y_norm: normalized y data
            n_peaks: number of peaks
            n_baseline: number of baseline parameters
            fixed_centers: list of fixed center positions for each peak
        
        Returns:
            lower_bounds, upper_bounds lists
        """
        lower_bounds = []
        upper_bounds = []
        x_range = np.max(x) - np.min(x)
        
        # Peak bounds with fixed centers
        for i in range(n_peaks):
            # Amplitude bounds
            lower_bounds.append(0)
            upper_bounds.append(2 * np.max(y_norm))
            
            # Center bounds - fixed to exact position (very narrow range)
            center_fixed = fixed_centers[i]
            lower_bounds.append(center_fixed - 1e-10)
            upper_bounds.append(center_fixed + 1e-10)
            
            # Sigma bounds
            lower_bounds.append(x_range * 0.001)
            upper_bounds.append(x_range * 0.5)
        
        # Baseline bounds (unchanged)
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
    
    def _create_bounds_with_limited_centers(self, x, y_norm, n_peaks, n_baseline, initial_centers, shift_range):
        """
        Create bounds with limited center movement for stage 2 optimization.
        
        Args:
            x: x data array
            y_norm: normalized y data
            n_peaks: number of peaks
            n_baseline: number of baseline parameters
            initial_centers: list of initial center positions for each peak
            shift_range: maximum allowed shift in x units (e.g., 3 * dx)
        
        Returns:
            lower_bounds, upper_bounds lists
        """
        lower_bounds = []
        upper_bounds = []
        x_range = np.max(x) - np.min(x)
        dx = np.mean(np.diff(x))
        max_shift = shift_range * dx
        
        # Peak bounds with limited center movement
        for i in range(n_peaks):
            # Amplitude bounds
            lower_bounds.append(0)
            upper_bounds.append(2 * np.max(y_norm))
            
            # Center bounds - allow limited shift
            center_initial = initial_centers[i]
            lower_bounds.append(max(np.min(x), center_initial - max_shift))
            upper_bounds.append(min(np.max(x), center_initial + max_shift))
            
            # Sigma bounds
            lower_bounds.append(x_range * 0.001)
            upper_bounds.append(x_range * 0.5)
        
        # Baseline bounds (unchanged)
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
    
    def _check_improvement(self, prev_metrics, curr_metrics, tolerance=1e-4):
        """
        Check if fit improved significantly.
        
        Args:
            prev_metrics: previous quality metrics dict
            curr_metrics: current quality metrics dict
            tolerance: minimum relative improvement in R²
        
        Returns:
            bool: True if improvement is significant
        """
        if prev_metrics is None or 'R²' not in prev_metrics or 'R²' not in curr_metrics:
            return True
        
        prev_r2 = prev_metrics.get('R²', 0)
        curr_r2 = curr_metrics.get('R²', 0)
        
        # Significant improvement if R² increased by more than tolerance
        return (curr_r2 - prev_r2) > tolerance
    
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
        
        # Calculate the integral of DRT to verify scaling
        if self.use_log_x:
            # Integration over d(ln τ) - this should equal R_pol
            self.drt_integral = np.trapezoid(self.y_original, np.log(self.x_linear))
        else:
            # Integration over dτ (not correct, for reference only)
            self.drt_integral = np.trapezoid(self.y_original, self.x_linear)
        
        print(f"DRT integral before deconvolution: {self.drt_integral:.6f} Ω")
        print(f"This should equal the polarization resistance R_pol from EIS")
        
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
        self.fit_y_original = None  # Added for original scale fit
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
            fit_quality='balanced', last_popt=None, progress_callback=None,
            use_iterative_refinement=True):
        """
        Perform fitting with iterative refinement (Stage 1: fixed centers, 
        Stage 2: limited centers, Stage 3: free centers).
        
        Args:
            use_iterative_refinement: If True, use 3-stage optimization.
                                     If False, use single-stage optimization.
        """
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks()
        
        if len(initial_params) == 0:
            if progress_callback:
                progress_callback(1.0, "No peaks detected!")
            return False
        
        n_peaks = len(initial_params) // 3
        n_baseline = self._get_n_baseline_params()
        
        # Extract initial centers
        initial_centers = [initial_params[3*i + 1] for i in range(n_peaks)]
        
        # Use last good parameters if available
        if last_popt is not None and not use_iterative_refinement:
            expected_len = n_peaks * 3 + n_baseline
            if len(last_popt) == expected_len:
                initial_params = last_popt.copy()
            else:
                initial_params = self._prepare_initial_params(initial_params, n_baseline)
        else:
            initial_params = self._prepare_initial_params(initial_params, n_baseline)
        
        # Set tolerances based on fit quality
        if fit_quality == 'fast':
            xtol, ftol, gtol = 1e-3, 1e-3, 1e-3
            maxfev_stage = min(maxfev // 3, 2000)
        elif fit_quality == 'balanced':
            xtol, ftol, gtol = 1e-5, 1e-5, 1e-5
            maxfev_stage = min(maxfev // 3, 3000)
        else:  # precise
            xtol, ftol, gtol = 1e-8, 1e-8, 1e-8
            maxfev_stage = min(maxfev // 3, 4000)
        
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
        
        # If not using iterative refinement, do single-stage fit
        if not use_iterative_refinement:
            lower_bounds, upper_bounds = self._create_bounds(self.x, self.y_norm, n_peaks, n_baseline)
            
            for i in range(len(initial_params)):
                initial_params[i] = np.clip(initial_params[i], lower_bounds[i], upper_bounds[i])
            
            try:
                if progress_callback:
                    progress_callback(0.3, "Initializing fit...")
                
                if progress_callback:
                    progress_callback(0.5, "Running curve_fit...")
                
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
                
                return self._process_fit_result(popt, n_peaks, n_baseline, progress_callback)
                
            except Exception as e:
                if progress_callback:
                    progress_callback(1.0, f"Fit failed: {e}")
                return False
        
        # ITERATIVE REFINEMENT - 3 STAGES
        current_params = initial_params.copy()
        prev_metrics = None
        stage_results = []
        
        # Stage 1: Fixed centers (only amplitudes and sigmas vary)
        if progress_callback:
            progress_callback(0.1, "Stage 1/3: Optimizing amplitudes and widths (centers fixed)...")
        
        try:
            lower_bounds_fixed, upper_bounds_fixed = self._create_bounds_with_fixed_centers(
                self.x, self.y_norm, n_peaks, n_baseline, initial_centers
            )
            
            for i in range(len(current_params)):
                current_params[i] = np.clip(current_params[i], lower_bounds_fixed[i], upper_bounds_fixed[i])
            
            popt_stage1, _ = curve_fit(
                model_func,
                self.x,
                self.y_norm,
                p0=current_params,
                bounds=(lower_bounds_fixed, upper_bounds_fixed),
                method=method,
                maxfev=maxfev_stage,
                xtol=xtol,
                ftol=ftol,
                gtol=gtol
            )
            
            current_params = popt_stage1.copy()
            
            # Calculate metrics for stage 1
            fit_y_norm_stage1 = model_func(self.x, *current_params)
            metrics_stage1 = FitQualityAnalyzer.calculate_metrics(
                self.y_norm, fit_y_norm_stage1, len(current_params)
            )
            stage_results.append(('Stage 1 (fixed centers)', metrics_stage1))
            prev_metrics = metrics_stage1
            
            if progress_callback:
                progress_callback(0.3, f"Stage 1 complete. R² = {metrics_stage1.get('R²', 0):.6f}")
            
        except Exception as e:
            if progress_callback:
                progress_callback(0.3, f"Stage 1 failed: {e}")
            # Fall back to regular fit
            return self._fallback_fit(model_func, initial_params, n_peaks, n_baseline, 
                                      method, maxfev, xtol, ftol, gtol, progress_callback)
        
        # Stage 2: Limited centers (±3 points)
        if progress_callback:
            progress_callback(0.4, "Stage 2/3: Refining peak positions (limited shift)...")
        
        try:
            # Extract updated centers from stage 1
            updated_centers = [current_params[3*i + 1] for i in range(n_peaks)]
            
            lower_bounds_limited, upper_bounds_limited = self._create_bounds_with_limited_centers(
                self.x, self.y_norm, n_peaks, n_baseline, updated_centers, shift_range=3
            )
            
            for i in range(len(current_params)):
                current_params[i] = np.clip(current_params[i], lower_bounds_limited[i], upper_bounds_limited[i])
            
            popt_stage2, _ = curve_fit(
                model_func,
                self.x,
                self.y_norm,
                p0=current_params,
                bounds=(lower_bounds_limited, upper_bounds_limited),
                method=method,
                maxfev=maxfev_stage,
                xtol=xtol,
                ftol=ftol,
                gtol=gtol
            )
            
            current_params = popt_stage2.copy()
            
            # Calculate metrics for stage 2
            fit_y_norm_stage2 = model_func(self.x, *current_params)
            metrics_stage2 = FitQualityAnalyzer.calculate_metrics(
                self.y_norm, fit_y_norm_stage2, len(current_params)
            )
            stage_results.append(('Stage 2 (limited centers)', metrics_stage2))
            
            # Check if improvement is significant
            if not self._check_improvement(prev_metrics, metrics_stage2):
                if progress_callback:
                    progress_callback(0.6, "Stage 2: No significant improvement, stopping early.")
            else:
                prev_metrics = metrics_stage2
            
            if progress_callback:
                progress_callback(0.6, f"Stage 2 complete. R² = {metrics_stage2.get('R²', 0):.6f}")
            
        except Exception as e:
            if progress_callback:
                progress_callback(0.6, f"Stage 2 failed: {e}, continuing with stage 1 results...")
        
        # Stage 3: Full freedom
        if progress_callback:
            progress_callback(0.7, "Stage 3/3: Final optimization (full freedom)...")
        
        try:
            lower_bounds_free, upper_bounds_free = self._create_bounds(
                self.x, self.y_norm, n_peaks, n_baseline
            )
            
            for i in range(len(current_params)):
                current_params[i] = np.clip(current_params[i], lower_bounds_free[i], upper_bounds_free[i])
            
            popt_stage3, _ = curve_fit(
                model_func,
                self.x,
                self.y_norm,
                p0=current_params,
                bounds=(lower_bounds_free, upper_bounds_free),
                method=method,
                maxfev=maxfev_stage,
                xtol=xtol,
                ftol=ftol,
                gtol=gtol
            )
            
            current_params = popt_stage3.copy()
            
            # Calculate metrics for stage 3
            fit_y_norm_stage3 = model_func(self.x, *current_params)
            metrics_stage3 = FitQualityAnalyzer.calculate_metrics(
                self.y_norm, fit_y_norm_stage3, len(current_params)
            )
            stage_results.append(('Stage 3 (full freedom)', metrics_stage3))
            
            if progress_callback:
                progress_callback(0.9, f"Stage 3 complete. R² = {metrics_stage3.get('R²', 0):.6f}")
            
        except Exception as e:
            if progress_callback:
                progress_callback(0.9, f"Stage 3 failed: {e}, using stage 2 results...")
        
        # Log stage results for debugging
        if self.show_warnings:
            print("\n=== Iterative Refinement Results ===")
            for stage_name, metrics in stage_results:
                print(f"{stage_name}: R² = {metrics.get('R²', 0):.8f}, RMSE = {metrics.get('RMSE', 0):.2e}")
        
        # Process final result
        if progress_callback:
            progress_callback(0.95, "Calculating components...")
        
        return self._process_fit_result(current_params, n_peaks, n_baseline, progress_callback)
    
    def _process_fit_result(self, popt, n_peaks, n_baseline, progress_callback=None):
        """
        Process fit result: extract components, calculate metrics, store results.
        
        Args:
            popt: optimized parameters from curve_fit
            n_peaks: number of peaks
            n_baseline: number of baseline parameters
            progress_callback: optional callback for progress updates
        
        Returns:
            bool: True if successful
        """
        # Calculate fit in normalized space
        def model_func(x, *params):
            if n_baseline == 0:
                return GaussianModel.multi_gaussian(x, *params)
            else:
                peak_params = params[:n_peaks*3]
                baseline_params = params[n_peaks*3:]
                return GaussianModel.multi_gaussian_with_baseline(
                    x, n_peaks, peak_params, baseline_params, self.baseline_method
                )
        
        fit_y_norm = model_func(self.x, *popt)
        # Calculate fit in original space
        fit_y_original = fit_y_norm * self.y_max
        
        # Extract components
        components = []
        peak_params = popt[:n_peaks*3]
        baseline_params = popt[n_peaks*3:] if n_baseline > 0 else []
        
        for i in range(n_peaks):
            amp_norm = peak_params[3*i]
            cen = peak_params[3*i + 1]
            sigma = abs(peak_params[3*i + 2])
            
            amp = amp_norm * self.y_max
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
        self.fit_y_original = fit_y_original
        self.total_area = total_area
        
        # Calculate quality metrics
        self.quality_metrics = FitQualityAnalyzer.calculate_metrics(
            self.y_norm, self.fit_y_norm, len(popt)
        )
        
        if progress_callback:
            progress_callback(1.0, "Fit complete!")
        
        return True
    
    def _fallback_fit(self, model_func, initial_params, n_peaks, n_baseline, 
                      method, maxfev, xtol, ftol, gtol, progress_callback):
        """Fallback to regular fit if iterative refinement fails"""
        try:
            lower_bounds, upper_bounds = self._create_bounds(self.x, self.y_norm, n_peaks, n_baseline)
            
            for i in range(len(initial_params)):
                initial_params[i] = np.clip(initial_params[i], lower_bounds[i], upper_bounds[i])
            
            popt, _ = curve_fit(
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
            
            return self._process_fit_result(popt, n_peaks, n_baseline, progress_callback)
            
        except Exception as e:
            if progress_callback:
                progress_callback(1.0, f"Fallback fit failed: {e}")
            return False
    
    def create_deconvolution_result(self) -> DeconvolutionResult:
        """Create result container from current components with peaks sorted by frequency (high to low)"""
        # Sort components by characteristic frequency (high to low) and reassign IDs
        components_with_freq = []
        for c in self.components:
            freq = 1.0 / (2 * np.pi * c['cen_linear'])
            components_with_freq.append((freq, c))
        
        # Sort by frequency descending (high to low)
        components_with_freq.sort(key=lambda x: x[0], reverse=True)
        
        # Create peaks with new IDs in sorted order
        peaks = []
        for new_id, (freq, c) in enumerate(components_with_freq, start=1):
            peak = GaussianPeak(
                id=new_id,
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
                y_norm=c['y_norm'],
                characteristic_frequency=freq
            )
            peaks.append(peak)
        
        # Use original y_original values
        y_original_restored = self.y_original.copy()
        
        # Calculate total area from components (should be close to drt_integral)
        total_component_area = sum([p.area for p in peaks])
        
        print(f"Original DRT integral: {self.drt_integral:.6f} Ω")
        print(f"Total area from deconvolution: {total_component_area:.6f} Ω")
        print(f"Ratio (should be ~1.0): {total_component_area/self.drt_integral:.4f}")
        
        # Reorder fit_y_original according to sorted peaks if needed for consistent display
        # For fit_y_original, we need to keep it as is (it's the total fit, not per-peak)
        
        return DeconvolutionResult(
            peaks=peaks,
            fit_y_norm=self.fit_y_norm if self.fit_y_norm is not None else np.zeros_like(self.x),
            fit_y_original=self.fit_y_original if self.fit_y_original is not None else None,
            x=self.x,
            y_norm=self.y_norm,
            y_original=y_original_restored,
            x_linear=self.x_linear,
            use_log_x=self.use_log_x,
            use_log_y=self.use_log_y,
            quality_metrics=self.quality_metrics,
            baseline_params=self.baseline_params,
            baseline_method=self.baseline_method,
            total_area=self.drt_integral,  # Use the original DRT integral as total area
            max_amplitude=max([p.amplitude for p in peaks]) if peaks else 0
        )
        
    def calculate_peak_area(self, amplitude: float, sigma: float, is_log_scale: bool = True) -> float:
        """
        Calculate area under Gaussian peak.
        
        For peaks in log space: area = amplitude * sigma * sqrt(2π)
        This area corresponds to contribution to polarization resistance in Ω.
        
        Args:
            amplitude: Peak amplitude in original units (Ω)
            sigma: Standard deviation in log space
            is_log_scale: Whether X is in log scale (should be True for DRT)
            
        Returns:
            Area under peak (Ω)
        """
        if is_log_scale:
            # For log-scale X, area = amplitude * sigma * sqrt(2π)
            # This integrates over d(ln τ)
            return amplitude * sigma * np.sqrt(2 * np.pi)
        else:
            # For linear scale (not typical for DRT)
            return amplitude * sigma * np.sqrt(2 * np.pi)
    
    def get_total_resistance(self) -> float:
        """
        Get total polarization resistance from DRT integral.
        
        Returns:
            float: Total polarization resistance in Ω
        """
        return self.drt_integral


# ============================================================================
# Helper function for number formatting with superscript
# ============================================================================

def format_with_superscript(number: float) -> str:
    """
    Format a number with superscript for powers of 10.
    Example: 1.23e-4 -> "1.23·10⁻⁴"
    """
    if number == 0:
        return "0"
    
    # Handle numbers that don't need scientific notation
    if 0.001 <= abs(number) <= 10000:
        # Check if it's a round number
        if abs(number - round(number)) < 1e-10:
            return str(int(round(number)))
        # Format with reasonable precision
        if abs(number) >= 1:
            return f"{number:.2f}".rstrip('0').rstrip('.')
        else:
            # For small numbers, use scientific notation with superscript
            pass
    
    # Scientific notation with superscript
    exponent = int(np.floor(np.log10(abs(number))))
    mantissa = number / (10 ** exponent)
    
    # Format mantissa
    if abs(mantissa - round(mantissa)) < 1e-10:
        mantissa_str = str(int(round(mantissa)))
    else:
        mantissa_str = f"{mantissa:.2f}".rstrip('0').rstrip('.')
    
    # Superscript mapping
    sup_map = {
        '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
    }
    
    # Handle negative exponent
    if exponent < 0:
        exp_str = ''.join(sup_map.get(c, c) for c in str(abs(exponent)))
        return f"{mantissa_str}·10⁻{exp_str}"
    else:
        exp_str = ''.join(sup_map.get(c, c) for c in str(exponent))
        return f"{mantissa_str}·10{exp_str}"


# ============================================================================
# Visualization Functions (from both codes)
# ============================================================================

def plot_nyquist_matplotlib(data: ImpedanceData, re_rec: Optional[np.ndarray] = None, 
                           im_rec: Optional[np.ndarray] = None, title: str = "Nyquist Plot",
                           highlight_idx: Optional[int] = None,
                           freq_range_idx: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """Create publication-quality Nyquist plot"""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Make axis tick labels black
    ax.tick_params(axis='both', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    
    is_cropped = (
        freq_range_idx is not None and 
        (freq_range_idx[0] > 0 or freq_range_idx[1] < data.n_points - 1)
    )
    
    if is_cropped:
        f_min_i, f_max_i = freq_range_idx
        ax.plot(data.re_z, data.im_z, 'o--', markersize=4, linewidth=1.0,
                alpha=0.35, color='#7f8c8d', label='Excluded', zorder=1)
        ax.plot(data.re_z[f_min_i:f_max_i + 1], data.im_z[f_min_i:f_max_i + 1],
                'o-', markersize=5, linewidth=1.8, label='Experimental (active)',
                color='#1f77b4', markeredgecolor='white', markeredgewidth=0.8, zorder=2)
        ax.plot([data.re_z[f_min_i], data.re_z[f_max_i]],
                [data.im_z[f_min_i], data.im_z[f_max_i]],
                's', markersize=7, color='#4a5568', markeredgecolor='black',
                label='Range bounds', zorder=3)
    else:
        ax.plot(data.re_z, data.im_z, 'o-', markersize=5, linewidth=1.8, 
                label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.8)
    
    if highlight_idx is not None and 0 <= highlight_idx < data.n_points:
        ax.plot(data.re_z[highlight_idx], data.im_z[highlight_idx], 'ro', 
                markersize=12, markeredgecolor='red', markerfacecolor='none', linewidth=2.5,
                label='Selected Point')
    
    if re_rec is not None and im_rec is not None:
        ax.plot(re_rec, im_rec, 's-', markersize=4, linewidth=1.2,
                label='Reconstructed', color='#ff7f0e', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    
    style = get_plot_style()
    ax.set_xlabel("Re(Z) / Ohm", fontweight='bold')
    ax.set_ylabel("-Im(Z) / Ohm", fontweight='bold')
    ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='black')
    ax.grid(style["show_grid"], alpha=0.3, linestyle='--')
    
    # Scientific formatting
    ax.ticklabel_format(style='scientific', scilimits=(-2, 2))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    ax.set_aspect('equal', adjustable='box')
    
    return fig


def plot_bode_matplotlib(data: ImpedanceData, re_rec: Optional[np.ndarray] = None, 
                         im_rec: Optional[np.ndarray] = None,
                         highlight_idx: Optional[int] = None,
                         freq_range_idx: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """Create publication-quality Bode plot with highlight point"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    
    # Make axis tick labels black
    ax1.tick_params(axis='both', colors='black')
    ax2.tick_params(axis='both', colors='black')
    ax1.xaxis.label.set_color('black')
    ax1.yaxis.label.set_color('black')
    ax2.xaxis.label.set_color('black')
    ax2.yaxis.label.set_color('black')
    
    is_cropped = (
        freq_range_idx is not None and 
        (freq_range_idx[0] > 0 or freq_range_idx[1] < data.n_points - 1)
    )
    
    mag = data.Z_mod
    if is_cropped:
        f_min_i, f_max_i = freq_range_idx
        f_start, f_end = data.freq[f_min_i], data.freq[f_max_i]
        if f_min_i > 0:
            ax1.axvspan(data.freq[0], f_start, color='#e2e8f0', alpha=0.65, zorder=0)
            ax1.axvline(x=f_start, color='#4a5568', linestyle=':', linewidth=1.5, zorder=2)
        if f_max_i < data.n_points - 1:
            ax1.axvspan(f_end, data.freq[-1], color='#e2e8f0', alpha=0.65, zorder=0)
            ax1.axvline(x=f_end, color='#4a5568', linestyle=':', linewidth=1.5, zorder=2)
        ax1.loglog(data.freq, mag, 'o--', markersize=4, linewidth=1.0,
                   alpha=0.35, color='#7f8c8d', label='Excluded', zorder=1)
        ax1.loglog(data.freq[f_min_i:f_max_i + 1], mag[f_min_i:f_max_i + 1],
                   'o-', markersize=5, linewidth=1.8, label='Experimental (active)',
                   color='#1f77b4', markeredgecolor='white', markeredgewidth=0.8, zorder=2)
    else:
        ax1.loglog(data.freq, mag, 'o-', markersize=5, linewidth=1.8, 
                   label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.8)
    
    if highlight_idx is not None and 0 <= highlight_idx < data.n_points:
        ax1.loglog(data.freq[highlight_idx], mag[highlight_idx], 'ro', 
                   markersize=12, markeredgecolor='red', markerfacecolor='none', linewidth=2.5)
    
    if re_rec is not None and im_rec is not None:
        mag_rec = np.sqrt(re_rec**2 + im_rec**2)
        ax1.loglog(data.freq, mag_rec, 's-', markersize=4, linewidth=1.2,
                   label='Reconstructed', color='#ff7f0e', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    
    style = get_plot_style()
    ax1.set_xlabel("Frequency / Hz", fontweight='bold')
    ax1.set_ylabel("|Z| / Ohm", fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(style["show_grid"], alpha=0.3, linestyle='--')
    
    # Scientific formatting for log axes
    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    
    phase = data.phase
    if is_cropped:
        f_min_i, f_max_i = freq_range_idx
        f_start, f_end = data.freq[f_min_i], data.freq[f_max_i]
        if f_min_i > 0:
            ax2.axvspan(data.freq[0], f_start, color='#e2e8f0', alpha=0.65, zorder=0)
            ax2.axvline(x=f_start, color='#4a5568', linestyle=':', linewidth=1.5, zorder=2)
        if f_max_i < data.n_points - 1:
            ax2.axvspan(f_end, data.freq[-1], color='#e2e8f0', alpha=0.65, zorder=0)
            ax2.axvline(x=f_end, color='#4a5568', linestyle=':', linewidth=1.5, zorder=2)
        ax2.semilogx(data.freq, phase, 'o--', markersize=4, linewidth=1.0,
                     alpha=0.35, color='#7f8c8d', label='Excluded', zorder=1)
        ax2.semilogx(data.freq[f_min_i:f_max_i + 1], phase[f_min_i:f_max_i + 1],
                     'o-', markersize=5, linewidth=1.8, label='Experimental (active)',
                     color='#1f77b4', markeredgecolor='white', markeredgewidth=0.8, zorder=2)
    else:
        ax2.semilogx(data.freq, phase, 'o-', markersize=5, linewidth=1.8,
                     label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.8)
    
    if highlight_idx is not None and 0 <= highlight_idx < data.n_points:
        ax2.semilogx(data.freq[highlight_idx], phase[highlight_idx], 'ro', 
                     markersize=12, markeredgecolor='red', markerfacecolor='none', linewidth=2.5)
    
    if re_rec is not None and im_rec is not None:
        phase_rec = np.arctan2(im_rec, re_rec) * 180 / np.pi
        ax2.semilogx(data.freq, phase_rec, 's-', markersize=4, linewidth=1.2,
                     label='Reconstructed', color='#ff7f0e', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    
    ax2.set_xlabel("Frequency / Hz", fontweight='bold')
    ax2.set_ylabel("Phase / deg", fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(style["show_grid"], alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Scientific formatting - отключаем LaTeX
    ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    
    # No title - removed as requested
    plt.tight_layout()
    
    return fig


def plot_drt_matplotlib(result: DRTResult, peaks: Optional[List[Dict[str, Any]]] = None,
                       title: str = "Distribution of Relaxation Times") -> plt.Figure:
    """Create publication-quality DRT plot with both tau and frequency axes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Make axis tick labels black
    ax1.tick_params(axis='both', colors='black')
    ax2.tick_params(axis='both', colors='black')
    ax1.xaxis.label.set_color('black')
    ax1.yaxis.label.set_color('black')
    ax2.xaxis.label.set_color('black')
    ax2.yaxis.label.set_color('black')
    
    # Отключаем LaTeX для всех текстовых элементов
    plt.rcParams['text.usetex'] = False
    plt.rcParams['axes.formatter.use_mathtext'] = False
    
    if result.gamma_std is not None:
        ax1.fill_between(result.tau_grid, result.gamma - 2*result.gamma_std, 
                        result.gamma + 2*result.gamma_std,
                        alpha=0.3, color='gray')  # Убран параметр label
    ax1.semilogx(result.tau_grid, result.gamma, '-', linewidth=2, color='#2ca02c', label='DRT')
    
    if peaks and len(peaks) > 0:
        peak_tau = [p['tau'] for p in peaks]
        peak_drt = [p['amplitude'] for p in peaks]
        ax1.plot(peak_tau, peak_drt, 'rv', markersize=8, label='Detected peaks')
        
        for i, (t, d) in enumerate(zip(peak_tau, peak_drt)):
            freq = 1/(2*np.pi*t)
            # Используем обычный текст без LaTeX
            ax1.annotate(f'tau={t:.2e}s\nf={freq:.2e}Hz',
                       xy=(t, d), xytext=(t*1.5, d*1.2),
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Используем обычный текст без LaTeX
    ax1.set_xlabel("Relaxation Time (s)", fontweight='bold')
    ax1.set_ylabel("gamma (Ohm)", fontweight='bold')
    # No title - removed as requested
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
                        alpha=0.3, color='gray')  # Убран параметр label
    ax2.semilogx(freqs_sorted, gamma_sorted, '-', linewidth=2, color='#2ca02c', label='DRT')
    
    if peaks and len(peaks) > 0:
        peak_freqs = [p['frequency'] for p in peaks]
        peak_amplitudes = [p['amplitude'] for p in peaks]
        peak_pairs = sorted(zip(peak_freqs, peak_amplitudes), key=lambda x: x[0], reverse=True)
        peak_freqs_sorted, peak_amplitudes_sorted = zip(*peak_pairs) if peak_pairs else ([], [])
        
        ax2.plot(peak_freqs_sorted, peak_amplitudes_sorted, 'rv', markersize=8, label='Detected peaks')
        
        for i, (f, d) in enumerate(zip(peak_freqs_sorted, peak_amplitudes_sorted)):
            tau_val = 1/(2*np.pi*f)
            # Используем обычный текст без LaTeX
            ax2.annotate(f'f={f:.2e}Hz\ntau={tau_val:.2e}s',
                       xy=(f, d), xytext=(f*1.5, d*1.2),
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax2.set_xlabel("Frequency (Hz)", fontweight='bold')
    ax2.set_ylabel("gamma(tau) (Ohm)", fontweight='bold')
    # No title - removed as requested
    ax2.legend(loc='best', frameon=True)
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')
    ax2.set_xscale('log')
    ax2.invert_xaxis()
    ax2.set_xlim(freqs_sorted[0], freqs_sorted[-1])
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    
    # No overall figure title - removed as requested
    plt.tight_layout()
    
    return fig

def plot_deconvolution_result(deconv_result: DeconvolutionResult, show_components: bool = True,
                              show_baseline: bool = True, title: str = "Gaussian Deconvolution Result",
                              preview_mode: bool = False, preview_fit: Optional[np.ndarray] = None) -> plt.Figure:
    """Plot deconvolution result with components and baseline"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Make axis tick labels black
    ax1.tick_params(axis='both', colors='black')
    ax2.tick_params(axis='both', colors='black')
    ax1.xaxis.label.set_color('black')
    ax1.yaxis.label.set_color('black')
    ax2.xaxis.label.set_color('black')
    ax2.yaxis.label.set_color('black')
    
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
    
    # Define a consistent color mapping based on peak ID
    colors = plt.cm.Set3(np.linspace(0, 1, len(deconv_result.peaks)))
    color_map = {peak.id: colors[i] for i, peak in enumerate(deconv_result.peaks)}
    
    if show_components and deconv_result.peaks:
        for peak in deconv_result.peaks:
            color = color_map[peak.id]
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
    elif deconv_result.fit_y_original is not None:
        # Use stored original scale fit and interpolate for smooth curve
        from scipy.interpolate import interp1d
        interp_func = interp1d(deconv_result.x_linear, deconv_result.fit_y_original, 
                               kind='cubic', fill_value='extrapolate')
        y_total_interp = interp_func(x_dense)
        ax1.plot(x_dense, y_total_interp, 'r--', linewidth=2.5, label='Total Fit', zorder=3)
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
    # No title - removed as requested
    ax1.legend(loc='upper right', fontsize=9, frameon=True, edgecolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right plot: Area distribution
    if deconv_result.peaks:
        # Sort peaks by ID (which are already sorted by frequency high to low)
        peaks_sorted = sorted(deconv_result.peaks, key=lambda p: p.id)
        peaks_ids = [f'Peak {p.id}' for p in peaks_sorted]
        fractions = [p.fraction_percent for p in peaks_sorted]
        colors_sorted = [color_map[p.id] for p in peaks_sorted]
        
        bars = ax2.bar(peaks_ids, fractions, color=colors_sorted, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Peak', fontweight='bold')
        ax2.set_ylabel('Fraction (%)', fontweight='bold')
        # No title - removed as requested
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
    
    # Make axis tick labels black
    ax.tick_params(axis='both', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    
    if deconv_result.use_log_x:
        ax.set_xscale('log')
    
    x_dense = np.linspace(np.min(deconv_result.x_linear), np.max(deconv_result.x_linear), 2000)
    if deconv_result.use_log_x:
        x_dense_log = np.log10(x_dense)
    else:
        x_dense_log = x_dense
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(deconv_result.peaks)))
    color_map = {peak.id: colors[i] for i, peak in enumerate(deconv_result.peaks)}
    
    for peak in deconv_result.peaks:
        color = color_map[peak.id]
        y_component = GaussianModelDeconv.gaussian(x_dense_log, peak.amplitude_norm, 
                                                   peak.center_log, peak.sigma_log)
        y_component_norm = y_component / max(peak.amplitude_norm, 1e-10)
        
        ax.plot(x_dense, y_component_norm, '-', color=color, linewidth=2,
               label=f'Peak {peak.id} (f={peak.get_characteristic_frequency():.2e} Hz)')
        ax.axvline(x=peak.center, color=color, linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('X' + (' (log scale)' if deconv_result.use_log_x else ''), fontweight='bold')
    ax.set_ylabel('Normalized Intensity', fontweight='bold')
    # No title - removed as requested
    ax.legend(loc='upper right', fontsize=9, frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def calculate_peak_characteristics(peak: GaussianPeak) -> Dict[str, float]:
    """
    Calculate characteristic frequency and capacitance for a Gaussian peak.
    
    Args:
        peak: GaussianPeak object with center (τ) and area (R)
    
    Returns:
        Dictionary with 'fmax_hz' and 'c_farad'
    """
    # Characteristic frequency: f = 1/(2πτ)
    fmax_hz = 1.0 / (2 * np.pi * peak.center)
    
    # Characteristic capacitance: C = τ/R = 1/(2π·f·R)
    # Using R = area (polarization resistance for this process)
    if peak.area > 0:
        c_farad = peak.center / peak.area  # C = τ/R
        # Alternative: c_farad = 1.0 / (2 * np.pi * fmax_hz * peak.area)
    else:
        c_farad = 0.0
    
    return {'fmax_hz': fmax_hz, 'c_farad': c_farad}


def calculate_rq_parameters(peak: GaussianPeak) -> Dict[str, float]:
    """
    Calculate equivalent ZARC (parallel R||Q) parameters (R_i, Q_i, alpha_i, C_i, f_i)
    for a distributed Gaussian DRT peak.
    """
    R_i = float(peak.area)
    tau_i = float(peak.center)
    sigma = max(float(peak.sigma_log), 1e-6)
    center_log = float(peak.center_log)
    
    if R_i <= 0 or tau_i <= 0:
        return {'R_i': 0.0, 'Q_i': 0.0, 'alpha_i': 1.0, 'C_i': 0.0, 'f_i': 0.0}
    
    # For very narrow Gaussian peak, process is an ideal capacitor (alpha = 1, Q = C = tau/R)
    if sigma < 1e-4:
        alpha_i = 1.0
    else:
        u_grid = np.linspace(center_log - 5.0 * sigma, center_log + 5.0 * sigma, 401)
        ratio = 10.0 ** (u_grid - center_log)  # tau / tau_i
        
        gauss_profile = np.exp(-0.5 * ((u_grid - center_log) / sigma) ** 2)
        profile_area = np.trapezoid(gauss_profile, u_grid)
        
        if profile_area > 0:
            kernel_apex = ratio / (1.0 + ratio ** 2)
            norm_apex = np.trapezoid(gauss_profile * kernel_apex, u_grid) / profile_area
            alpha_i = float(np.clip((4.0 / np.pi) * np.arctan(2.0 * norm_apex), 1e-4, 1.0))
        else:
            alpha_i = 1.0
    
    # CPE parameter Q_i from tau_i^alpha_i = R_i * Q_i  (units: F·s^(alpha-1) = S·s^alpha)
    Q_i = (tau_i ** alpha_i) / R_i
    
    # Effective capacitance: C_i = ((R_i * Q_i)^(1 / alpha_i)) / R_i
    rq_product = R_i * Q_i
    C_i = (rq_product ** (1.0 / alpha_i)) / R_i if R_i > 0 else 0.0
    
    # Characteristic frequency: f_i = ((R_i * Q_i)^(-1 / alpha_i)) / (2 * pi)
    f_i = (rq_product ** (-1.0 / alpha_i)) / (2.0 * np.pi) if rq_product > 0 else 0.0
    
    return {
        'R_i': R_i,
        'Q_i': float(Q_i),
        'alpha_i': alpha_i,
        'C_i': float(C_i),
        'f_i': float(f_i)
    }
    
def refine_rq_zarc_model(deconv_result: DeconvolutionResult,
                         drt_result: DRTResult,
                         data: ImpedanceData) -> Dict[str, Any]:
    """
    Complex Non-Linear Least Squares (CNLS) refinement of the sequential RQ (ZARC) model:
        Z(w) = R_inf + i*w*L + sum_k [ R_k / (1 + (i*w)^alpha_k * R_k * Q_k) ]
    Uses DRT deconvolution results as initial guesses.
    """
    sort_idx = np.argsort(data.freq)
    freq_exp = data.freq[sort_idx]
    re_exp = data.re_z[sort_idx]
    im_exp = data.im_z[sort_idx]  # -Im(Z)
    omega = 2.0 * np.pi * freq_exp

    # Initial R_inf (zero-crossing if inductive, else drt_result.R_inf)
    R_inf_init = float(drt_result.R_inf)
    for i in range(len(im_exp) - 1):
        if im_exp[i] >= 0 and im_exp[i + 1] < 0:
            t = -im_exp[i] / (im_exp[i + 1] - im_exp[i])
            R_inf_init = float(re_exp[i] + t * (re_exp[i + 1] - re_exp[i]))
            break

    has_L = (drt_result.L > 0) or np.any(im_exp < 0)
    L_init = max(float(drt_result.L), 1e-12) if has_L else 0.0

    peaks_sorted = sorted(deconv_result.peaks, key=lambda p: p.id)
    n_proc = len(peaks_sorted)

    # Parameter vector: [R_inf, (L if has_L), R_1, log10(Q_1), alpha_1, ...]
    p0 = [max(R_inf_init, 1e-9)]
    lb = [0.0]
    ub = [max(np.max(re_exp), R_inf_init * 2.0, 1.0)]

    if has_L:
        p0.append(L_init)
        lb.append(0.0)
        ub.append(max(L_init * 100.0, 1e-2))

    R_total_est = max(np.max(re_exp) - np.min(re_exp), 1e-6)
    for peak in peaks_sorted:
        rq = calculate_rq_parameters(peak)
        R_k = max(rq['R_i'], 1e-9)
        Q_k = max(rq['Q_i'], 1e-15)
        alpha_k = float(np.clip(rq['alpha_i'], 0.15, 0.999))

        p0.extend([R_k, np.log10(Q_k), alpha_k])
        lb.extend([1e-9, -15.0, 0.1])
        ub.extend([R_total_est * 3.0, 3.0, 1.0])

    def eval_z_model(params, w_arr):
        idx = 0
        r_inf = params[idx]; idx += 1
        l_val = params[idx] if has_L else 0.0
        if has_L:
            idx += 1

        Z = np.full_like(w_arr, r_inf, dtype=complex)
        if l_val > 0:
            Z += 1j * w_arr * l_val

        for k in range(n_proc):
            R_k = params[idx]
            Q_k = 10.0 ** params[idx + 1]
            alpha_k = params[idx + 2]
            idx += 3
            Z += R_k / (1.0 + ((1j * w_arr) ** alpha_k) * R_k * Q_k)
        return Z

    def residuals_func(params):
        Z_mod = eval_z_model(params, omega)
        res_re = re_exp - np.real(Z_mod)
        res_im = im_exp - (-np.imag(Z_mod))
        return np.concatenate([res_re, res_im])

    res = least_squares(
        residuals_func,
        x0=np.clip(p0, lb, ub),
        bounds=(lb, ub),
        method='trf',
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=10000
    )

    opt = res.x
    idx = 0
    r_inf_opt = float(opt[idx]); idx += 1
    l_opt = float(opt[idx]) if has_L else 0.0
    if has_L:
        idx += 1

    processes = []
    for k, peak in enumerate(peaks_sorted):
        R_k = float(opt[idx])
        Q_k = float(10.0 ** opt[idx + 1])
        alpha_k = float(opt[idx + 2])
        idx += 3

        rq_prod = R_k * Q_k
        C_k = float((rq_prod ** (1.0 / alpha_k)) / R_k) if R_k > 0 else 0.0
        f_k = float((rq_prod ** (-1.0 / alpha_k)) / (2.0 * np.pi)) if rq_prod > 0 else 0.0

        processes.append({
            'id': peak.id,
            'R_i': R_k,
            'Q_i': Q_k,
            'alpha_i': alpha_k,
            'C_i': C_k,
            'f_i': f_k
        })

    # Re-sort processes by characteristic frequency (high to low) and reassign IDs
    processes.sort(key=lambda p: p['f_i'], reverse=True)
    for new_id, p in enumerate(processes, start=1):
        p['id'] = new_id

    # Fit quality metrics
    Z_opt = eval_z_model(opt, omega)
    y_true = np.concatenate([re_exp, im_exp])
    y_pred = np.concatenate([np.real(Z_opt), -np.imag(Z_opt)])
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        'R_inf': r_inf_opt,
        'L': l_opt,
        'processes': processes,
        'r2': r2,
        'rmse': rmse,
        'success': bool(res.success)
    }

def plot_original_nyquist_with_frequency_labels(data: ImpedanceData, title: str = "Original Impedance Spectrum") -> plt.Figure:
    """
    Plot original Nyquist spectrum with proper sign handling:
    - Capacitive loops appear ABOVE the x-axis (positive -Im(Z))
    - Inductive loops appear BELOW the x-axis (negative -Im(Z))
    Min and max frequency points are plotted as labeled points (not in legend).
    """
    fig, ax = plt.subplots(figsize=(9, 10))
    
    # Make axis tick labels black
    ax.tick_params(axis='both', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    
    # Используем текущие данные (уже инвертированы в __post_init__)
    re_z_plot = data.re_z
    im_z_plot = data.im_z  # Уже инвертированы, емкостные процессы положительны
    freq_plot = data.freq
    
    # Plot full spectrum
    ax.plot(re_z_plot, im_z_plot, 'o-', markersize=5, linewidth=1.8,
            label='Experimental', color='#1f77b4', markeredgecolor='white', markeredgewidth=0.8)
    
    # Find extreme points
    min_freq_idx = np.argmin(freq_plot)
    max_freq_idx = np.argmax(freq_plot)
    
    # Highlight extreme points with labels (not in legend)
    ax.plot(re_z_plot[min_freq_idx], im_z_plot[min_freq_idx], 'ro', 
            markersize=10, markeredgecolor='darkred', markerfacecolor='red', alpha=0.8)
    ax.plot(re_z_plot[max_freq_idx], im_z_plot[max_freq_idx], 'go', 
            markersize=10, markeredgecolor='darkgreen', markerfacecolor='green', alpha=0.8)
    
    # Add labels for min and max frequency points with superscript formatting
    f_min = freq_plot[min_freq_idx]
    f_max = freq_plot[max_freq_idx]
    
    # Calculate offset for labels based on data range
    x_range = np.max(re_z_plot) - np.min(re_z_plot)
    y_range = np.max(im_z_plot) - np.min(im_z_plot)
    offset_x = x_range * 0.03
    offset_y = y_range * 0.03
    
    # Format frequency labels with superscript
    f_min_label = format_with_superscript(f_min)
    f_max_label = format_with_superscript(f_max)
    
    # Adjust label position based on sign of imaginary part
    if im_z_plot[min_freq_idx] >= 0:
        text_y_offset_min = offset_y
    else:
        text_y_offset_min = -offset_y * 1.5
    
    if im_z_plot[max_freq_idx] >= 0:
        text_y_offset_max = offset_y
    else:
        text_y_offset_max = -offset_y * 1.5
    
    ax.annotate(f'{f_min_label} Hz', 
               xy=(re_z_plot[min_freq_idx], im_z_plot[min_freq_idx]),
               xytext=(offset_x, text_y_offset_min),
               textcoords='offset points',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.2, edgecolor='red'))
    
    ax.annotate(f'{f_max_label} Hz', 
               xy=(re_z_plot[max_freq_idx], im_z_plot[max_freq_idx]),
               xytext=(offset_x, text_y_offset_max),
               textcoords='offset points',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.2, edgecolor='green'))
    
    # Find and label decade frequency points
    log_freqs = np.log10(freq_plot)
    freq_min_log = np.floor(np.min(log_freqs))
    freq_max_log = np.ceil(np.max(log_freqs))
    
    decade_freqs = []
    for exponent in range(int(freq_min_log), int(freq_max_log) + 1):
        decade_freq = 10.0 ** exponent
        # Find closest point in frequency array
        idx = np.argmin(np.abs(freq_plot - decade_freq))
        if idx not in [min_freq_idx, max_freq_idx]:
            decade_freqs.append((idx, decade_freq))
    
    # Plot decade points with labels using superscript formatting
    for idx, dec_freq in decade_freqs:
        ax.plot(re_z_plot[idx], im_z_plot[idx], 'mo', markersize=8,
                markeredgecolor='purple', markerfacecolor='magenta', alpha=0.7)
        
        # Format frequency label with superscript
        label = format_with_superscript(dec_freq) + ' Hz'
        
        # Offset label position
        x_range = np.max(re_z_plot) - np.min(re_z_plot)
        y_range = np.max(im_z_plot) - np.min(im_z_plot)
        offset_x = x_range * 0.02
        offset_y = y_range * 0.02
        
        # Adjust label position based on sign of imaginary part
        if im_z_plot[idx] >= 0:
            # Above the curve - label above point
            text_y_offset = offset_y
        else:
            # Below the curve - label below point
            text_y_offset = -offset_y * 1.5
        
        ax.annotate(label, 
                   xy=(re_z_plot[idx], im_z_plot[idx]),
                   xytext=(offset_x, text_y_offset),
                   textcoords='offset points',
                   fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Add shading for inductive region (below x-axis) - only if there are negative values
    if np.any(im_z_plot < 0):
        # Find the actual minimum of the data (not the axis limit)
        data_y_min = np.min(im_z_plot)
        ax.fill_between([np.min(re_z_plot), np.max(re_z_plot)], data_y_min, 0,
                        alpha=0.1, color='red', label='Inductive region (L)')
    
    ax.set_xlabel("Re(Z) / Ohm", fontweight='bold', fontsize=12)
    ax.set_ylabel("-Im(Z) / Ohm", fontweight='bold', fontsize=12)
    # No title - removed as requested
    ax.legend(loc='best', fontsize=9, frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    # Set y-axis limits to show inductive tail as a small portion (1/5 of positive range)
    if np.any(im_z_plot < 0):
        # Find positive values (capacitive part)
        positive_mask = im_z_plot >= 0
        negative_mask = im_z_plot < 0
        
        if np.any(positive_mask) and np.any(negative_mask):
            # Get the maximum positive value
            positive_max = np.max(im_z_plot[positive_mask])
            
            # Get the minimum negative value (most negative)
            negative_min = np.min(im_z_plot[negative_mask])
            
            # Calculate the range of positive values
            positive_range = positive_max
            
            # Set y-axis limits:
            # Upper limit: 10% above the maximum positive value
            y_max_data = positive_max * 1.1
            
            # Lower limit: extend only 1/5 of positive range below zero
            # This makes the inductive tail visible but not dominant
            y_min_data = -positive_range / 5
            
            # But if the actual negative data goes beyond this, we need to show at least some of it
            if negative_min < y_min_data:
                # Still limit to 1/3 of positive range maximum
                y_min_data = max(negative_min * 0.8, -positive_range / 3)
            
            ax.set_ylim(y_min_data, y_max_data)
            
            # Add annotation if inductive tail is truncated
            if negative_min < y_min_data:
                ax.annotate(f'Inductive tail extends to {negative_min:.2e} Ω\n(truncated for clarity)',
                           xy=(np.min(re_z_plot), y_min_data * 0.9),
                           xytext=(np.min(re_z_plot) + (np.max(re_z_plot)-np.min(re_z_plot))*0.05, 
                                  y_min_data * 0.7),
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))
        else:
            # Only negative values (unlikely for typical EIS)
            y_min_data = np.min(im_z_plot) * 1.1
            y_max_data = np.max(im_z_plot) * 1.1
            ax.set_ylim(y_min_data, y_max_data)
    else:
        # No inductive behavior, standard scaling
        y_min_data = np.min(im_z_plot) * 0.9 if np.min(im_z_plot) > 0 else np.min(im_z_plot) * 1.1
        y_max_data = np.max(im_z_plot) * 1.1
        ax.set_ylim(y_min_data, y_max_data)
    
    plt.tight_layout()
    return fig
    
def plot_deconvolution_vs_frequency(deconv_result: DeconvolutionResult, drt_result: DRTResult = None,
                                      title: str = "Gaussian Deconvolution vs Frequency") -> plt.Figure:
    """
    Plot Gaussian deconvolution result with Frequency on x-axis (high to low).
    Legend is placed outside the plot area.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    
    # Make axis tick labels black
    ax.tick_params(axis='both', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    
    # Convert relaxation times to frequencies
    frequencies = 1.0 / (2 * np.pi * deconv_result.x_linear)
    
    # Sort by frequency descending (high to low)
    sort_idx = np.argsort(frequencies)[::-1]
    freqs_sorted = frequencies[sort_idx]
    y_original_sorted = deconv_result.y_original[sort_idx]
    
    # Plot original DRT data vs frequency
    ax.semilogx(freqs_sorted, y_original_sorted, 'o-', markersize=3, linewidth=1,
                color='black', alpha=0.5, label='DRT Data', zorder=1)
    
    # Create dense frequency grid for smooth curves
    freqs_dense = np.logspace(np.log10(np.min(frequencies)), np.log10(np.max(frequencies)), 2000)
    freqs_dense_sorted = np.sort(freqs_dense)[::-1]  # High to low
    
    # Calculate log frequency for Gaussian evaluation
    log_tau_dense = -np.log10(2 * np.pi * freqs_dense)  # log10(τ) = -log10(2πf)
    
    # Define consistent color mapping based on peak ID
    style = get_plot_style()
    cmap = plt.get_cmap(style["cmap"])
    colors = cmap(np.linspace(0, 1, max(len(deconv_result.peaks), 1)))
    color_map = {peak.id: colors[i] for i, peak in enumerate(deconv_result.peaks)}
    
    # Plot individual Gaussian components
    if deconv_result.peaks:
        for peak in deconv_result.peaks:
            color = color_map[peak.id]
            # Calculate component vs frequency
            y_component = peak.amplitude * GaussianModelDeconv.gaussian(
                log_tau_dense, 1.0, peak.center_log, peak.sigma_log
            )
            # Sort for high-to-low frequency
            y_component_sorted = y_component[::-1]
            
            ax.fill_between(freqs_dense_sorted, 0, y_component_sorted,
                           color=color, alpha=0.3, linewidth=0)
            ax.plot(freqs_dense_sorted, y_component_sorted, '-', color=color, linewidth=1.5,
                   label=f'Peak {peak.id} ({peak.fraction_percent:.1f}%)',
                   zorder=2)
    
    # Plot total fit
    if deconv_result.fit_y_original is not None:
        fit_sorted = deconv_result.fit_y_original[sort_idx]
        ax.semilogx(freqs_sorted, fit_sorted, 'r--', linewidth=1.5,
                   label='Total Fit', zorder=3)
    
    # Invert x-axis (high frequency to left)
    ax.invert_xaxis()
    
    ax.set_xlabel("Frequency / Hz", fontweight='bold')
    ax.set_ylabel("γ(τ) / Ohm", fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, edgecolor='black')
    ax.grid(style["show_grid"], alpha=0.3, linestyle='--', which='both')
    
    # Add quality metrics
    if deconv_result.quality_metrics:
        metrics_text = f"R² = {deconv_result.quality_metrics.get('R²', 0):.4f}\n"
        metrics_text += f"RMSE = {deconv_result.quality_metrics.get('RMSE', 0):.2e}"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    return fig

def plot_peak_area_distribution_with_values(deconv_result: DeconvolutionResult,
                                              drt_result: DRTResult = None,
                                              title: str = "Peak Area Distribution") -> plt.Figure:
    """
    Plot bar chart with peak areas (absolute values) and percentages in parentheses.
    Legend shows total DRT resistance (Rpol). Legend placed outside plot area.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Make axis tick labels black
    ax.tick_params(axis='both', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    
    if not deconv_result.peaks:
        ax.text(0.5, 0.5, "No peaks to display", ha='center', va='center', fontsize=10)
        return fig
    
    # Sort peaks by ID (which are already sorted by frequency high to low)
    peaks_sorted = sorted(deconv_result.peaks, key=lambda p: p.id)
    peaks_ids = [f'Process {p.id}' for p in peaks_sorted]
    areas = [p.area for p in peaks_sorted]
    fractions_percent = [p.fraction_percent for p in peaks_sorted]
    
    style = get_plot_style()
    cmap = plt.get_cmap(style["cmap"])
    colors = cmap(np.linspace(0, 1, max(len(deconv_result.peaks), 1)))
    color_map = {peak.id: colors[i] for i, peak in enumerate(deconv_result.peaks)}
    colors_sorted = [color_map[p.id] for p in peaks_sorted]
    
    bars = ax.bar(peaks_ids, areas, color=colors_sorted, edgecolor='black', alpha=0.7)
    
    # Add labels with absolute value and percentage
    for bar, area, frac in zip(bars, areas, fractions_percent):
        height = bar.get_height()
        label = f'{area:.3e} Ω\n({frac:.1f}%)'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Get total resistance
    if deconv_result is not None and deconv_result.peaks:
        total_resistance = sum([p.area for p in deconv_result.peaks])
        legend_text = f'Total R_pol = {total_resistance:.4f} Ω'
    elif drt_result is not None:
        total_resistance = drt_result.R_pol
        legend_text = f'Total R_pol = {total_resistance:.4f} Ω'
    else:
        total_resistance = deconv_result.total_area
        legend_text = f'Total Area = {total_resistance:.4f} Ω·s'
    
    # Add legend with total resistance outside plot
    ax.axhline(y=total_resistance, color='red', linestyle='--', linewidth=1, label=legend_text)
    
    ax.set_xlabel('Relaxation Process', fontweight='bold', fontsize=10)
    ax.set_ylabel('Resistance Contribution / Ω', fontweight='bold', fontsize=10)
    # No title - removed as requested
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8, frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis with scientific notation
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
    
    plt.tight_layout()
    return fig

def plot_sequential_rc_model(deconv_result: DeconvolutionResult,
                              drt_result: DRTResult,
                              data: ImpedanceData,
                              title: str = "Experimental vs Sequential RC Model",
                              refined_rq: Optional[Dict[str, Any]] = None) -> plt.Figure:
    """
    Plot experimental impedance spectrum with sequential RQ (ZARC) model
    and residual curve: Δ(-Im(Z)) vs Re(Z) with R² and RMSE.
    """
    fig = plt.figure(figsize=(8.5, 6.2))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[3, 1.2],
        width_ratios=[1, 0.025],
        hspace=0.08,
        wspace=0.03
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax)
    cax = fig.add_subplot(gs[1, 1])

    for axis in (ax, ax_res):
        axis.tick_params(axis='both', colors='black')
        axis.xaxis.label.set_color('black')
        axis.yaxis.label.set_color('black')

    sort_idx_exp = np.argsort(data.freq)
    re_exp_sorted = data.re_z[sort_idx_exp]
    im_exp_sorted = data.im_z[sort_idx_exp]
    freq_exp_sorted = data.freq[sort_idx_exp]
    omega_exp = 2.0 * np.pi * freq_exp_sorted

    ax.plot(re_exp_sorted, im_exp_sorted, 'o-', markersize=3, linewidth=1,
            color='#1f77b4', alpha=0.7, label='Experimental', zorder=1)

    f_min_exp = max(np.min(freq_exp_sorted), 1e-12)
    f_max_exp = np.max(freq_exp_sorted)
    freq_model_dense = np.logspace(np.log10(f_min_exp), np.log10(f_max_exp), 500)
    omega_model = 2.0 * np.pi * freq_model_dense

    style = get_plot_style()
    cmap = plt.get_cmap(style["cmap"])
    rc_curves = []

    if refined_rq is not None:
        # --- Mode A: CNLS-refined exact ZARC model ---
        R_inf_start = refined_rq['R_inf']
        L_val = refined_rq['L']
        procs = refined_rq['processes']
        colors = cmap(np.linspace(0, 1, max(len(procs), 1)))
        color_map = {p['id']: colors[i] for i, p in enumerate(procs)}

        Z_total = np.full_like(omega_model, R_inf_start, dtype=complex)
        Z_total_exp = np.full_like(omega_exp, R_inf_start, dtype=complex)
        if L_val > 0:
            Z_total += 1j * omega_model * L_val
            Z_total_exp += 1j * omega_exp * L_val

        cumulative_R = R_inf_start
        for p in procs:
            R_i, Q_i, alpha_i, f_char = p['R_i'], p['Q_i'], p['alpha_i'], p['f_i']
            Z_i_dense = R_i / (1.0 + ((1j * omega_model) ** alpha_i) * R_i * Q_i)
            Z_i_exp = R_i / (1.0 + ((1j * omega_exp) ** alpha_i) * R_i * Q_i)
            Z_total += Z_i_dense
            Z_total_exp += Z_i_exp

            f_arc = np.logspace(np.log10(max(f_char, 1e-12)) - 5.0, np.log10(max(f_char, 1e-12)) + 5.0, 300)
            w_arc = 2.0 * np.pi * f_arc
            Z_i_arc = R_i / (1.0 + ((1j * w_arc) ** alpha_i) * R_i * Q_i)

            cumulative_R_prev = cumulative_R
            cumulative_R += R_i
            rc_curves.append({
                'id': p['id'],
                'R': R_i,
                'Z': cumulative_R_prev + Z_i_arc,
                'start_R': cumulative_R_prev,
                'end_R': cumulative_R
            })
        total_R = cumulative_R
        model_label = 'Total Model (CNLS Refined)'
    else:
        # --- Mode B: Direct DRT Gaussian quadrature ---
        R_inf_start = drt_result.R_inf
        for i in range(len(im_exp_sorted) - 1):
            if im_exp_sorted[i] >= 0 and im_exp_sorted[i + 1] < 0:
                t = -im_exp_sorted[i] / (im_exp_sorted[i + 1] - im_exp_sorted[i])
                R_inf_start = float(re_exp_sorted[i] + t * (re_exp_sorted[i + 1] - re_exp_sorted[i]))
                break
        L_val = drt_result.L
        peaks_sorted = sorted(deconv_result.peaks, key=lambda p: p.id)
        colors = cmap(np.linspace(0, 1, max(len(peaks_sorted), 1)))
        color_map = {peak.id: colors[i] for i, peak in enumerate(peaks_sorted)}

        def calc_gaussian_process_impedance(omega_arr: np.ndarray, peak: GaussianPeak) -> np.ndarray:
            sigma = max(float(peak.sigma_log), 1e-6)
            center_log = float(peak.center_log)
            if sigma < 1e-4:
                return peak.area / (1.0 + 1j * omega_arr * peak.center)
            u_grid = np.linspace(center_log - 5.0 * sigma, center_log + 5.0 * sigma, 401)
            tau_sub = 10.0 ** u_grid
            gauss_profile = np.exp(-0.5 * ((u_grid - center_log) / sigma) ** 2)
            profile_area = np.trapezoid(gauss_profile, u_grid)
            g_weights = (peak.area / profile_area) * gauss_profile if profile_area > 0 else np.zeros_like(u_grid)
            denom = 1.0 + 1j * np.outer(omega_arr, tau_sub)
            return np.trapezoid(g_weights[np.newaxis, :] / denom, u_grid, axis=1)

        Z_total = np.full_like(omega_model, R_inf_start, dtype=complex)
        Z_total_exp = np.full_like(omega_exp, R_inf_start, dtype=complex)
        if L_val > 0:
            Z_total += 1j * omega_model * L_val
            Z_total_exp += 1j * omega_exp * L_val

        cumulative_R = R_inf_start
        for peak in peaks_sorted:
            R_i = peak.area
            f_char = peak.get_characteristic_frequency()
            Z_total += calc_gaussian_process_impedance(omega_model, peak)
            Z_total_exp += calc_gaussian_process_impedance(omega_exp, peak)

            span_decades = max(4.0, 5.0 * peak.sigma_log + 2.5)
            f_arc = np.logspace(np.log10(f_char) - span_decades, np.log10(f_char) + span_decades, 300)
            Z_i_arc = calc_gaussian_process_impedance(2.0 * np.pi * f_arc, peak)

            cumulative_R_prev = cumulative_R
            cumulative_R += R_i
            rc_curves.append({
                'id': peak.id,
                'R': R_i,
                'Z': cumulative_R_prev + Z_i_arc,
                'start_R': cumulative_R_prev,
                'end_R': cumulative_R
            })
        total_R = cumulative_R
        model_label = 'Total Model (DRT)'

    re_model = np.real(Z_total)
    im_model = -np.imag(Z_total)

    ax.plot(re_model, im_model, 'r--', linewidth=1.6, label=model_label, zorder=3)

    for curve in rc_curves:
        color = color_map[curve['id']]
        ax.plot(np.real(curve['Z']), -np.imag(curve['Z']), '-', linewidth=1.3, color=color,
                label=f'Process {curve["id"]}: R={curve["R"]:.3e} Ω', zorder=2)
        ax.plot(curve['start_R'], 0, 's', color=color, markersize=4,
                markeredgecolor='black', markeredgewidth=0.5, zorder=4)
        ax.plot(curve['end_R'], 0, 'o', color=color, markersize=4,
                markeredgecolor='black', markeredgewidth=0.5, zorder=4)

    ax.plot(R_inf_start, 0, '^', color='red', markersize=6,
            markeredgecolor='black', markeredgewidth=0.8,
            label=f'R∞ = {R_inf_start:.4f} Ω', zorder=5)
    ax.plot(total_R, 0, 'd', color='darkred', markersize=6,
            markeredgecolor='black', markeredgewidth=0.8,
            label=f'Total R = {total_R:.4f} Ω', zorder=5)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    all_re = np.concatenate([re_exp_sorted, [R_inf_start, total_R]])
    x_span = np.max(all_re) - np.min(all_re)
    x_pad = x_span * 0.05 if x_span > 0 else 0.1
    ax.set_xlim(np.min(all_re) - x_pad, np.max(all_re) + x_pad)

    positive_exp = im_exp_sorted[im_exp_sorted >= 0]
    positive_mod = im_model[im_model >= 0]
    pos_max = max(
        np.max(positive_exp) if len(positive_exp) > 0 else 0.0,
        np.max(positive_mod) if len(positive_mod) > 0 else 0.0
    )

    if np.any(im_exp_sorted < 0) and pos_max > 0:
        negative_min = np.min(im_exp_sorted[im_exp_sorted < 0])
        y_max_data = pos_max * 1.15
        y_min_data = -pos_max / 5.0
        if negative_min < y_min_data:
            y_min_data = max(negative_min * 0.8, -pos_max / 3.0)
        ax.set_ylim(y_min_data, y_max_data)
        ax.fill_between([np.min(all_re) - x_pad, np.max(all_re) + x_pad], y_min_data, 0,
                        alpha=0.1, color='red', label='Inductive region', zorder=0)
    elif pos_max > 0:
        ax.set_ylim(-pos_max * 0.05, pos_max * 1.15)

    ax.set_ylabel("-Im(Z) / Ohm", fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, edgecolor='black', ncol=1)
    ax.grid(style["show_grid"], alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelbottom=False)

    # --- Residual subplot: Δ(-Im(Z)) vs Re(Z) ---
    im_model_exp = -np.imag(Z_total_exp)
    re_model_exp = np.real(Z_total_exp)
    res_im = im_exp_sorted - im_model_exp

    # Complex R² and RMSE (over both Re and -Im)
    y_true_all = np.concatenate([re_exp_sorted, im_exp_sorted])
    y_pred_all = np.concatenate([re_model_exp, im_model_exp])
    ss_res = np.sum((y_true_all - y_pred_all) ** 2)
    ss_tot = np.sum((y_true_all - np.mean(y_true_all)) ** 2)
    r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse_val = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))
    max_abs_res = max(float(np.max(np.abs(res_im))), 1e-9)

    ax_res.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=1)
    ax_res.plot(re_exp_sorted, res_im, '-', color='gray', alpha=0.5, linewidth=1.0, zorder=2)
    sc = ax_res.scatter(
        re_exp_sorted, res_im,
        c=res_im, cmap='coolwarm',
        vmin=-max_abs_res, vmax=max_abs_res,
        s=24, edgecolors='black', linewidths=0.3, zorder=3
    )

    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label('Δ(-Im Z) / Ω', fontsize=8, fontweight='bold')
    cbar.ax.tick_params(labelsize=7, colors='black')

    ax_res.set_xlabel("Re(Z) / Ohm", fontweight='bold')
    ax_res.set_ylabel("Residuals\nΔ(-Im Z) / Ω", fontweight='bold', fontsize=9)
    ax_res.set_ylim(-max_abs_res * 1.3, max_abs_res * 1.3)
    ax_res.grid(style["show_grid"], alpha=0.3, linestyle='--')

    metrics_text = f"R² = {r2_val:.5f}  |  RMSE = {rmse_val:.2e} Ω"
    ax_res.text(
        0.015, 0.90, metrics_text, transform=ax_res.transAxes,
        fontsize=8, fontweight='bold', verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0.85, edgecolor='gray')
    )

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
                    df = parse_uploaded_dataframe(uploaded_file, nrows=5)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.subheader("Column Mapping")
                    n_cols = len(df.columns)
                    col_freq = st.selectbox("Frequency column", df.columns, index=0)
                    col_re = st.selectbox("Re(Z) column", df.columns, index=min(1, n_cols - 1))
                    col_im = st.selectbox("-Im(Z) column", df.columns, index=min(2, n_cols - 1))
                    
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
                f_min_actual = data.freq[min(f_min_idx, data.n_points - 1)]
                f_max_actual = data.freq[min(f_max_idx, data.n_points - 1)]
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
                    safe_idx = min(point_idx, data.n_points - 1)
                    st.caption(f"f = {data.freq[safe_idx]:.2e} Hz")
                    st.caption(f"Z = {data.Z_mod[safe_idx]:.2e} Ω")
            
            with col_c:
                st.write("")
                st.write("")
                if st.button("🗑️ Remove Point", use_container_width=True, key="remove_point_btn"):
                    if point_idx < data.n_points:
                        data.remove_point(point_idx)
                        # Adjust indices after removal
                        new_max = data.n_points - 1
                        # Update selected point index
                        st.session_state.selected_point_idx = min(point_idx, new_max)
                        # Update frequency range indices
                        current_min_idx, current_max_idx = st.session_state.freq_range_idx
                        new_min_idx = min(current_min_idx, new_max)
                        new_max_idx = min(current_max_idx, new_max)
                        # Ensure min <= max
                        if new_min_idx > new_max_idx:
                            new_min_idx = new_max_idx
                        st.session_state.freq_range_idx = (new_min_idx, new_max_idx)
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
        def cached_plots(freq_tuple, re_tuple, im_tuple, point_idx, freq_range_idx, n_points, original_len, style_items):
            """Cache plots to prevent rerender on slider movement"""
            apply_publication_plot_style(dict(style_items))
            temp_data = ImpedanceData(np.array(freq_tuple), np.array(re_tuple), np.array(im_tuple), auto_convention=False)
            fig_nyquist = plot_nyquist_matplotlib(temp_data, highlight_idx=point_idx, freq_range_idx=freq_range_idx)
            fig_bode = plot_bode_matplotlib(temp_data, highlight_idx=point_idx, freq_range_idx=freq_range_idx)
            return fig_nyquist, fig_bode
        
        # Convert data to tuples for caching
        freq_tuple = tuple(data.freq)
        re_tuple = tuple(data.re_z)
        im_tuple = tuple(data.im_z)
        style_items = tuple(sorted(get_plot_style().items()))
        
        # Get cached plots
        fig_nyquist, fig_bode = cached_plots(
            freq_tuple, re_tuple, im_tuple, 
            st.session_state.selected_point_idx,
            st.session_state.freq_range_idx,
            data.n_points, len(data.original_freq),
            style_items
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
    
    # Auto-detect inductive behavior
    has_inductive = data.detect_inductive_behavior()
    if has_inductive:
        st.info("🔍 **Inductive behavior detected** in the high frequency region. Consider using 'Fitting with Inductance' mode for better results.")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("⚙️ Analysis Parameters")
        
        analysis_method = st.selectbox(
            "DRT Calculation Method",
            ["Tikhonov Regularization (NNLS)",
             "Maximum Entropy (auto-λ)"]
        )
        
        # Auto-select inductance handling based on detection
        if has_inductive:
            default_induct_index = 1  # "Fitting with Inductance"
        else:
            default_induct_index = 0  # "Fitting w/o Inductance"
        
        induct_mode = st.selectbox(
            "Inductance handling",
            ["Fitting w/o Inductance", "Fitting with Inductance", "Discard Inductive Data"],
            index=default_induct_index,
            help="""
            - Fitting w/o Inductance: Standard DRT model without inductance
            - Fitting with Inductance: Includes L as fitting parameter: Z = R∞ + i·2πf·L + DRT
            - Discard Inductive Data: Remove points with positive -Im(Z)
            """
        )
        
        n_tau = st.slider("Number of time points", 50, 300, 150)
        
        # Method-specific parameters
        lambda_auto = True
        lambda_value = None
        reg_order = 2
        
        if analysis_method == "Tikhonov Regularization (NNLS)":
            reg_order = st.selectbox("Regularization order", [0, 1, 2], index=2)
            lambda_auto = st.checkbox("Automatic λ selection (L-curve)", value=True)
            if not lambda_auto:
                lambda_value = st.number_input("λ value", value=1e-4, format="%.1e")
        
        elif analysis_method == "Maximum Entropy (auto-λ)":
            entropy_lambda_auto = st.checkbox("Auto-select λ", value=True)
            if not entropy_lambda_auto:
                st.session_state.app_state.drt_parameters['entropy_lambda'] = st.number_input("Entropy λ", value=0.1, format="%.2f")
            st.session_state.app_state.drt_parameters['lambda_auto'] = entropy_lambda_auto
        
        # Store parameters
        st.session_state.app_state.drt_method = analysis_method
        st.session_state.app_state.drt_parameters.update({
            'n_tau': n_tau,
            'induct_mode': induct_mode,
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
                        # Handle data discarding if needed
                        if induct_mode == "Discard Inductive Data":
                            # Индуктивные точки имеют отрицательный data.im_z (так как data.im_z хранит -Im(Z))
                            mask = data.im_z < 0
                            n_discarded = np.sum(mask)
                            if n_discarded > 0:
                                st.info(f"Discarding {n_discarded} inductive data points")
                                # Create temporary data object with discarded points
                                temp_data = ImpedanceData(
                                    data.freq[~mask],
                                    data.re_z[~mask],
                                    data.im_z[~mask],
                                    auto_convention=False
                                )
                            else:
                                temp_data = data
                        else:
                            temp_data = data
                        
                        # Check if we have enough points after discarding
                        if len(temp_data.freq) < 3:
                            st.error("Not enough data points after discarding inductive points. Please use a different mode.")
                            return
                        
                        # Create solver with appropriate settings
                        include_inductive = (induct_mode == "Fitting with Inductance")
                        
                        if analysis_method == "Tikhonov Regularization (NNLS)":
                            drt_solver = TikhonovDRT(temp_data, regularization_order=reg_order, 
                                                     include_inductive=include_inductive)
                            
                            if include_inductive:
                                result = drt_solver.compute_with_inductance(
                                    n_tau=n_tau, 
                                    lambda_value=lambda_value, 
                                    lambda_auto=lambda_auto
                                )
                            else:
                                result = drt_solver.compute(
                                    n_tau=n_tau, 
                                    lambda_value=lambda_value, 
                                    lambda_auto=lambda_auto
                                )
                        
                        elif analysis_method == "Maximum Entropy (auto-λ)":
                            drt_solver = MaxEntropyDRT(temp_data, include_inductive=include_inductive)
                            lambda_auto_val = st.session_state.app_state.drt_parameters.get('lambda_auto', True)
                            lambda_val = st.session_state.app_state.drt_parameters.get('entropy_lambda', None)
                            
                            if include_inductive:
                                result = drt_solver.compute_with_inductance(
                                    n_tau=n_tau, 
                                    lambda_value=lambda_val, 
                                    lambda_auto=lambda_auto_val
                                )
                            else:
                                result = drt_solver.compute(
                                    n_tau=n_tau, 
                                    lambda_value=lambda_val, 
                                    lambda_auto=lambda_auto_val
                                )
                        
                        # Store results
                        st.session_state.app_state.drt_result = result
                        st.session_state.app_state.drt_solver = drt_solver
                        st.session_state.app_state.drt_calculated = True
                        
                        # IMPORTANT: Clear deconvolution state when DRT is recalculated
                        # This ensures Step 3 uses the updated DRT spectrum
                        st.session_state.app_state.deconvolver = None
                        st.session_state.app_state.deconv_result = None
                        st.session_state.app_state.deconv_calculated = False
                        st.session_state.app_state.peak_info = None
                        st.session_state.app_state.initial_peak_params = None
                        st.session_state.app_state.manual_peaks = []
                        st.session_state.app_state.residuals_peaks = []
                        st.session_state.app_state.deconv_tau_range_idx = None
                        st.session_state.app_state.refined_rq_params = None
                        
                        # Display summary
                        st.success("✅ DRT calculation complete!")
                        if include_inductive and result.L > 0:
                            st.info(f"📊 Fitted inductance L = {result.L:.3e} H")
                        if result.lambda_opt is not None:
                            st.info(f"📊 Optimal regularization parameter λ = {result.lambda_opt:.3e}")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"DRT calculation failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    with col2:
        if st.session_state.app_state.drt_calculated and st.session_state.app_state.drt_result:
            st.subheader("📊 DRT Results")
            
            result = st.session_state.app_state.drt_result
            solver = st.session_state.app_state.drt_solver
            
            st.info(f"""
            **Method:** {result.method}
            **R∞:** {result.R_inf:.4f} Ω
            **Rpol:** {result.R_pol:.4f} Ω
            **τ range:** {result.tau_grid.min():.2e} - {result.tau_grid.max():.2e} s
            """)
            
            if result.L > 0:
                st.info(f"**Inductance L:** {result.L:.3e} H")
            
            if result.lambda_opt is not None:
                st.info(f"**λ:** {result.lambda_opt:.3e}")
            
            # Display integral verification
            integral, ratio = result.verify_integral()
            if abs(ratio - 1.0) < 0.05:
                st.success(f"✅ DRT integral verification: {integral:.4f} Ω (ratio = {ratio:.4f})")
            else:
                st.warning(f"⚠️ DRT integral verification: {integral:.4f} Ω (ratio = {ratio:.4f})")
            
            # Reconstruct impedance from DRT for validation
            if solver is not None:
                Z_rec_real, Z_rec_imag = solver.reconstruct_impedance(result.tau_grid, result.gamma, result.L)
                
                # Calculate reconstruction error
                Z_original = data.Z
                Z_reconstructed = Z_rec_real + 1j * Z_rec_imag
                error_percent = np.abs((Z_original - Z_reconstructed) / (Z_original + 1e-10)) * 100
                mean_error = np.mean(error_percent)
                max_error = np.max(error_percent)
                
                st.info(f"""
                **Reconstruction Quality:**
                **Mean Error:** {mean_error:.2f}%
                **Max Error:** {max_error:.2f}%
                """)
                
                # Display validation plots
                st.markdown("---")
                st.subheader("🔍 Validation: Original vs Reconstructed Impedance")
                
                # Nyquist plot
                fig_nyq, ax_nyq = plt.subplots(figsize=(8, 6))
                # Make axis tick labels black
                ax_nyq.tick_params(axis='both', colors='black')
                ax_nyq.xaxis.label.set_color('black')
                ax_nyq.yaxis.label.set_color('black')
                ax_nyq.plot(data.re_z, data.im_z, 'o', markersize=6, 
                           label='Experimental', color='#1f77b4', alpha=0.7)
                ax_nyq.plot(Z_rec_real, Z_rec_imag, '-', linewidth=2.5, 
                           label='Reconstructed from DRT', color='#ff7f0e')
                ax_nyq.set_xlabel("Re(Z) / Ohm", fontweight='bold')
                ax_nyq.set_ylabel("-Im(Z) / Ohm", fontweight='bold')
                # No title - removed as requested
                ax_nyq.legend(loc='best', frameon=True)
                ax_nyq.grid(True, alpha=0.3, linestyle='--')
                ax_nyq.set_aspect('equal', adjustable='box')
                st.pyplot(fig_nyq)
                plt.close()
                
                # Bode plots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
                # Make axis tick labels black
                ax1.tick_params(axis='both', colors='black')
                ax2.tick_params(axis='both', colors='black')
                ax1.xaxis.label.set_color('black')
                ax1.yaxis.label.set_color('black')
                ax2.xaxis.label.set_color('black')
                ax2.yaxis.label.set_color('black')
                
                # Magnitude plot
                mag_exp = data.Z_mod
                mag_rec = np.sqrt(Z_rec_real**2 + Z_rec_imag**2)
                ax1.loglog(data.freq, mag_exp, 'o', markersize=6, 
                          label='Experimental', color='#1f77b4', alpha=0.7)
                ax1.loglog(data.freq, mag_rec, '-', linewidth=2.5, 
                          label='Reconstructed', color='#ff7f0e')
                ax1.set_xlabel("Frequency / Hz", fontweight='bold')
                ax1.set_ylabel("|Z| / Ohm", fontweight='bold')
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3, linestyle='--')
                
                # Phase plot
                phase_exp = data.phase
                phase_rec = np.arctan2(Z_rec_imag, Z_rec_real) * 180 / np.pi
                ax2.semilogx(data.freq, phase_exp, 'o', markersize=6, 
                            label='Experimental', color='#1f77b4', alpha=0.7)
                ax2.semilogx(data.freq, phase_rec, '-', linewidth=2.5, 
                            label='Reconstructed', color='#ff7f0e')
                ax2.set_xlabel("Frequency / Hz", fontweight='bold')
                ax2.set_ylabel("Phase / deg", fontweight='bold')
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
                
                # No overall figure title - removed as requested
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Error plot
                fig_err, ax_err = plt.subplots(figsize=(8, 5))
                ax_err.tick_params(axis='both', colors='black')
                ax_err.xaxis.label.set_color('black')
                ax_err.yaxis.label.set_color('black')
                ax_err.semilogx(data.freq, error_percent, 'o-', markersize=5, 
                               linewidth=1.5, color='#d62728')
                ax_err.set_xlabel("Frequency / Hz", fontweight='bold')
                ax_err.set_ylabel("Relative Error / %", fontweight='bold')
                # No title - removed as requested
                ax_err.grid(True, alpha=0.3, linestyle='--')
                ax_err.axhline(y=mean_error, color='gray', linestyle='--', 
                              label=f'Mean Error: {mean_error:.2f}%')
                ax_err.legend()
                st.pyplot(fig_err)
                plt.close()
                
                # If inductance was fitted, show its contribution
                if result.L > 0:
                    st.markdown("---")
                    st.subheader("🔌 Inductance Contribution")
                    
                    fig_L, ax_L = plt.subplots(figsize=(8, 5))
                    ax_L.tick_params(axis='both', colors='black')
                    ax_L.xaxis.label.set_color('black')
                    ax_L.yaxis.label.set_color('black')
                    omega = 2 * np.pi * data.freq
                    L_contribution = omega * result.L
                    ax_L.loglog(data.freq, L_contribution, '-', linewidth=2, color='#9467bd', label='L contribution to -Im(Z)')
                    ax_L.set_xlabel("Frequency / Hz", fontweight='bold')
                    ax_L.set_ylabel("-ωL / Ohm", fontweight='bold')
                    # No title - removed as requested
                    ax_L.grid(True, alpha=0.3, linestyle='--')
                    ax_L.legend()
                    st.pyplot(fig_L)
                    plt.close()
            
            # DRT plot
            st.markdown("---")
            st.subheader("📈 DRT Spectrum")
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
    
    # Полные исходные массивы DRT (включая возможные хвосты)
    full_tau = drt_result.tau_grid
    full_gamma = drt_result.gamma
    n_total_tau = len(full_tau)
    
    # Инициализация диапазона фитинга по индексам точек сетки tau
    if (st.session_state.app_state.deconv_tau_range_idx is None or 
        st.session_state.app_state.deconv_tau_range_idx[0] < 0 or
        st.session_state.app_state.deconv_tau_range_idx[1] >= n_total_tau):
        st.session_state.app_state.deconv_tau_range_idx = (0, n_total_tau - 1)
    
    curr_min_idx, curr_max_idx = st.session_state.app_state.deconv_tau_range_idx
    
    # Срез данных строго внутри выбранной области фитинга (без хвостов)
    cropped_tau = full_tau[curr_min_idx:curr_max_idx + 1]
    cropped_gamma = full_gamma[curr_min_idx:curr_max_idx + 1]
    
    # Create deconvolver if not exists OR after range/DRT update
    if st.session_state.app_state.deconvolver is None:
        deconvolver = GaussianDeconvolver(
            x_linear=cropped_tau,
            y_original=cropped_gamma,  # Используем обрезанные оригинальные ненормированные значения
            use_log_x=True,
            use_log_y=False,
            clip_negative=st.session_state.app_state.clip_negative,
            show_warnings=st.session_state.app_state.show_warnings,
            baseline_method=st.session_state.app_state.baseline_method,
            smoothing_level=st.session_state.app_state.smoothing_level
        )
        st.session_state.app_state.deconvolver = deconvolver
        
        # Auto-detect peaks immediately on entry or range change
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
            st.session_state.app_state.last_popt = None
    
    deconvolver = st.session_state.app_state.deconvolver
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("✂️ Fitting Region (Exclude Tails)")
        
        range_min_idx, range_max_idx = st.slider(
            "Select τ fitting range (by point index)",
            min_value=0,
            max_value=n_total_tau - 1,
            value=(curr_min_idx, curr_max_idx),
            step=1,
            help="Move left/right sliders to exclude regularization tails from peak detection and Gaussian fitting"
        )
        
        # Проверка минимального числа точек для стабильной работы фильтра Савицкого-Голея и curve_fit
        if range_max_idx - range_min_idx < 10:
            st.warning("⚠️ Please select at least 10 points for reliable fitting.")
        elif (range_min_idx, range_max_idx) != (curr_min_idx, curr_max_idx):
            st.session_state.app_state.deconv_tau_range_idx = (range_min_idx, range_max_idx)
            st.session_state.app_state.deconvolver = None
            st.session_state.app_state.manual_peak_position = None
            st.rerun()
        
        tau_start = full_tau[curr_min_idx]
        tau_end = full_tau[curr_max_idx]
        f_low = 1.0 / (2 * np.pi * tau_end)
        f_high = 1.0 / (2 * np.pi * tau_start)
        
        st.caption(
            f"**Active τ:** `{tau_start:.2e}` – `{tau_end:.2e}` s  \n"
            f"**Active f:** `{f_low:.2e}` – `{f_high:.2e}` Hz  \n"
            f"**Points in fit:** {curr_max_idx - curr_min_idx + 1} / {n_total_tau}"
        )
        
        if (curr_min_idx, curr_max_idx) != (0, n_total_tau - 1):
            if st.button("🔄 Reset Fitting Range", use_container_width=True):
                st.session_state.app_state.deconv_tau_range_idx = (0, n_total_tau - 1)
                st.session_state.app_state.deconvolver = None
                st.session_state.app_state.manual_peak_position = None
                st.rerun()
        
        st.markdown("---")
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
        
        # Get number of data points in active region for point selection
        n_points = len(deconvolver.x_linear)
        
        # Create slider by point index (1-based for user-friendly)
        point_index = st.slider("Select peak by point index:",
                               min_value=1,
                               max_value=n_points,
                               value=n_points // 2,
                               step=1,
                               help="Select point index (1 to {}) inside active region to add peak at that position".format(n_points))
        
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
                            progress_callback=update_progress,
                            use_iterative_refinement=True  # Enable 3-stage optimization
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if success:
                            st.session_state.app_state.last_popt = deconvolver.popt
                            st.session_state.app_state.deconv_result = deconvolver.create_deconvolution_result()
                            st.session_state.app_state.deconv_calculated = True
                            st.session_state.app_state.refined_rq_params = None
                            st.session_state.app_state.current_step = 4
                            st.success("✅ Deconvolution complete!")
                            st.rerun()
                        else:
                            st.error("Deconvolution failed. Try adjusting parameters.")
    
    with col2:
        st.subheader("📊 Peak Detection Preview")
        
        if st.session_state.app_state.peak_info is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Make axis tick labels black
            ax.tick_params(axis='both', colors='black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            
            if deconvolver.use_log_x:
                ax.set_xscale('log')
            
            # Визуализация отсекаемых хвостов серым фоном и пунктиром
            is_cropped = (curr_min_idx > 0) or (curr_max_idx < n_total_tau - 1)
            if is_cropped:
                ax.plot(full_tau, full_gamma, 'o--', markersize=2.5, linewidth=1, 
                        alpha=0.35, color='#7f8c8d', label='Excluded Tails', zorder=1)
                if curr_min_idx > 0:
                    ax.axvspan(full_tau[0], tau_start, color='#e2e8f0', alpha=0.65, zorder=0)
                    ax.axvline(x=tau_start, color='#4a5568', linestyle=':', linewidth=1.5, zorder=2)
                if curr_max_idx < n_total_tau - 1:
                    ax.axvspan(tau_end, full_tau[-1], color='#e2e8f0', alpha=0.65, zorder=0)
                    ax.axvline(x=tau_end, color='#4a5568', linestyle=':', linewidth=1.5, zorder=2)
            
            # Отображаем активную область DRT оригинального масштаба
            ax.plot(deconvolver.x_linear, deconvolver.y_original, 
                   'o-', markersize=3, linewidth=1.2, alpha=0.85, 
                   label='DRT Data (active region)', color='black', zorder=2)
            
            source_colors = {'auto': '#2ca02c', 'manual': '#ff7f0e', 'residuals': '#1f77b4'}
            peaks_legend_added = False
            for idx, info in enumerate(st.session_state.app_state.peak_info):
                source = info.get('source', 'auto')
                color = source_colors.get(source, '#2ca02c')
                peak_label = 'Detected Peaks' if not peaks_legend_added else "_nolegend_"
                peaks_legend_added = True
                
                # Используем y_original для отображения
                y_val = info.get('y_original', info.get('y', 0))
                ax.plot(info['x_linear'], y_val, 'o', 
                       markersize=8, markeredgecolor='darkred', 
                       markerfacecolor=color, label=peak_label, zorder=4)
                ax.text(info['x_linear'], y_val * 1.05, 
                       f'τ={info["x_linear"]:.2e}s', ha='center', 
                       fontsize=8, rotation=45)
            
            if st.session_state.app_state.manual_peak_position is not None:
                idx = np.argmin(np.abs(deconvolver.x_linear - st.session_state.app_state.manual_peak_position))
                y_at_position = deconvolver.y_original[idx]
                ax.axvline(x=st.session_state.app_state.manual_peak_position, 
                          color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)
                ax.plot(st.session_state.app_state.manual_peak_position, y_at_position, 
                       'ro', markersize=10, zorder=5)
            
            ax.set_xlabel('Relaxation Time τ (s)', fontweight='bold')
            ax.set_ylabel('γ(τ) (Ω)', fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            st.pyplot(fig)
            plt.close()
            
            # Подсказка с пересчитанным интегралом сопротивления без учета хвостов
            full_integral = drt_result.get_integral()
            active_integral = deconvolver.drt_integral
            st.caption(
                f"💡 **Active region $R_{{pol}}$ integral:** `{active_integral:.4f} Ω` "
                f"(Full spectrum integral with tails: `{full_integral:.4f} Ω`)"
            )
            
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
                        # Отображаем оригинальное значение y
                        y_value = info.get('y_original', info.get('y', 0))
                        st.write(f"γ = {y_value:.4e} Ω")
                    with col_e:
                        if st.button("🗑️", key=f"delete_peak_{i}", help=f"Delete peak {i+1}"):
                            if 0 <= i < len(st.session_state.app_state.peak_info):
                                # Удаляем информацию о пике
                                st.session_state.app_state.peak_info.pop(i)
                                # Удаляем параметры пика (3 параметра на пик)
                                if st.session_state.app_state.initial_peak_params:
                                    start_idx = i * 3
                                    del st.session_state.app_state.initial_peak_params[start_idx:start_idx + 3]
                                st.success(f"Peak {i+1} removed")
                                st.rerun()
                
                st.markdown("---")
                
                # Additional statistics
                col1_stat, col2_stat, col3_stat = st.columns(3)
                with col1_stat:
                    auto_count = sum(1 for p in st.session_state.app_state.peak_info if p.get('source', 'auto') == 'auto')
                    st.metric("Auto-detected", auto_count)
                with col2_stat:
                    manual_count = sum(1 for p in st.session_state.app_state.peak_info if p.get('source', '') == 'manual')
                    st.metric("Manually added", manual_count)
                with col3_stat:
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
    drt_result = st.session_state.app_state.drt_result
    data = st.session_state.app_state.impedance_data
    
    # Navigation buttons
    col_prev, col_next = st.columns([1, 5])
    with col_prev:
        if st.button("⬅️ Back to Deconvolution", use_container_width=True):
            st.session_state.app_state.current_step = 3
            st.rerun()
    
    st.markdown("---")
    
    # Tabs for different views - UPDATED: added "Output Graphs" tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Deconvolution Plot", "📊 Area Distribution", "📋 Complete Dataset", 
                                                   "📈 Normalized View", "📊 Output Graphs", "📥 Export"])
    
    with tab1:
        st.subheader("Gaussian Deconvolution Result")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Make axis tick labels black
        ax.tick_params(axis='both', colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        
        if deconv_result.use_log_x:
            ax.set_xscale('log')
        
        ax.scatter(deconv_result.x_linear, deconv_result.y_original, 
                   s=15, alpha=0.5, color='black', label='Original DRT Data', zorder=1)
        
        if deconv_result.use_log_x:
            x_min = max(np.min(deconv_result.x_linear[deconv_result.x_linear > 0]), 1e-15)
            x_max = np.max(deconv_result.x_linear)
            x_dense = np.logspace(np.log10(x_min), np.log10(x_max), 2000)
            x_dense_log = np.log10(x_dense)
        else:
            x_dense = np.linspace(np.min(deconv_result.x_linear), 
                                  np.max(deconv_result.x_linear), 2000)
            x_dense_log = x_dense
        
        style = get_plot_style()
        colors = plt.get_cmap(style["cmap"])(np.linspace(0, 1, max(len(deconv_result.peaks), 1)))
        color_map = {peak.id: colors[i] for i, peak in enumerate(deconv_result.peaks)}
        
        if deconv_result.peaks:
            for peak in deconv_result.peaks:
                color = color_map[peak.id]
                y_component = peak.amplitude * GaussianModelDeconv.gaussian(
                    x_dense_log, 1.0, peak.center_log, peak.sigma_log
                )
                
                ax.fill_between(x_dense, 0, y_component, 
                                color=color, alpha=0.3, linewidth=0)
                ax.plot(x_dense, y_component, '-', color=color, linewidth=2,
                       label=f'Peak {peak.id}: {peak.fraction_percent:.1f}%', zorder=2)
        
        if deconv_result.baseline_params and deconv_result.baseline_method != 'none':
            if deconv_result.baseline_method == 'constant':
                y_baseline = np.full_like(x_dense, deconv_result.baseline_params[0])
                ax.plot(x_dense, y_baseline, 'gray', linestyle=':', linewidth=1.5, label='Baseline', zorder=1)
            elif deconv_result.baseline_method == 'linear':
                y_baseline = deconv_result.baseline_params[0] + deconv_result.baseline_params[1] * x_dense_log
                ax.plot(x_dense, y_baseline, 'gray', linestyle=':', linewidth=1.5, label='Baseline', zorder=1)
            elif deconv_result.baseline_method == 'quadratic':
                y_baseline = (deconv_result.baseline_params[0] + 
                             deconv_result.baseline_params[1] * x_dense_log +
                             deconv_result.baseline_params[2] * x_dense_log**2)
                ax.plot(x_dense, y_baseline, 'gray', linestyle=':', linewidth=1.5, label='Baseline', zorder=1)
        
        if deconv_result.fit_y_original is not None:
            from scipy.interpolate import interp1d
            interp_func = interp1d(deconv_result.x_linear, deconv_result.fit_y_original, 
                                   kind='cubic', fill_value='extrapolate')
            y_total_interp = interp_func(x_dense)
            ax.plot(x_dense, y_total_interp, 'r--', linewidth=2.5, label='Total Fit', zorder=3)
        elif deconv_result.peaks:
            y_total = np.zeros_like(x_dense)
            for peak in deconv_result.peaks:
                y_total += peak.amplitude * GaussianModelDeconv.gaussian(
                    x_dense_log, 1.0, peak.center_log, peak.sigma_log
                )
            
            if deconv_result.baseline_params and deconv_result.baseline_method != 'none':
                if deconv_result.baseline_method == 'constant':
                    y_total += deconv_result.baseline_params[0]
                elif deconv_result.baseline_method == 'linear':
                    y_total += deconv_result.baseline_params[0] + deconv_result.baseline_params[1] * x_dense_log
                elif deconv_result.baseline_method == 'quadratic':
                    y_total += (deconv_result.baseline_params[0] + 
                               deconv_result.baseline_params[1] * x_dense_log +
                               deconv_result.baseline_params[2] * x_dense_log**2)
            
            ax.plot(x_dense, y_total, 'r--', linewidth=2.5, label='Total Fit', zorder=3)
        
        ax.set_xlabel('Relaxation Time τ (s)', fontweight='bold', fontsize=12)
        ax.set_ylabel('γ(τ) (Ω)', fontweight='bold', fontsize=12)
        # No title - removed as requested
        ax.legend(loc='upper left', fontsize=9, frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if deconv_result.quality_metrics:
            metrics_text = f"R² = {deconv_result.quality_metrics.get('R²', 0):.4f}\n"
            metrics_text += f"RMSE = {deconv_result.quality_metrics.get('RMSE', 0):.2e}"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.tight_layout()
        render_plot_with_download(fig, "deconvolution_plot", key_prefix="tab1")
    
    with tab2:
        st.subheader("Area Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            # Make axis tick labels black
            ax.tick_params(axis='both', colors='black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            
            # Sort peaks by ID
            peaks_sorted = sorted(deconv_result.peaks, key=lambda p: p.id)
            peaks_ids = [f'Peak {p.id}' for p in peaks_sorted]
            fractions = [p.fraction_percent for p in peaks_sorted]
            colors = plt.cm.Set3(np.linspace(0, 1, len(deconv_result.peaks)))
            color_map = {peak.id: colors[i] for i, peak in enumerate(deconv_result.peaks)}
            colors_sorted = [color_map[p.id] for p in peaks_sorted]
            
            bars = ax.bar(peaks_ids, fractions, color=colors_sorted, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Peak', fontweight='bold')
            ax.set_ylabel('Fraction (%)', fontweight='bold')
            # No title - removed as requested
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
            fig, ax = plt.subplots(figsize=(8, 6))
            wedges, texts, autotexts = ax.pie(fractions, labels=peaks_ids, autopct='%1.1f%%',
                                               colors=colors_sorted, startangle=90,
                                               textprops={'fontweight': 'bold'})
            # No title - removed as requested
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
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
        
        # Create detailed table with Fmax and C columns
        data_rows = []
        for peak in deconv_result.peaks:
            char = calculate_peak_characteristics(peak)
            data_rows.append({
                'Peak ID': peak.id,
                'Center (τ, s)': f"{peak.center:.4e}",
                'Center (log τ)': f"{peak.center_log:.4f}",
                'Fmax (Hz)': f"{char['fmax_hz']:.4e}",
                'C (F)': f"{char['c_farad']:.4e}",
                'Amplitude (Ω)': f"{peak.amplitude:.4e}",
                'Amplitude (norm)': f"{peak.amplitude_norm:.4f}",
                'Sigma (log)': f"{peak.sigma_log:.4f}",
                'FWHM': f"{peak.fwhm:.4f}",
                'Area (Ω·s)': f"{peak.area:.4e}",
                'Fraction (%)': f"{peak.fraction_percent:.2f}",
                'Source': peak.source
            })
        
        df = pd.DataFrame(data_rows)
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
        
        if drt_result:
            st.markdown("---")
            st.subheader("DRT Calculation Metadata")
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.metric("Method", drt_result.method)
                st.metric("R∞", f"{drt_result.R_inf:.4f} Ω")
            with meta_col2:
                # Rpol should equal sum of peak areas from deconvolution
                if deconv_result is not None and deconv_result.peaks:
                    rpol_from_peaks = sum([p.area for p in deconv_result.peaks])
                    st.metric("Rpol (sum of peak areas)", f"{rpol_from_peaks:.4f} Ω")
                    # Also show DRT integral for reference
                    st.caption(f"DRT integral: {drt_result.R_pol:.4f} Ω")
                else:
                    st.metric("Rpol", f"{drt_result.R_pol:.4f} Ω")
                if drt_result.L > 0:
                    st.metric("Inductance L", f"{drt_result.L:.3e} H")
            with meta_col3:
                if drt_result.lambda_opt is not None:
                    st.metric("Optimal λ", f"{drt_result.lambda_opt:.3e}")
                integral, ratio = drt_result.verify_integral()
                st.metric("Integral Ratio", f"{ratio:.3f}")
    
        
    with tab4:
        st.subheader("Normalized Gaussian Deconvolution Result")
        
        fig = plt.figure(figsize=(12, 9))
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[3, 1.2],
            width_ratios=[1, 0.025],
            hspace=0.08,
            wspace=0.03
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_res = fig.add_subplot(gs[1, 0], sharex=ax)
        cax = fig.add_subplot(gs[1, 1])
        
        # Make axis tick labels black
        for axis in (ax, ax_res):
            axis.tick_params(axis='both', colors='black')
            axis.xaxis.label.set_color('black')
            axis.yaxis.label.set_color('black')
        
        if deconv_result.use_log_x:
            ax.set_xscale('log')
            ax_res.set_xscale('log')
        
        max_amp = max(deconv_result.y_original) if len(deconv_result.y_original) > 0 else 1.0
        y_original_norm = deconv_result.y_original / max_amp
        
        ax.scatter(deconv_result.x_linear, y_original_norm, 
                   s=15, alpha=0.5, color='black', label='Original DRT Data (normalized)', zorder=1)
        
        if deconv_result.use_log_x:
            x_min = max(np.min(deconv_result.x_linear[deconv_result.x_linear > 0]), 1e-15)
            x_max = np.max(deconv_result.x_linear)
            x_dense = np.logspace(np.log10(x_min), np.log10(x_max), 2000)
            x_dense_log = np.log10(x_dense)
            x_data_eval = np.log10(np.maximum(deconv_result.x_linear, 1e-15))
        else:
            x_dense = np.linspace(np.min(deconv_result.x_linear), 
                                  np.max(deconv_result.x_linear), 2000)
            x_dense_log = x_dense
            x_data_eval = deconv_result.x_linear
        
        style = get_plot_style()
        colors = plt.get_cmap(style["cmap"])(np.linspace(0, 1, max(len(deconv_result.peaks), 1)))
        color_map = {peak.id: colors[i] for i, peak in enumerate(deconv_result.peaks)}
        
        if deconv_result.peaks:
            for peak in deconv_result.peaks:
                color = color_map[peak.id]
                y_component_norm = (peak.amplitude * GaussianModelDeconv.gaussian(
                    x_dense_log, 1.0, peak.center_log, peak.sigma_log
                )) / max_amp
                
                ax.fill_between(x_dense, 0, y_component_norm, 
                                color=color, alpha=0.3, linewidth=0)
                ax.plot(x_dense, y_component_norm, '-', color=color, linewidth=2,
                       label=f'Peak {peak.id}: {peak.fraction_percent:.1f}%', zorder=2)
        
        if deconv_result.baseline_params and deconv_result.baseline_method != 'none':
            if deconv_result.baseline_method == 'constant':
                y_baseline_norm = deconv_result.baseline_params[0] / max_amp
                ax.axhline(y=y_baseline_norm, color='gray', linestyle=':', linewidth=1.5, label='Baseline', zorder=1)
            elif deconv_result.baseline_method == 'linear':
                y_baseline_norm = (deconv_result.baseline_params[0] + 
                                  deconv_result.baseline_params[1] * x_dense_log) / max_amp
                ax.plot(x_dense, y_baseline_norm, 'gray', linestyle=':', linewidth=1.5, label='Baseline', zorder=1)
            elif deconv_result.baseline_method == 'quadratic':
                y_baseline_norm = (deconv_result.baseline_params[0] + 
                                  deconv_result.baseline_params[1] * x_dense_log +
                                  deconv_result.baseline_params[2] * x_dense_log**2) / max_amp
                ax.plot(x_dense, y_baseline_norm, 'gray', linestyle=':', linewidth=1.5, label='Baseline', zorder=1)
        
        # Calculate normalized total fit at dense grid and at exact data points
        if deconv_result.fit_y_original is not None:
            from scipy.interpolate import interp1d
            fit_at_data_norm = deconv_result.fit_y_original / max_amp
            interp_func = interp1d(deconv_result.x_linear, fit_at_data_norm, 
                                   kind='cubic', fill_value='extrapolate')
            y_total_norm_interp = interp_func(x_dense)
            ax.plot(x_dense, y_total_norm_interp, 'r--', linewidth=2.5, label='Total Fit (normalized)', zorder=3)
        else:
            y_total_norm = np.zeros_like(x_dense)
            fit_at_data_norm = np.zeros_like(deconv_result.x_linear, dtype=float)
            for peak in deconv_result.peaks:
                y_total_norm += (peak.amplitude * GaussianModelDeconv.gaussian(
                    x_dense_log, 1.0, peak.center_log, peak.sigma_log
                )) / max_amp
                fit_at_data_norm += (peak.amplitude * GaussianModelDeconv.gaussian(
                    x_data_eval, 1.0, peak.center_log, peak.sigma_log
                )) / max_amp
            
            if deconv_result.baseline_params and deconv_result.baseline_method != 'none':
                if deconv_result.baseline_method == 'constant':
                    y_total_norm += deconv_result.baseline_params[0] / max_amp
                    fit_at_data_norm += deconv_result.baseline_params[0] / max_amp
                elif deconv_result.baseline_method == 'linear':
                    y_total_norm += (deconv_result.baseline_params[0] + 
                                    deconv_result.baseline_params[1] * x_dense_log) / max_amp
                    fit_at_data_norm += (deconv_result.baseline_params[0] + 
                                        deconv_result.baseline_params[1] * x_data_eval) / max_amp
                elif deconv_result.baseline_method == 'quadratic':
                    y_total_norm += (deconv_result.baseline_params[0] + 
                                    deconv_result.baseline_params[1] * x_dense_log +
                                    deconv_result.baseline_params[2] * x_dense_log**2) / max_amp
                    fit_at_data_norm += (deconv_result.baseline_params[0] + 
                                        deconv_result.baseline_params[1] * x_data_eval +
                                        deconv_result.baseline_params[2] * x_data_eval**2) / max_amp
            
            ax.plot(x_dense, y_total_norm, 'r--', linewidth=2.5, label='Total Fit (normalized)', zorder=3)
        
        ax.set_ylabel('Normalized Intensity (γ(τ) / γ_max)', fontweight='bold', fontsize=12)
        ax.legend(loc='upper left', fontsize=9, frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelbottom=False)
        
        # --- Residual curve with thermal color scale ---
        residuals_norm = y_original_norm - fit_at_data_norm
        ss_res = np.sum(residuals_norm ** 2)
        ss_tot = np.sum((y_original_norm - np.mean(y_original_norm)) ** 2)
        r2_norm = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse_norm = float(np.sqrt(np.mean(residuals_norm ** 2)))
        max_abs_res = max(float(np.max(np.abs(residuals_norm))), 1e-9)
        
        ax_res.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=1)
        ax_res.plot(deconv_result.x_linear, residuals_norm, '-', color='gray', alpha=0.5, linewidth=1.0, zorder=2)
        sc = ax_res.scatter(
            deconv_result.x_linear, residuals_norm,
            c=residuals_norm, cmap='coolwarm',
            vmin=-max_abs_res, vmax=max_abs_res,
            s=28, edgecolors='black', linewidths=0.3, zorder=3
        )
        
        # Colorbar aligned with residual subplot
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label('Δ (norm)', fontsize=9, fontweight='bold')
        cbar.ax.tick_params(labelsize=8, colors='black')
        
        ax_res.set_xlabel('Relaxation Time τ (s)', fontweight='bold', fontsize=12)
        ax_res.set_ylabel('Residuals\n(Data - Fit)', fontweight='bold', fontsize=10)
        ax_res.set_ylim(-max_abs_res * 1.25, max_abs_res * 1.25)
        ax_res.grid(True, alpha=0.3, linestyle='--')
        
        metrics_text = f"R² = {r2_norm:.5f}  |  RMSE = {rmse_norm:.2e}"
        ax_res.text(
            0.015, 0.90, metrics_text, transform=ax_res.transAxes,
            fontsize=9, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.85, edgecolor='gray')
        )
        
        render_plot_with_download(fig, "normalized_deconvolution", key_prefix="tab4")
        
        st.caption(f"Normalized to maximum value γ_max = {max_amp:.4e} Ω")
    
    with tab5:
        st.subheader("📊 Output Graphs")
        st.markdown("Comprehensive visualization of impedance spectroscopy analysis results")
        
        # Graph 2.1 - Original Impedance Spectrum
        st.markdown("**2.1 Original Impedance Spectrum**")
        if data is not None:
            fig1 = plot_original_nyquist_with_frequency_labels(data, "Original Nyquist Spectrum")
            render_plot_with_download(fig1, "original_nyquist", key_prefix="tab5")
        else:
            st.warning("No impedance data available")
        
        st.markdown("***")  # Separator
        
        # Graph 2.2 - Gaussian Deconvolution vs Frequency
        st.markdown("**2.2 Gaussian Deconvolution vs Frequency**")
        if deconv_result is not None:
            fig2 = plot_deconvolution_vs_frequency(deconv_result, drt_result, 
                                                   "DRT Deconvolution vs Frequency")
            render_plot_with_download(fig2, "deconvolution_vs_freq", key_prefix="tab5")
        else:
            st.warning("No deconvolution results available")
        
        st.markdown("***")  # Separator
        
        # Graph 2.3 - Peak Area Distribution (Resistance Contributions)
        st.markdown("**2.3 Peak Area Distribution (Resistance Contributions)**")
        if deconv_result is not None:
            fig3 = plot_peak_area_distribution_with_values(deconv_result, drt_result,
                                                           "Resistance Distribution by Process")
            render_plot_with_download(fig3, "resistance_distribution", key_prefix="tab5")
        else:
            st.warning("No deconvolution results available")
        
        st.markdown("***")  # Separator

        # Graph 2.4 - Experimental vs Sequential RQ Model
        st.markdown("**2.4 Experimental vs Sequential RQ (ZARC) Model**")
        if deconv_result is not None and drt_result is not None and data is not None:
            refined_rq = st.session_state.app_state.refined_rq_params

            col_ref1, col_ref2 = st.columns([2, 1])
            with col_ref1:
                if st.button("🎯 Refine RQ (ZARC) Model on Experimental EIS (CNLS)",
                             type="primary", use_container_width=True, key="btn_refine_rq"):
                    with st.spinner("Optimizing ZARC parameters against experimental EIS..."):
                        st.session_state.app_state.refined_rq_params = refine_rq_zarc_model(
                            deconv_result, drt_result, data
                        )
                    st.rerun()
            with col_ref2:
                if refined_rq is not None:
                    if st.button("🔄 Reset to DRT Model", use_container_width=True, key="btn_reset_rq"):
                        st.session_state.app_state.refined_rq_params = None
                        st.rerun()

            if refined_rq is not None:
                st.success(
                    f"✅ CNLS Refinement active | R∞ = {refined_rq['R_inf']:.4f} Ω | "
                    f"R² = {refined_rq['r2']:.5f} | RMSE = {refined_rq['rmse']:.2e} Ω"
                )

            fig4 = plot_sequential_rc_model(
                deconv_result, drt_result, data,
                "Experimental vs Sequential RQ Model",
                refined_rq=refined_rq
            )
            render_plot_with_download(fig4, "sequential_rq_model", key_prefix="tab5")

            # Display R_i, Q_i, alpha_i, C_i, f_i table directly under Graph 2.4
            if refined_rq is not None:
                st.markdown("**Equivalent RQ (ZARC) Parameters per Process (CNLS Refined):**")
                rq_rows = [
                    {
                        'Process': f"Process {p['id']}",
                        'R_i (Ω)': f"{p['R_i']:.4e}",
                        'Q_i (F·s^(α-1))': f"{p['Q_i']:.4e}",
                        'α_i': f"{p['alpha_i']:.4f}",
                        'C_i (F)': f"{p['C_i']:.4e}",
                        'f_i (Hz)': f"{p['f_i']:.4e}"
                    }
                    for p in refined_rq['processes']
                ]
                st.dataframe(pd.DataFrame(rq_rows), use_container_width=True, hide_index=True)
            elif deconv_result.peaks:
                st.markdown("**Equivalent RQ (ZARC) Parameters per Process (from DRT):**")
                rq_rows = []
                for peak in sorted(deconv_result.peaks, key=lambda p: p.id):
                    rq = calculate_rq_parameters(peak)
                    rq_rows.append({
                        'Process': f"Process {peak.id}",
                        'R_i (Ω)': f"{rq['R_i']:.4e}",
                        'Q_i (F·s^(α-1))': f"{rq['Q_i']:.4e}",
                        'α_i': f"{rq['alpha_i']:.4f}",
                        'C_i (F)': f"{rq['C_i']:.4e}",
                        'f_i (Hz)': f"{rq['f_i']:.4e}"
                    })
                st.dataframe(pd.DataFrame(rq_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("Insufficient data for model comparison")
        
        # Add explanatory text
        st.markdown("---")
        st.markdown("""
        **Graph Explanations:**
- **2.1** - Original impedance spectrum with marked extreme and decade frequency points
- **2.2** - DRT deconvolution results plotted against frequency (high to low)
- **2.3** - Bar chart showing resistance contribution (Area in Ω) and percentage for each process
- **2.4** - Comparison of experimental data with sequential RQ (ZARC) model (each depressed semicircle represents one relaxation process)
        """)
        
        
    with tab6:
        st.subheader("Export Results")
        
        export_format = st.radio(
            "Select export format:",
            options=["CSV (comma-separated)", "TXT (tab-separated)"],
            horizontal=True,
            key="export_format_selector"
        )
        is_txt = export_format.startswith("TXT")
        file_ext = "txt" if is_txt else "csv"
        file_sep = "\t" if is_txt else ","
        file_mime = "text/plain" if is_txt else "text/csv"
        fmt_label = "TXT" if is_txt else "CSV"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Export Peak Data** *(for Complete Dataset table)*")
            
            # Create peaks DataFrame with Fmax and C columns
            peaks_data = []
            for p in deconv_result.peaks:
                char = calculate_peak_characteristics(p)
                peaks_data.append({
                    'Peak_ID': p.id,
                    'Center_tau_s': p.center,
                    'Center_log_tau': p.center_log,
                    'Fmax_Hz': char['fmax_hz'],
                    'C_Farad': char['c_farad'],
                    'Amplitude_Ohm': p.amplitude,
                    'Amplitude_Normalized': p.amplitude_norm,
                    'Sigma_log': p.sigma_log,
                    'FWHM': p.fwhm,
                    'Area_Ohm_s': p.area,
                    'Fraction': p.fraction,
                    'Fraction_Percent': p.fraction_percent,
                    'Source': p.source,
                    'Characteristic_Frequency_Hz': p.get_characteristic_frequency(),
                    'Resistance_Contribution_Ohm': p.get_resistance_contribution()
                })
            
            peaks_df = pd.DataFrame(peaks_data)
            data_peaks = peaks_df.to_csv(index=False, sep=file_sep)
            st.download_button(
                f"📥 Export Peaks as {fmt_label}",
                data=data_peaks,
                file_name=f"deconvolution_peaks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                mime=file_mime,
                use_container_width=True
            )
            
            st.markdown("**Export Fitting Data** *(from Gaussian Deconvolution Result)*")
            
            if deconv_result.fit_y_original is not None:
                fit_values = deconv_result.fit_y_original
            else:
                fit_values = np.zeros_like(deconv_result.x_linear)
                for i, tau in enumerate(deconv_result.x_linear):
                    if deconv_result.use_log_x:
                        log_tau = np.log10(tau)
                    else:
                        log_tau = tau
                    
                    total = 0
                    for peak in deconv_result.peaks:
                        total += peak.amplitude * GaussianModelDeconv.gaussian(
                            log_tau, 1.0, peak.center_log, peak.sigma_log
                        )
                    
                    if deconv_result.baseline_params and deconv_result.baseline_method != 'none':
                        if deconv_result.baseline_method == 'constant':
                            total += deconv_result.baseline_params[0]
                        elif deconv_result.baseline_method == 'linear':
                            total += deconv_result.baseline_params[0] + deconv_result.baseline_params[1] * log_tau
                        elif deconv_result.baseline_method == 'quadratic':
                            total += (deconv_result.baseline_params[0] + 
                                     deconv_result.baseline_params[1] * log_tau +
                                     deconv_result.baseline_params[2] * log_tau**2)
                    
                    fit_values[i] = total
            
            residuals = deconv_result.y_original - fit_values
            
            fit_data = pd.DataFrame({
                'tau_s': deconv_result.x_linear,
                'gamma_tau_Ohm': deconv_result.y_original,
                'gamma_fit_Ohm': fit_values,
                'Residuals_Ohm': residuals
            })
            
            data_fit = fit_data.to_csv(index=False, sep=file_sep)
            st.download_button(
                f"📥 Export Fitting Data as {fmt_label}",
                data=data_fit,
                file_name=f"deconvolution_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                mime=file_mime,
                use_container_width=True
            )
            
            # --- Export Normalized Gaussian Deconvolution Result ---
            st.markdown("**Export Normalized Deconvolution Data** *(Normalized spectrum + Total Fit + Residuals + Peaks)*")
            
            peaks_sorted_exp = sorted(deconv_result.peaks, key=lambda p: p.id)
            max_amp_exp = max(deconv_result.y_original) if len(deconv_result.y_original) > 0 else 1.0
            x_eval = np.log10(np.maximum(deconv_result.x_linear, 1e-15)) if deconv_result.use_log_x else deconv_result.x_linear
            
            norm_dict = {
                'tau_s': deconv_result.x_linear,
                'gamma_tau_norm': deconv_result.y_original / max_amp_exp,
                'Total_Fit_norm': fit_values / max_amp_exp,
                'Residuals_norm': residuals / max_amp_exp
            }
            for peak in peaks_sorted_exp:
                y_comp_norm = (peak.amplitude * GaussianModelDeconv.gaussian(
                    x_eval, 1.0, peak.center_log, peak.sigma_log
                )) / max_amp_exp
                norm_dict[f'Peak_{peak.id}_norm'] = y_comp_norm
            
            data_norm = pd.DataFrame(norm_dict).to_csv(index=False, sep=file_sep)
            st.download_button(
                f"📥 Export Normalized Deconvolution as {fmt_label}",
                data=data_norm,
                file_name=f"deconvolution_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                mime=file_mime,
                use_container_width=True
            )

            # --- Export 2.2 Gaussian Deconvolution vs Frequency ---
            st.markdown("**Export 2.2 Deconvolution vs Frequency** *(Spectrum + Total Fit + Peaks)*")
            
            freqs_deconv = 1.0 / (2.0 * np.pi * deconv_result.x_linear)
            sort_idx_f = np.argsort(freqs_deconv)[::-1]  # high to low, as on Graph 2.2
            log_tau_f = np.log10(np.maximum(deconv_result.x_linear[sort_idx_f], 1e-15))
            
            freq_deconv_dict = {
                'Frequency_Hz': freqs_deconv[sort_idx_f],
                'gamma_tau_Ohm': deconv_result.y_original[sort_idx_f],
                'Total_Fit_Ohm': fit_values[sort_idx_f]
            }
            for peak in peaks_sorted_exp:
                y_comp_f = peak.amplitude * GaussianModelDeconv.gaussian(
                    log_tau_f, 1.0, peak.center_log, peak.sigma_log
                )
                freq_deconv_dict[f'Peak_{peak.id}_Ohm'] = y_comp_f
            
            data_freq_deconv = pd.DataFrame(freq_deconv_dict).to_csv(index=False, sep=file_sep)
            st.download_button(
                f"📥 Export 2.2 Deconvolution vs Frequency as {fmt_label}",
                data=data_freq_deconv,
                file_name=f"deconvolution_vs_freq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                mime=file_mime,
                use_container_width=True
            )

            # --- Export 2.4 Experimental vs Sequential RQ (ZARC) Model ---
            if data is not None and drt_result is not None:
                st.markdown("**Export 2.4 Sequential RQ (ZARC) Model** *(Experimental + Total Model + Residuals + Processes)*")
                
                sort_idx_exp = np.argsort(data.freq)
                freq_exp_s = data.freq[sort_idx_exp]
                re_exp_s = data.re_z[sort_idx_exp]
                im_exp_s = data.im_z[sort_idx_exp]
                omega_exp_s = 2.0 * np.pi * freq_exp_s
                
                refined_rq = st.session_state.app_state.refined_rq_params
                process_cols = {}
                
                if refined_rq is not None:
                    R_inf_start = refined_rq['R_inf']
                    Z_total_exp = np.full_like(omega_exp_s, R_inf_start, dtype=complex)
                    if refined_rq['L'] > 0:
                        Z_total_exp += 1j * omega_exp_s * refined_rq['L']
                    
                    cumulative_R = R_inf_start
                    for p in refined_rq['processes']:
                        Z_i = p['R_i'] / (1.0 + ((1j * omega_exp_s) ** p['alpha_i']) * p['R_i'] * p['Q_i'])
                        Z_total_exp += Z_i
                        Z_indiv = cumulative_R + Z_i
                        process_cols[f"Re_Process_{p['id']}_Ohm"] = np.real(Z_indiv)
                        process_cols[f"NegIm_Process_{p['id']}_Ohm"] = -np.imag(Z_indiv)
                        cumulative_R += p['R_i']
                else:
                    R_inf_start = drt_result.R_inf
                    for i in range(len(im_exp_s) - 1):
                        if im_exp_s[i] >= 0 and im_exp_s[i + 1] < 0:
                            t = -im_exp_s[i] / (im_exp_s[i + 1] - im_exp_s[i])
                            R_inf_start = float(re_exp_s[i] + t * (re_exp_s[i + 1] - re_exp_s[i]))
                            break
                    
                    Z_total_exp = np.full_like(omega_exp_s, R_inf_start, dtype=complex)
                    if drt_result.L > 0:
                        Z_total_exp += 1j * omega_exp_s * drt_result.L
                    
                    cumulative_R = R_inf_start
                    for peak in peaks_sorted_exp:
                        sigma = max(float(peak.sigma_log), 1e-6)
                        center_log = float(peak.center_log)
                        if sigma < 1e-4:
                            Z_i = peak.area / (1.0 + 1j * omega_exp_s * peak.center)
                        else:
                            u_grid = np.linspace(center_log - 5.0 * sigma, center_log + 5.0 * sigma, 401)
                            tau_sub = 10.0 ** u_grid
                            gauss_profile = np.exp(-0.5 * ((u_grid - center_log) / sigma) ** 2)
                            profile_area = np.trapezoid(gauss_profile, u_grid)
                            g_weights = (peak.area / profile_area) * gauss_profile if profile_area > 0 else np.zeros_like(u_grid)
                            denom = 1.0 + 1j * np.outer(omega_exp_s, tau_sub)
                            Z_i = np.trapezoid(g_weights[np.newaxis, :] / denom, u_grid, axis=1)
                        
                        Z_total_exp += Z_i
                        Z_indiv = cumulative_R + Z_i
                        process_cols[f'Re_Process_{peak.id}_Ohm'] = np.real(Z_indiv)
                        process_cols[f'NegIm_Process_{peak.id}_Ohm'] = -np.imag(Z_indiv)
                        cumulative_R += peak.area
                
                re_total_mod = np.real(Z_total_exp)
                im_total_mod = -np.imag(Z_total_exp)
                
                rq_export_dict = {
                    'Frequency_Hz': freq_exp_s,
                    'Re_Exp_Ohm': re_exp_s,
                    'NegIm_Exp_Ohm': im_exp_s,
                    'Re_Total_Model_Ohm': re_total_mod,
                    'NegIm_Total_Model_Ohm': im_total_mod,
                    'Residuals_NegIm_Ohm': im_exp_s - im_total_mod
                }
                rq_export_dict.update(process_cols)
                
                data_rq_model = pd.DataFrame(rq_export_dict).to_csv(index=False, sep=file_sep)
                st.download_button(
                    f"📥 Export 2.4 Sequential RQ Model as {fmt_label}",
                    data=data_rq_model,
                    file_name=f"sequential_rq_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                    mime=file_mime,
                    use_container_width=True
                )
        
        with col2:
            st.markdown("**Generate Report**")
            
            if st.button("📄 Generate Detailed Report", use_container_width=True):
                max_amp = max([p.amplitude for p in deconv_result.peaks]) if deconv_result.peaks else 1.0
                refined_rq = st.session_state.app_state.refined_rq_params
                
                report = f"""GAUSSIAN DECONVOLUTION REPORT
{"="*80}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Number of points: {len(deconv_result.x_linear)}
τ range: [{deconv_result.x_linear[0]:.2e}, {deconv_result.x_linear[-1]:.2e}] s
Logarithmic X scale: {deconv_result.use_log_x}
Baseline method: {deconv_result.baseline_method}
Smoothing level: {st.session_state.app_state.smoothing_level}

"""
                
                if drt_result:
                    report += f"""DRT METADATA:
{"-"*40}
Method: {drt_result.method}
R∞: {drt_result.R_inf:.6f} Ω
Rpol: {drt_result.R_pol:.6f} Ω
"""
                    if drt_result.L > 0:
                        report += f"Inductance L: {drt_result.L:.6e} H\n"
                    if drt_result.lambda_opt is not None:
                        report += f"Optimal λ: {drt_result.lambda_opt:.6e}\n"
                    integral, ratio = drt_result.verify_integral()
                    report += f"DRT Integral: {integral:.6f} Ω\n"
                    report += f"Integral/Rpol Ratio: {ratio:.4f}\n\n"
                
                report += f"""QUALITY METRICS:
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
                
                report += f"""PEAK PARAMETERS (DRT DECONVOLUTION):
{"-"*100}
ID    Center (s)      Fmax (Hz)       C (F)           Area (Ω·s)     Fraction(%)    
{"-"*100}"""
                
                for p in deconv_result.peaks:
                    char = calculate_peak_characteristics(p)
                    report += f"\n{p.id:<4} {p.center:.4e}   {char['fmax_hz']:.4e}   {char['c_farad']:.4e}   {p.area:.4e}   {p.fraction_percent:.2f}"
                
                if refined_rq is not None:
                    report += f"""

CNLS REFINED RQ (ZARC) MODEL PARAMETERS:
{"-"*100}
R∞: {refined_rq['R_inf']:.6f} Ω | L: {refined_rq['L']:.6e} H | R²: {refined_rq['r2']:.6f} | RMSE: {refined_rq['rmse']:.4e} Ω
ID    R_i (Ω)         Q_i (F·s^(α-1)) α_i        C_i (F)         f_i (Hz)
{"-"*100}"""
                    for p in refined_rq['processes']:
                        report += f"\n{p['id']:<4} {p['R_i']:.4e}   {p['Q_i']:.4e}      {p['alpha_i']:.4f}     {p['C_i']:.4e}   {p['f_i']:.4e}"
                
                report += f"""

{"="*80}
Total Area (Rpol): {deconv_result.total_area:.6e} Ω·s
Maximum Amplitude: {max_amp:.6e} Ω
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

# ============================================================================
# Navigation and Main Application
# ============================================================================

def show_step_indicator():
    """Display step indicator and settings in sidebar"""
    # Add logo at the top of sidebar
    try:
        from PIL import Image
        logo = Image.open("logo.png")
        st.sidebar.image(logo, use_container_width=True)
    except Exception as e:
        st.sidebar.warning(f"Logo not found: {e}")
    
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
    
    # Publication Plot Settings expander
    style = get_plot_style()
    with st.sidebar.expander("🎨 Publication Plot Settings", expanded=False):
        font_choice = st.selectbox(
            "Font family",
            options=["serif", "sans-serif"],
            index=0 if style["font_family"] == "serif" else 1,
            help="serif = Times New Roman, sans-serif = Arial"
        )
        fs_label = st.slider("Axis label font size", 8, 18, int(style["fs_label"]))
        fs_tick = st.slider("Tick label font size", 7, 16, int(style["fs_tick"]))
        fs_legend = st.slider("Legend font size", 6, 16, int(style["fs_legend"]))
        lw = st.slider("Line width", 1.0, 4.0, float(style["lw"]), 0.25)
        ms = st.slider("Marker size", 2, 10, int(style["ms"]))
        show_grid = st.checkbox("Show grid lines", value=bool(style["show_grid"]))
        tick_dir = st.radio(
            "Tick direction",
            options=["in", "out"],
            index=0 if style["tick_dir"] == "in" else 1,
            horizontal=True
        )
        axes_bg = st.radio(
            "Plot background",
            options=["white", "#f8f9fa"],
            index=0 if style["axes_bg"] == "white" else 1,
            horizontal=True
        )
        cmap_options = ["Set3", "tab10", "Dark2", "viridis", "plasma", "coolwarm", "turbo"]
        cmap_idx = cmap_options.index(style["cmap"]) if style["cmap"] in cmap_options else 0
        cmap = st.selectbox("Peak / Process colormap", options=cmap_options, index=cmap_idx)
        dpi = st.selectbox("Export PNG DPI", options=[300, 600, 900], index=[300, 600, 900].index(style["dpi"]))

        st.session_state.plot_style = {
            "font_family": font_choice,
            "fs_label": fs_label,
            "fs_tick": fs_tick,
            "fs_legend": fs_legend,
            "lw": lw,
            "ms": ms,
            "show_grid": show_grid,
            "tick_dir": tick_dir,
            "axes_bg": axes_bg,
            "cmap": cmap,
            "dpi": dpi,
        }

    # Apply publication style globally to matplotlib
    apply_publication_plot_style(st.session_state.plot_style)
    
    # Global Settings expander
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
    st.markdown("⚡ **EIS-DRT Analysis Tool v5.0** | Multi-stage workflow: DRT Analysis → Gaussian Deconvolution → Area Distribution Analysis | Updated with proper inductance handling (DRTtools approach)")


if __name__ == "__main__":
    main()
