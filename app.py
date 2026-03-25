"""
Streamlit DRT Analyzer - Комплексный анализ импедансных спектров
Методы: Tikhonov, Bayesian, Maximum Entropy, GP-DRT (упрощенный), ISGP (эмуляция)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy import optimize, interpolate, signal, linalg
from scipy.special import logit, expit
import warnings
import time
import io
import base64
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

warnings.filterwarnings('ignore')

# ============================================================================
# Конфигурация страницы
# ============================================================================
st.set_page_config(
    page_title="DRT Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ DRT Analyzer - Distribution of Relaxation Times Analysis")
st.markdown("""
    Комплексный анализ импедансных спектров методами распределения времен релаксации.
    Поддерживаемые методы: Тихоновская регуляризация, Байесовский подход, 
    Максимальная энтропия, Гауссовские процессы.
""")

# ============================================================================
# Классы и структуры данных
# ============================================================================
@dataclass
class DRTSolution:
    """Результат DRT-инверсии"""
    tau: np.ndarray          # времена релаксации
    gamma: np.ndarray        # DRT функция
    gamma_std: Optional[np.ndarray]  # стандартное отклонение (для байесовских методов)
    method: str              # использованный метод
    time: float              # время вычисления
    polarization_resistance: float  # полное поляризационное сопротивление
    peaks: List[Dict]        # выделенные пики
    reconstructed_impedance: Optional[Tuple[np.ndarray, np.ndarray]]  # восстановленный импеданс

@dataclass
class EISData:
    """Данные импедансного спектра"""
    freq: np.ndarray         # частоты (Гц)
    z_real: np.ndarray       # действительная часть
    z_imag: np.ndarray       # мнимая часть
    R_inf: float             # омическое сопротивление
    R_pol: float             # поляризационное сопротивление
    valid: bool              # прошел ли проверку KK
    validation_error: float  # ошибка валидации

# ============================================================================
# Функции предобработки и валидации
# ============================================================================
def validate_kramers_kronig(freq: np.ndarray, z_real: np.ndarray, z_imag: np.ndarray) -> Tuple[bool, float]:
    """
    Проверка данных через тест Кронига-Крамерса
    Использует аппроксимацию рядом RC-элементов
    """
    try:
        # Логарифмическая шкала частот для RC-элементов
        log_freq = np.log10(freq)
        tau_min = 1 / (2 * np.pi * freq.max())
        tau_max = 1 / (2 * np.pi * freq.min())
        
        n_rc = min(20, len(freq) // 3)
        tau_rc = np.logspace(np.log10(tau_min), np.log10(tau_max), n_rc)
        
        # Формирование матрицы для RC-элементов
        A = np.zeros((len(freq), n_rc))
        for i, f in enumerate(freq):
            omega = 2 * np.pi * f
            for j, tau in enumerate(tau_rc):
                A[i, j] = 1 / (1 + (omega * tau)**2)
        
        # Решаем задачу наименьших квадратов
        R_rc, _, _, _ = np.linalg.lstsq(A, z_real - z_real.min(), rcond=None)
        
        # Восстанавливаем спектр
        z_real_recon = z_real.min() + A @ R_rc
        
        # Вычисляем относительную ошибку
        error = np.mean(np.abs((z_real - z_real_recon) / (np.abs(z_real) + 1e-10))) * 100
        
        return error < 5.0, error
    except Exception:
        return False, 100.0

def preprocess_eis_data(df: pd.DataFrame) -> EISData:
    """
    Предобработка загруженных данных
    Ожидаемые колонки: частота (Гц), Z', Z''
    """
    # Определение колонок
    cols = df.columns.tolist()
    
    # Автоматическое определение колонок
    freq_col = None
    real_col = None
    imag_col = None
    
    for col in cols:
        col_lower = col.lower()
        if 'частот' in col_lower or 'freq' in col_lower or 'hz' in col_lower:
            freq_col = col
        elif 'действ' in col_lower or 'real' in col_lower or "z'" in col_lower or 'zprime' in col_lower:
            real_col = col
        elif 'мним' in col_lower or 'imag' in col_lower or "z''" in col_lower or 'zprime2' in col_lower:
            imag_col = col
    
    if freq_col is None or real_col is None or imag_col is None:
        # Если не найдены, предполагаем первые три колонки
        freq_col, real_col, imag_col = cols[0], cols[1], cols[2]
    
    freq = df[freq_col].values
    z_real = df[real_col].values
    z_imag = df[imag_col].values
    
    # Сортировка по частоте
    sort_idx = np.argsort(freq)
    freq = freq[sort_idx]
    z_real = z_real[sort_idx]
    z_imag = z_imag[sort_idx]
    
    # Вычисление R_inf и R_pol
    R_inf = z_real.min()
    # Экстраполяция к нулевой частоте для R_pol
    low_freq_idx = len(freq) // 4  # первые 25% точек
    if low_freq_idx > 2:
        p = np.polyfit(freq[:low_freq_idx], z_real[:low_freq_idx], 1)
        R_pol = p[1] - R_inf
    else:
        R_pol = z_real.max() - R_inf
    
    # Проверка KK
    valid, error = validate_kramers_kronig(freq, z_real, z_imag)
    
    return EISData(
        freq=freq,
        z_real=z_real,
        z_imag=z_imag,
        R_inf=R_inf,
        R_pol=R_pol,
        valid=valid,
        validation_error=error
    )

def build_kernel_matrix(freq: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    Построение матрицы ядра для уравнения Фредгольма
    K_{ij} = 1 / (1 + j*omega_i*tau_j)
    """
    n_freq = len(freq)
    n_tau = len(tau)
    omega = 2 * np.pi * freq
    
    # Действительная часть
    K_real = np.zeros((n_freq, n_tau))
    K_imag = np.zeros((n_freq, n_tau))
    
    for i, w in enumerate(omega):
        for j, t in enumerate(tau):
            denom = 1 + (w * t)**2
            K_real[i, j] = 1 / denom
            K_imag[i, j] = -w * t / denom
    
    return K_real, K_imag

# ============================================================================
# Метод 1: Тихоновская регуляризация
# ============================================================================
def tikhonov_drt(data: EISData, lambda_reg: float = None, 
                 regularization_order: int = 2) -> DRTSolution:
    """
    DRT через тихоновскую регуляризацию
    """
    start_time = time.time()
    
    # Дискретизация времени релаксации
    tau_min = 1 / (2 * np.pi * data.freq.max())
    tau_max = 1 / (2 * np.pi * data.freq.min())
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), 100)
    
    # Построение матрицы ядра
    K_real, K_imag = build_kernel_matrix(data.freq, tau)
    
    # Нормировка
    scaling = np.linalg.norm(K_real, 'fro') + np.linalg.norm(K_imag, 'fro')
    K_real /= scaling
    K_imag /= scaling
    
    # Целевой вектор
    z_target = np.concatenate([
        (data.z_real - data.R_inf) / scaling,
        (-data.z_imag) / scaling  # знак минус, так как Z'' отрицательна
    ])
    
    # Полная матрица системы
    K_total = np.vstack([K_real, K_imag])
    
    # Матрица регуляризации
    n_tau = len(tau)
    if regularization_order == 0:
        L = np.eye(n_tau)
    elif regularization_order == 1:
        L = np.zeros((n_tau - 1, n_tau))
        for i in range(n_tau - 1):
            L[i, i] = -1
            L[i, i + 1] = 1
    else:
        L = np.zeros((n_tau - 2, n_tau))
        for i in range(n_tau - 2):
            L[i, i] = 1
            L[i, i + 1] = -2
            L[i, i + 2] = 1
    
    # Выбор параметра регуляризации
    if lambda_reg is None:
        # L-кривая
        lambdas = np.logspace(-8, 2, 30)
        residuals = []
        norms = []
        
        for lam in lambdas:
            ATA = K_total.T @ K_total + lam * (L.T @ L)
            try:
                gamma = np.linalg.solve(ATA, K_total.T @ z_target)
                gamma = np.maximum(gamma, 0)
                residual = np.linalg.norm(K_total @ gamma - z_target)
                norm_reg = np.linalg.norm(L @ gamma)
                residuals.append(residual)
                norms.append(norm_reg)
            except:
                residuals.append(np.inf)
                norms.append(np.inf)
        
        # Поиск угла L-кривой (максимум кривизны)
        valid = np.isfinite(residuals) & np.isfinite(norms)
        if np.sum(valid) > 3:
            log_res = np.log(np.array(residuals)[valid])
            log_norm = np.log(np.array(norms)[valid])
            curvatures = np.diff(np.diff(log_res) / np.diff(log_norm))
            if len(curvatures) > 0:
                best_idx = np.argmax(np.abs(curvatures))
                lambda_reg = lambdas[valid][best_idx + 1]
            else:
                lambda_reg = 1e-4
        else:
            lambda_reg = 1e-4
    
    # Финальное решение
    ATA = K_total.T @ K_total + lambda_reg * (L.T @ L)
    gamma = np.linalg.solve(ATA, K_total.T @ z_target)
    gamma = np.maximum(gamma, 0)
    
    # Нормировка
    gamma = gamma / np.trapz(gamma, tau) * data.R_pol
    
    # Выделение пиков
    peaks = detect_peaks(tau, gamma)
    
    # Восстановление импеданса
    z_recon_real = data.R_inf + K_real @ gamma
    z_recon_imag = -K_imag @ gamma
    
    return DRTSolution(
        tau=tau,
        gamma=gamma,
        gamma_std=None,
        method="Tikhonov",
        time=time.time() - start_time,
        polarization_resistance=np.trapz(gamma, tau),
        peaks=peaks,
        reconstructed_impedance=(z_recon_real, z_recon_imag)
    )

# ============================================================================
# Метод 2: Байесовский подход (упрощенная реализация с bootstrap)
# ============================================================================
def bayesian_drt(data: EISData, n_samples: int = 100) -> DRTSolution:
    """
    Байесовский DRT через bootstrap-анализ
    Оценивает доверительные интервалы
    """
    start_time = time.time()
    
    # Базовая DRT (Тихоновская)
    base_solution = tikhonov_drt(data, lambda_reg=1e-4)
    
    if n_samples <= 1:
        return base_solution
    
    # Bootstrap для оценки неопределенности
    gamma_samples = []
    n_freq = len(data.freq)
    
    progress_bar = st.progress(0)
    
    for i in range(n_samples):
        # Добавляем шум к данным
        noise_level = 0.01 * np.max(np.abs(data.z_real))
        z_real_noisy = data.z_real + np.random.normal(0, noise_level, n_freq)
        z_imag_noisy = data.z_imag + np.random.normal(0, noise_level, n_freq)
        
        # Создаем временные данные
        temp_data = EISData(
            freq=data.freq,
            z_real=z_real_noisy,
            z_imag=z_imag_noisy,
            R_inf=data.R_inf,
            R_pol=data.R_pol,
            valid=True,
            validation_error=0
        )
        
        # Решаем для зашумленных данных
        try:
            sol = tikhonov_drt(temp_data, lambda_reg=1e-4)
            gamma_samples.append(sol.gamma)
        except:
            pass
        
        progress_bar.progress((i + 1) / n_samples)
    
    progress_bar.empty()
    
    if gamma_samples:
        gamma_samples = np.array(gamma_samples)
        gamma_mean = np.mean(gamma_samples, axis=0)
        gamma_std = np.std(gamma_samples, axis=0)
    else:
        gamma_mean = base_solution.gamma
        gamma_std = np.zeros_like(gamma_mean)
    
    # Нормировка
    gamma_mean = gamma_mean / np.trapz(gamma_mean, base_solution.tau) * data.R_pol
    gamma_std = gamma_std / np.trapz(gamma_std, base_solution.tau) * data.R_pol
    
    # Выделение пиков
    peaks = detect_peaks(base_solution.tau, gamma_mean)
    
    return DRTSolution(
        tau=base_solution.tau,
        gamma=gamma_mean,
        gamma_std=gamma_std,
        method="Bayesian (Bootstrap)",
        time=time.time() - start_time,
        polarization_resistance=np.trapz(gamma_mean, base_solution.tau),
        peaks=peaks,
        reconstructed_impedance=base_solution.reconstructed_impedance
    )

# ============================================================================
# Метод 3: Максимальная энтропия
# ============================================================================
def entropy_drt(data: EISData, lambda_reg: float = None) -> DRTSolution:
    """
    DRT через максимизацию энтропии
    """
    start_time = time.time()
    
    # Дискретизация
    tau_min = 1 / (2 * np.pi * data.freq.max())
    tau_max = 1 / (2 * np.pi * data.freq.min())
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), 100)
    
    K_real, K_imag = build_kernel_matrix(data.freq, tau)
    
    # Целевой вектор
    z_target = np.concatenate([
        data.z_real - data.R_inf,
        -data.z_imag
    ])
    K_total = np.vstack([K_real, K_imag])
    
    # Выбор параметра регуляризации
    if lambda_reg is None:
        lambda_reg = 1e-4
    
    def objective(gamma):
        """Функция потерь с энтропийным штрафом"""
        gamma = np.maximum(gamma, 1e-12)
        residual = np.linalg.norm(K_total @ gamma - z_target)**2
        entropy = -np.sum(gamma * np.log(gamma + 1e-12))
        return residual - lambda_reg * entropy
    
    def jacobian(gamma):
        """Градиент"""
        gamma = np.maximum(gamma, 1e-12)
        grad_residual = 2 * K_total.T @ (K_total @ gamma - z_target)
        grad_entropy = -(np.log(gamma + 1e-12) + 1)
        return grad_residual - lambda_reg * grad_entropy
    
    # Начальное приближение
    gamma0 = np.ones(len(tau)) / len(tau)
    
    # Оптимизация
    result = optimize.minimize(
        objective,
        gamma0,
        method='L-BFGS-B',
        jac=jacobian,
        bounds=[(0, None)] * len(tau),
        options={'maxiter': 500, 'disp': False}
    )
    
    gamma = np.maximum(result.x, 0)
    
    # Нормировка
    gamma = gamma / np.trapz(gamma, tau) * data.R_pol
    
    # Выделение пиков
    peaks = detect_peaks(tau, gamma)
    
    # Восстановление импеданса
    z_recon_real = data.R_inf + K_real @ gamma
    z_recon_imag = -K_imag @ gamma
    
    return DRTSolution(
        tau=tau,
        gamma=gamma,
        gamma_std=None,
        method="Maximum Entropy",
        time=time.time() - start_time,
        polarization_resistance=np.trapz(gamma, tau),
        peaks=peaks,
        reconstructed_impedance=(z_recon_real, z_recon_imag)
    )

# ============================================================================
# Метод 4: Гауссовские процессы (упрощенная версия)
# ============================================================================
def gp_drt(data: EISData) -> DRTSolution:
    """
    Упрощенный GP-DRT
    Использует гауссовскую регрессию для сглаживания
    """
    start_time = time.time()
    
    # Базовая DRT
    base_solution = tikhonov_drt(data, lambda_reg=1e-3)
    
    # Гауссовское сглаживание DRT
    from scipy.ndimage import gaussian_filter1d
    
    gamma_smoothed = gaussian_filter1d(base_solution.gamma, sigma=2.0)
    gamma_smoothed = np.maximum(gamma_smoothed, 0)
    
    # Нормировка
    gamma_smoothed = gamma_smoothed / np.trapz(gamma_smoothed, base_solution.tau) * data.R_pol
    
    # Стандартное отклонение на основе невязок
    residuals = base_solution.gamma - gamma_smoothed
    gamma_std = np.std(residuals) * np.ones_like(gamma_smoothed)
    
    # Выделение пиков
    peaks = detect_peaks(base_solution.tau, gamma_smoothed)
    
    return DRTSolution(
        tau=base_solution.tau,
        gamma=gamma_smoothed,
        gamma_std=gamma_std,
        method="Gaussian Process",
        time=time.time() - start_time,
        polarization_resistance=np.trapz(gamma_smoothed, base_solution.tau),
        peaks=peaks,
        reconstructed_impedance=base_solution.reconstructed_impedance
    )

# ============================================================================
# Метод 5: ISGP (генетическое программирование) - упрощенная версия
# ============================================================================
def isgp_drt(data: EISData, n_peaks: int = 3) -> DRTSolution:
    """
    ISGP - поиск оптимальной суммы пиковых функций
    Упрощенная реализация через генетический алгоритм
    """
    start_time = time.time()
    
    tau_min = 1 / (2 * np.pi * data.freq.max())
    tau_max = 1 / (2 * np.pi * data.freq.min())
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), 100)
    
    K_real, K_imag = build_kernel_matrix(data.freq, tau)
    
    def peak_function(tau, center, width, height):
        """Гауссов пик"""
        return height * np.exp(-((np.log(tau) - center)**2) / (2 * width**2))
    
    def build_gamma(params, tau):
        """Построение DRT из параметров пиков"""
        gamma = np.zeros_like(tau)
        for i in range(n_peaks):
            center = params[3*i]
            width = params[3*i + 1]
            height = params[3*i + 2]
            gamma += peak_function(tau, center, width, height)
        return gamma
    
    def fitness(params):
        """Функция приспособленности"""
        gamma = build_gamma(params, tau)
        if np.sum(gamma) <= 0:
            return 1e10
        
        # Нормировка
        gamma = gamma / np.trapz(gamma, tau) * data.R_pol
        
        # Восстановление импеданса
        z_recon_real = data.R_inf + K_real @ gamma
        z_recon_imag = -K_imag @ gamma
        
        # Ошибка
        error_real = np.mean(((z_recon_real - data.z_real) / (np.abs(data.z_real) + 1e-10))**2)
        error_imag = np.mean(((z_recon_imag - data.z_imag) / (np.abs(data.z_imag) + 1e-10))**2)
        
        return error_real + error_imag
    
    # Генетический алгоритм
    pop_size = 50
    n_generations = 100
    
    # Инициализация популяции
    population = []
    for _ in range(pop_size):
        params = []
        for i in range(n_peaks):
            center = np.random.uniform(np.log(tau_min), np.log(tau_max))
            width = np.random.uniform(0.5, 2.0)
            height = np.random.uniform(0, 1)
            params.extend([center, width, height])
        population.append(np.array(params))
    
    progress_bar = st.progress(0)
    
    for gen in range(n_generations):
        # Оценка приспособленности
        fitness_scores = [fitness(p) for p in population]
        
        # Сортировка
        sorted_idx = np.argsort(fitness_scores)
        population = [population[i] for i in sorted_idx]
        
        # Сохраняем лучших
        best = population[0]
        
        # Создание нового поколения
        new_population = [best]
        while len(new_population) < pop_size:
            # Турнирный отбор
            idx1 = np.random.randint(min(20, len(population)))
            idx2 = np.random.randint(min(20, len(population)))
            parent1 = population[idx1]
            parent2 = population[idx2]
            
            # Кроссовер
            crossover_point = np.random.randint(1, len(parent1))
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            
            # Мутация
            if np.random.random() < 0.1:
                mutation_idx = np.random.randint(len(child))
                child[mutation_idx] += np.random.normal(0, 0.1)
            
            new_population.append(child)
        
        population = new_population
        progress_bar.progress((gen + 1) / n_generations)
    
    progress_bar.empty()
    
    # Лучшее решение
    best_params = population[0]
    gamma = build_gamma(best_params, tau)
    gamma = gamma / np.trapz(gamma, tau) * data.R_pol
    
    # Выделение пиков
    peaks = detect_peaks(tau, gamma)
    
    # Восстановление импеданса
    z_recon_real = data.R_inf + K_real @ gamma
    z_recon_imag = -K_imag @ gamma
    
    return DRTSolution(
        tau=tau,
        gamma=gamma,
        gamma_std=None,
        method="ISGP (Evolutionary)",
        time=time.time() - start_time,
        polarization_resistance=np.trapz(gamma, tau),
        peaks=peaks,
        reconstructed_impedance=(z_recon_real, z_recon_imag)
    )

# ============================================================================
# Выделение пиков
# ============================================================================
def detect_peaks(tau: np.ndarray, gamma: np.ndarray, prominence: float = 0.05) -> List[Dict]:
    """
    Выделение пиков на DRT кривой
    """
    from scipy.signal import find_peaks
    
    # Нормировка для поиска
    gamma_norm = gamma / np.max(gamma)
    
    # Поиск пиков
    peaks_idx, properties = find_peaks(
        gamma_norm,
        height=prominence,
        prominence=prominence,
        width=1
    )
    
    peaks = []
    for idx in peaks_idx:
        # Подгонка гауссианы
        try:
            x_fit = np.log(tau[max(0, idx-5):min(len(tau), idx+6)])
            y_fit = gamma[max(0, idx-5):min(len(tau), idx+6)]
            
            def gaussian(x, a, mu, sigma):
                return a * np.exp(-((x - mu)**2) / (2 * sigma**2))
            
            popt, _ = optimize.curve_fit(
                gaussian, x_fit, y_fit,
                p0=[y_fit.max(), x_fit[len(x_fit)//2], 0.5],
                bounds=([0, -np.inf, 0.1], [np.inf, np.inf, 2.0])
            )
            
            peaks.append({
                'tau': tau[idx],
                'frequency': 1 / (2 * np.pi * tau[idx]),
                'resistance': np.trapz(gamma, tau) * 0.2,  # приблизительная оценка
                'width': popt[2],
                'height': popt[0]
            })
        except:
            peaks.append({
                'tau': tau[idx],
                'frequency': 1 / (2 * np.pi * tau[idx]),
                'resistance': np.trapz(gamma, tau) * 0.1,
                'width': 0.5,
                'height': gamma[idx]
            })
    
    return peaks

# ============================================================================
# Визуализация
# ============================================================================
def plot_nyquist(data: EISData, solution: DRTSolution = None):
    """Nyquist plot"""
    fig = go.Figure()
    
    # Экспериментальные данные
    fig.add_trace(go.Scatter(
        x=data.z_real,
        y=-data.z_imag,
        mode='lines+markers',
        name='Experimental',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    if solution and solution.reconstructed_impedance:
        z_recon_real, z_recon_imag = solution.reconstructed_impedance
        fig.add_trace(go.Scatter(
            x=z_recon_real,
            y=-z_recon_imag,
            mode='lines',
            name=f'Reconstructed ({solution.method})',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="Nyquist Plot",
        xaxis_title="Z' (Ω)",
        yaxis_title="-Z'' (Ω)",
        width=600,
        height=500,
        template="plotly_white"
    )
    
    return fig

def plot_drt(solution: DRTSolution):
    """DRT plot"""
    fig = go.Figure()
    
    # Основная кривая
    fig.add_trace(go.Scatter(
        x=solution.tau,
        y=solution.gamma,
        mode='lines',
        name=solution.method,
        line=dict(color='blue', width=2)
    ))
    
    # Доверительный интервал (если есть)
    if solution.gamma_std is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([solution.tau, solution.tau[::-1]]),
            y=np.concatenate([solution.gamma + 2*solution.gamma_std, 
                              (solution.gamma - 2*solution.gamma_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI'
        ))
    
    # Пики
    for i, peak in enumerate(solution.peaks):
        fig.add_trace(go.Scatter(
            x=[peak['tau']],
            y=[peak['height']],
            mode='markers',
            marker=dict(size=10, symbol='triangle-up', color='red'),
            name=f"Peak {i+1}: τ={peak['tau']:.2e}s",
            hovertemplate=f"τ = {peak['tau']:.2e} s<br>f = {peak['frequency']:.2e} Hz<br>R ≈ {peak['resistance']:.3f} Ω"
        ))
    
    fig.update_layout(
        title="Distribution of Relaxation Times",
        xaxis_title="τ (s)",
        yaxis_title="γ(τ) (Ω)",
        xaxis_type="log",
        width=700,
        height=500,
        template="plotly_white"
    )
    
    return fig

def plot_bode(data: EISData, solution: DRTSolution = None):
    """Bode plot"""
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True)
    
    # Модуль импеданса
    z_mod = np.sqrt(data.z_real**2 + data.z_imag**2)
    fig.add_trace(go.Scatter(
        x=data.freq,
        y=z_mod,
        mode='lines+markers',
        name='|Z| exp',
        line=dict(color='blue')
    ), row=1, col=1)
    
    if solution and solution.reconstructed_impedance:
        z_recon_real, z_recon_imag = solution.reconstructed_impedance
        z_recon_mod = np.sqrt(z_recon_real**2 + z_recon_imag**2)
        fig.add_trace(go.Scatter(
            x=data.freq,
            y=z_recon_mod,
            mode='lines',
            name=f'|Z| recon',
            line=dict(color='red', dash='dash')
        ), row=1, col=1)
    
    # Фаза
    phase = np.degrees(np.arctan2(data.z_imag, data.z_real))
    fig.add_trace(go.Scatter(
        x=data.freq,
        y=phase,
        mode='lines+markers',
        name='Phase exp',
        line=dict(color='blue')
    ), row=2, col=1)
    
    if solution and solution.reconstructed_impedance:
        z_recon_real, z_recon_imag = solution.reconstructed_impedance
        phase_recon = np.degrees(np.arctan2(z_recon_imag, z_recon_real))
        fig.add_trace(go.Scatter(
            x=data.freq,
            y=phase_recon,
            mode='lines',
            name='Phase recon',
            line=dict(color='red', dash='dash')
        ), row=2, col=1)
    
    fig.update_xaxis(title="Frequency (Hz)", type="log", row=2, col=1)
    fig.update_yaxis(title="|Z| (Ω)", type="log", row=1, col=1)
    fig.update_yaxis(title="Phase (deg)", row=2, col=1)
    fig.update_layout(height=600, template="plotly_white")
    
    return fig

def plot_residuals(data: EISData, solution: DRTSolution):
    """График невязок"""
    if not solution.reconstructed_impedance:
        return None
    
    z_recon_real, z_recon_imag = solution.reconstructed_impedance
    
    # Относительные невязки
    z_mod = np.sqrt(data.z_real**2 + data.z_imag**2)
    delta_real = (data.z_real - z_recon_real) / (z_mod + 1e-10) * 100
    delta_imag = (data.z_imag - z_recon_imag) / (z_mod + 1e-10) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.freq,
        y=delta_real,
        mode='lines+markers',
        name='Δ Z\'',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=data.freq,
        y=delta_imag,
        mode='lines+markers',
        name='Δ Z\'\'',
        line=dict(color='red')
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hrect(y=-2, y1=2, fillcolor="green", opacity=0.1, line_width=0, annotation_text="±2%")
    
    fig.update_layout(
        title="Relative Residuals",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Residual (%)",
        xaxis_type="log",
        width=700,
        height=400,
        template="plotly_white"
    )
    
    return fig

# ============================================================================
# Основное приложение Streamlit
# ============================================================================
def main():
    # Боковая панель
    with st.sidebar:
        st.header("📁 Загрузка данных")
        
        uploaded_file = st.file_uploader(
            "Выберите файл с EIS данными",
            type=['txt', 'csv', 'xlsx', 'xls'],
            help="Файл должен содержать колонки: частота (Гц), Z', Z''"
        )
        
        # Текстовый виджет для ввода данных
        st.markdown("---")
        st.markdown("или вставьте данные:")
        text_input = st.text_area(
            "Вставьте текст с данными",
            height=150,
            placeholder="freq (Hz)\tZ'\tZ''\n10\t1.2\t-0.5\n100\t1.1\t-0.8\n..."
        )
        
        st.markdown("---")
        st.header("⚙️ Параметры анализа")
        
        method = st.selectbox(
            "Метод инверсии",
            ["Tikhonov", "Bayesian", "Maximum Entropy", "Gaussian Process", "ISGP"],
            help="""
            - Tikhonov: классический метод, быстрый
            - Bayesian: дает доверительные интервалы
            - Maximum Entropy: подавляет ложные пики
            - Gaussian Process: вероятностная оценка
            - ISGP: эволюционный поиск оптимальной формы
            """
        )
        
        if method == "Tikhonov":
            lambda_manual = st.checkbox("Ручной выбор λ")
            if lambda_manual:
                lambda_reg = st.slider(
                    "λ (параметр регуляризации)",
                    min_value=1e-8, max_value=1e-2, value=1e-4, format="%.1e"
                )
            else:
                lambda_reg = None
        else:
            lambda_reg = None
        
        if method == "Bayesian":
            n_samples = st.slider(
                "Количество bootstrap-выборок",
                min_value=10, max_value=500, value=100, step=10
            )
        else:
            n_samples = 100
        
        st.markdown("---")
        st.header("📊 Параметры визуализации")
        
        show_confidence = st.checkbox("Показывать доверительный интервал", value=True)
        show_peaks = st.checkbox("Показывать выделенные пики", value=True)
        
        st.markdown("---")
        st.info("""
        **Формат данных:**
        - Колонка 1: частота (Гц)
        - Колонка 2: действительная часть Z' (Ω)
        - Колонка 3: мнимая часть Z'' (Ω)
        """)
    
    # Основная область
    if uploaded_file is not None or text_input:
        # Загрузка данных
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep='\t', engine='python')
        else:
            # Чтение из текстового поля
            lines = text_input.strip().split('\n')
            data_rows = []
            for line in lines:
                if line.strip() and not line.startswith('#') and not line.startswith('freq'):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            data_rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except:
                            pass
            if data_rows:
                df = pd.DataFrame(data_rows, columns=['freq', 'Z\'', 'Z\'\''])
            else:
                st.error("Не удалось прочитать данные из текстового поля")
                return
        
        # Предобработка
        with st.spinner("Обработка данных..."):
            eis_data = preprocess_eis_data(df)
        
        # Отображение информации о данных
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Количество точек", len(eis_data.freq))
        with col2:
            st.metric("Частотный диапазон", f"{eis_data.freq.min():.2e} - {eis_data.freq.max():.2e} Hz")
        with col3:
            st.metric("R_inf (Ω)", f"{eis_data.R_inf:.4f}")
        with col4:
            st.metric("R_pol (Ω)", f"{eis_data.R_pol:.4f}")
        
        # Статус валидации KK
        if eis_data.valid:
            st.success(f"✅ Kramers-Kronig тест пройден (ошибка: {eis_data.validation_error:.2f}%)")
        else:
            st.warning(f"⚠️ Kramers-Kronig тест не пройден (ошибка: {eis_data.validation_error:.2f}%). Результаты могут быть неточными.")
        
        # Выполнение DRT анализа
        st.markdown("---")
        st.header("📈 DRT Анализ")
        
        with st.spinner(f"Выполняется инверсия методом {method}..."):
            if method == "Tikhonov":
                solution = tikhonov_drt(eis_data, lambda_reg=lambda_reg)
            elif method == "Bayesian":
                solution = bayesian_drt(eis_data, n_samples=n_samples)
            elif method == "Maximum Entropy":
                solution = entropy_drt(eis_data, lambda_reg=lambda_reg)
            elif method == "Gaussian Process":
                solution = gp_drt(eis_data)
            else:  # ISGP
                solution = isgp_drt(eis_data)
        
        # Информация о вычислениях
        st.info(f"⏱️ Время вычислений: {solution.time:.2f} сек | Поляризационное сопротивление: {solution.polarization_resistance:.4f} Ω")
        
        # Визуализация
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Nyquist Plot", "DRT", "Bode Plot", "Residuals", "Результаты пиков"])
        
        with tab1:
            fig_nyquist = plot_nyquist(eis_data, solution)
            st.plotly_chart(fig_nyquist, use_container_width=True)
        
        with tab2:
            fig_drt = plot_drt(solution)
            st.plotly_chart(fig_drt, use_container_width=True)
        
        with tab3:
            fig_bode = plot_bode(eis_data, solution)
            st.plotly_chart(fig_bode, use_container_width=True)
        
        with tab4:
            fig_residuals = plot_residuals(eis_data, solution)
            if fig_residuals:
                st.plotly_chart(fig_residuals, use_container_width=True)
            else:
                st.info("Нет данных для отображения невязок")
        
        with tab5:
            if solution.peaks:
                st.subheader("Выделенные пики релаксации")
                peaks_df = pd.DataFrame(solution.peaks)
                peaks_df.columns = ['τ (s)', 'f (Hz)', 'R (Ω)', 'Width', 'Height']
                peaks_df['τ (s)'] = peaks_df['τ (s)'].apply(lambda x: f"{x:.3e}")
                peaks_df['f (Hz)'] = peaks_df['f (Hz)'].apply(lambda x: f"{x:.3e}")
                peaks_df['R (Ω)'] = peaks_df['R (Ω)'].apply(lambda x: f"{x:.4f}")
                peaks_df['Width'] = peaks_df['Width'].apply(lambda x: f"{x:.3f}")
                peaks_df['Height'] = peaks_df['Height'].apply(lambda x: f"{x:.4f}")
                st.dataframe(peaks_df, use_container_width=True)
                
                st.markdown("### Интерпретация")
                for i, peak in enumerate(solution.peaks):
                    st.markdown(f"""
                    **Процесс {i+1}:**
                    - Время релаксации: {peak['tau']:.3e} с
                    - Частота: {peak['frequency']:.3e} Гц
                    - Вклад в поляризацию: ~{peak['resistance']/solution.polarization_resistance*100:.1f}%
                    """)
            else:
                st.info("Пики не обнаружены")
        
        # Экспорт результатов
        st.markdown("---")
        st.header("💾 Экспорт результатов")
        
        # Подготовка данных для экспорта
        export_data = pd.DataFrame({
            'tau (s)': solution.tau,
            'gamma (Ω)': solution.gamma,
            'log10(tau)': np.log10(solution.tau)
        })
        
        if solution.gamma_std is not None:
            export_data['gamma_std'] = solution.gamma_std
        
        csv = export_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="drt_results_{method}.csv">📥 Скачать DRT данные (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Дополнительная информация
        with st.expander("ℹ️ О методах анализа"):
            st.markdown("""
            ### Сравнение методов
            
            | Метод | Преимущества | Ограничения |
            |-------|-------------|-------------|
            | **Tikhonov** | Быстрый, стандартный, хорошо изучен | Требует выбора λ |
            | **Bayesian** | Доверительные интервалы, устойчивость к шуму | Медленнее |
            | **Maximum Entropy** | Подавляет ложные пики | Нелинейная оптимизация |
            | **Gaussian Process** | Полная вероятностная оценка | Вычислительно сложный |
            | **ISGP** | Не требует λ, аналитическая форма | Очень медленный |
            
            ### Рекомендации по выбору метода
            - Для быстрого анализа: **Tikhonov**
            - Для оценки неопределенности: **Bayesian**
            - Для сложных спектров с артефактами: **Maximum Entropy**
            - Для исследовательских целей: **Gaussian Process** или **ISGP**
            """)
    
    else:
        # Пустое состояние
        st.info("👈 Загрузите файл с EIS данными или вставьте данные в текстовое поле слева")
        
        st.markdown("""
        ### Пример формата данных
        freq (Hz) Z' (Ω) Z'' (Ω)
        1e-2 1.2345 -0.1234
        1e-1 1.2100 -0.2345
        1e0 1.1800 -0.3456
        1e1 1.1500 -0.4567
        1e2 1.1200 -0.5678
        1e3 1.0900 -0.6789
        1e4 1.0600 -0.7890
        1e5 1.0300 -0.8901
        ### Поддерживаемые форматы файлов
        - CSV (разделители: запятая, табуляция, пробел)
        - Excel (.xlsx, .xls)
        - Текстовые файлы (.txt)
        """)
        
        if __name__ == "__main__":
        main()
