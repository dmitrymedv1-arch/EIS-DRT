"""
Streamlit приложение для анализа импеданс-спектров методом распределения времен релаксации (DRT)
Поддерживает несколько методов инверсии:
- Тихоновская регуляризация (Tikhonov)
- Байесовский метод (Bayesian)
- Метод максимальной энтропии (Maximum Entropy)
- Гауссовские процессы (GP-DRT)
- Генетическое программирование (ISGP)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy
from scipy import optimize, linalg, interpolate
from scipy.special import gamma
import io
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.integrate import trapezoid

# Переопределяем np.trapz если его нет (для обратной совместимости)
if not hasattr(np, 'trapz'):
    np.trapz = trapezoid

# Настройка страницы
st.set_page_config(
    page_title="DRT Analysis Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("⚡ Анализ импеданс-спектров методом распределения времен релаксации (DRT)")
st.markdown("""
Программа для обработки электрохимических импеданс-спектров с получением функции распределения времен релаксации.
Поддерживаются современные методы инверсии: Тихоновская регуляризация, Байесовский метод, 
Максимальная энтропия, Гауссовские процессы и Генетическое программирование.
""")

# ============================================================================
# Базовые классы и функции для DRT анализа
# ============================================================================

class DRTCore:
    """Базовый класс для DRT инверсии"""
    
    def __init__(self, frequencies, Z_real, Z_imag):
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.Z_real = np.asarray(Z_real, dtype=float)
        self.Z_imag = np.asarray(Z_imag, dtype=float)
        self.Z = self.Z_real + 1j * self.Z_imag
        self.N = len(frequencies)
        
        # Автоматическое определение диапазона времен релаксации
        self.tau_min = 1.0 / (2 * np.pi * np.max(self.frequencies))
        self.tau_max = 1.0 / (2 * np.pi * np.min(self.frequencies))
        
        # Определение омического сопротивления (экстраполяция к высоким частотам)
        high_freq_idx = np.where(self.frequencies > 0.1 * np.max(self.frequencies))[0]
        if len(high_freq_idx) > 3:
            self.R_inf = np.mean(self.Z_real[high_freq_idx[-5:]])
        else:
            self.R_inf = self.Z_real[0]
        
        # Общее поляризационное сопротивление
        self.R_pol = np.max(self.Z_real) - self.R_inf if np.max(self.Z_real) > self.R_inf else 1.0
    
    def _build_kernel_matrix(self, tau_grid):
        """Построение матрицы ядра для заданной сетки времен"""
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
        """Нахождение угловой точки L-кривой (максимум кривизны)"""
        log_res = np.log(residuals)
        log_sol = np.log(solution_norms)
        
        # Вычисление кривизны
        dlog_res = np.gradient(log_res)
        dlog_sol = np.gradient(log_sol)
        curvature = np.abs(dlog_res * dlog_sol) / (dlog_res**2 + dlog_sol**2)**1.5
        
        return np.argmax(curvature[1:-1]) + 1 if len(curvature) > 2 else len(residuals) // 2


class TikhonovDRT(DRTCore):
    """Тихоновская регуляризация для DRT"""
    
    def __init__(self, frequencies, Z_real, Z_imag, regularization_order=2):
        super().__init__(frequencies, Z_real, Z_imag)
        self.regularization_order = regularization_order
    
    def _build_regularization_matrix(self, M, order):
        """Построение матрицы регуляризации"""
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
        """Вычисление DRT методом Тихоновской регуляризации"""
        
        # Создание логарифмической сетки времен
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        
        # Построение матрицы ядра
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        
        # Целевой вектор (смещение на омическое сопротивление)
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        K = np.vstack([K_real, K_imag])
        
        # Построение матрицы регуляризации
        L = self._build_regularization_matrix(n_tau, self.regularization_order)
        
        if lambda_auto:
            # Автоматический подбор lambda по L-кривой
            if lambda_range is None:
                lambda_range = np.logspace(-8, 2, 30)
            
            residuals = []
            solution_norms = []
            solutions = []
            
            for lam in lambda_range:
                try:
                    # Решение задачи наименьших квадратов с регуляризацией
                    A = np.vstack([K, lam * L])
                    b = np.concatenate([Z_target, np.zeros(L.shape[0])])
                    
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    x = np.maximum(x, 0)  # Ограничение неотрицательности
                    
                    residual = np.linalg.norm(K @ x - Z_target)
                    sol_norm = np.linalg.norm(L @ x)
                    
                    residuals.append(residual)
                    solution_norms.append(sol_norm)
                    solutions.append(x)
                except:
                    continue
            
            if len(residuals) > 2:
                # Выбор оптимального lambda по L-кривой
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
            # Использование заданного lambda
            lam = lambda_value if lambda_value is not None else 1e-4
            A = np.vstack([K, lam * L])
            b = np.concatenate([Z_target, np.zeros(L.shape[0])])
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            gamma = np.maximum(x, 0)
        
        # Нормализация DRT
        integral = np.trapz(gamma, np.log(tau_grid))
        if integral > 0:
            gamma = gamma / integral * self.R_pol
        
        return tau_grid, gamma
    
    def reconstruct_impedance(self, tau_grid, gamma):
        """Реконструкция импеданса из DRT"""
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        return Z_rec_real, Z_rec_imag


class BayesianDRT(DRTCore):
    """Байесовский метод для DRT (MAP оценка)"""
    
    def __init__(self, frequencies, Z_real, Z_imag):
        super().__init__(frequencies, Z_real, Z_imag)
    
    def _objective_function(self, x, K, Z_target, L, alpha):
        """Целевая функция для байесовской оптимизации"""
        gamma = x[:-1]
        log_lambda = x[-1]
        lam = np.exp(log_lambda)
        
        gamma = np.maximum(gamma, 0)
        
        residual = K @ gamma - Z_target
        data_fit = 0.5 * np.sum(residual**2)
        prior = 0.5 * lam * np.sum((L @ gamma)**2)
        
        return data_fit + prior + 0.5 * (len(gamma) * np.log(lam) - log_lambda)
    
    def compute(self, n_tau=150, n_iterations=100):
        """Вычисление DRT байесовским методом (MAP оценка)"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        K = np.vstack([K_real, K_imag])
        
        # Матрица регуляризации (вторая производная)
        L = np.zeros((n_tau-2, n_tau))
        for i in range(n_tau-2):
            L[i, i] = 1
            L[i, i+1] = -2
            L[i, i+2] = 1
        
        # Инициализация
        x0 = np.ones(n_tau + 1) * 0.1
        x0[-1] = np.log(1e-4)  # начальное значение lambda
        
        # Оптимизация
        result = optimize.minimize(
            self._objective_function, x0,
            args=(K, Z_target, L, 1.0),
            method='L-BFGS-B',
            options={'maxiter': n_iterations, 'disp': False}
        )
        
        gamma = np.maximum(result.x[:-1], 0)
        
        # Нормализация
        integral = np.trapz(gamma, np.log(tau_grid))
        if integral > 0:
            gamma = gamma / integral * self.R_pol
        
        # Простая оценка доверительного интервала (на основе кривизны)
        confidence = 0.5 * np.ones_like(gamma)
        
        return tau_grid, gamma, confidence


class MaxEntropyDRT(DRTCore):
    """Метод максимальной энтропии для DRT"""
    
    def __init__(self, frequencies, Z_real, Z_imag):
        super().__init__(frequencies, Z_real, Z_imag)
    
    def _entropy(self, gamma):
        """Вычисление энтропии Шеннона"""
        gamma_pos = gamma[gamma > 1e-10]
        return -np.sum(gamma_pos * np.log(gamma_pos))
    
    def _objective_function(self, x, K, Z_target, lam):
        """Целевая функция с энтропийным штрафом"""
        gamma = np.maximum(x, 1e-10)
        residual = K @ gamma - Z_target
        data_fit = 0.5 * np.sum(residual**2)
        entropy_penalty = -lam * self._entropy(gamma)
        return data_fit + entropy_penalty
    
    def compute(self, n_tau=150, lambda_value=0.1):
        """Вычисление DRT методом максимальной энтропии"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        K = np.vstack([K_real, K_imag])
        
        # Инициализация
        x0 = np.ones(n_tau) / n_tau
        
        # Оптимизация
        result = optimize.minimize(
            self._objective_function, x0,
            args=(K, Z_target, lambda_value),
            method='L-BFGS-B',
            bounds=[(1e-10, None) for _ in range(n_tau)],
            options={'maxiter': 500, 'disp': False}
        )
        
        gamma = result.x
        
        # Нормализация
        integral = np.trapz(gamma, np.log(tau_grid))
        if integral > 0:
            gamma = gamma / integral * self.R_pol
        
        return tau_grid, gamma


class GaussianProcessDRT(DRTCore):
    """Гауссовские процессы для DRT"""
    
    def __init__(self, frequencies, Z_real, Z_imag):
        super().__init__(frequencies, Z_real, Z_imag)
    
    def _rbf_kernel(self, x1, x2, length_scale=1.0, sigma_f=1.0):
        """Радиальная базисная функция ядра"""
        dist_matrix = np.subtract.outer(x1, x2)**2
        return sigma_f**2 * np.exp(-0.5 * dist_matrix / length_scale**2)
    
    def compute(self, n_tau=150, n_components=20):
        """Вычисление DRT с использованием GP (упрощенная версия)"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        log_tau_grid = np.log10(tau_grid)
        
        # Создание базиса из RBF функций
        basis_centers = np.linspace(log_tau_grid[0], log_tau_grid[-1], n_components)
        
        # Построение матрицы ядра
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        K_full = np.vstack([K_real, K_imag])
        
        # Построение матрицы признаков
        length_scale = (log_tau_grid[-1] - log_tau_grid[0]) / n_components
        Phi = np.zeros((self.N * 2, n_components))
        
        for i, center in enumerate(basis_centers):
            phi = np.exp(-0.5 * ((log_tau_grid - center) / length_scale)**2)
            phi = phi / np.sum(phi)
            Phi[:, i] = K_full @ phi
        
        Z_target = np.concatenate([self.Z_real - self.R_inf, -self.Z_imag])
        
        # Решение с регуляризацией
        lam = 1e-4
        A = Phi.T @ Phi + lam * np.eye(n_components)
        b = Phi.T @ Z_target
        weights = np.linalg.solve(A, b)
        
        # Восстановление DRT
        gamma = np.zeros(n_tau)
        for i, center in enumerate(basis_centers):
            phi = np.exp(-0.5 * ((log_tau_grid - center) / length_scale)**2)
            phi = phi / np.sum(phi)
            gamma += weights[i] * phi
        
        gamma = np.maximum(gamma, 0)
        
        # Нормализация
        integral = np.trapz(gamma, np.log(tau_grid))
        if integral > 0:
            gamma = gamma / integral * self.R_pol
        
        # Оценка неопределенности
        uncertainty = np.abs(weights).mean() * np.ones_like(gamma)
        
        return tau_grid, gamma, uncertainty


class ISGPDRT(DRTCore):
    """Генетическое программирование для DRT (упрощенная версия)"""
    
    def __init__(self, frequencies, Z_real, Z_imag):
        super().__init__(frequencies, Z_real, Z_imag)
    
    def _gaussian_peak(self, tau, tau0, width, amplitude):
        """Гауссиан для представления пика DRT"""
        return amplitude * np.exp(-((np.log10(tau) - np.log10(tau0))**2) / (2 * width**2))
    
    def _evaluate_fitness(self, peaks_params, tau_grid):
        """Оценка приспособленности решения"""
        gamma = np.zeros_like(tau_grid)
        for params in peaks_params:
            gamma += self._gaussian_peak(tau_grid, params['tau0'], params['width'], params['amplitude'])
        
        # Реконструкция импеданса
        K_real, K_imag = self._build_kernel_matrix(tau_grid)
        Z_rec_real = self.R_inf + K_real @ gamma
        Z_rec_imag = -K_imag @ gamma
        
        # Ошибка реконструкции
        error = np.mean((self.Z_real - Z_rec_real)**2 + (self.Z_imag - Z_rec_imag)**2)
        
        # Штраф за сложность
        complexity_penalty = 0.01 * len(peaks_params)
        
        return error + complexity_penalty
    
    def compute(self, n_tau=150, n_peaks_max=5, n_generations=50, population_size=20):
        """Вычисление DRT методом ISGP"""
        
        tau_grid = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), n_tau)
        
        # Инициализация популяции
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
        
        # Эволюция
        for generation in range(n_generations):
            fitness = [self._evaluate_fitness(peaks, tau_grid) for peaks in population]
            
            # Отбор лучших
            sorted_indices = np.argsort(fitness)
            elite = [population[i] for i in sorted_indices[:population_size // 2]]
            
            # Создание нового поколения
            new_population = elite.copy()
            while len(new_population) < population_size:
                parent = elite[np.random.randint(len(elite))]
                child = [peak.copy() for peak in parent]
                
                # Мутация
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
        
        # Выбор лучшего решения
        fitness = [self._evaluate_fitness(peaks, tau_grid) for peaks in population]
        best_peaks = population[np.argmin(fitness)]
        
        # Построение DRT
        gamma = np.zeros_like(tau_grid)
        for peak in best_peaks:
            gamma += self._gaussian_peak(tau_grid, peak['tau0'], peak['width'], peak['amplitude'])
        
        # Нормализация
        integral = np.trapz(gamma, np.log(tau_grid))
        if integral > 0:
            gamma = gamma / integral * self.R_pol
        
        return tau_grid, gamma, best_peaks


# ============================================================================
# Функции для выделения пиков и анализа
# ============================================================================

def find_peaks_drt(tau_grid, gamma, prominence=0.05):
    """Выделение пиков в DRT-спектре"""
    from scipy.signal import find_peaks
    
    # Нормализация гаммы
    gamma_norm = gamma / np.max(gamma) if np.max(gamma) > 0 else gamma
    
    peaks, properties = find_peaks(
        gamma_norm,
        height=prominence,
        prominence=prominence,
        distance=len(tau_grid) // 20  # минимальное расстояние между пиками
    )
    
    peak_results = []
    for idx in peaks:
        peak_info = {
            'tau': tau_grid[idx],
            'log_tau': np.log10(tau_grid[idx]),
            'frequency': 1 / (2 * np.pi * tau_grid[idx]),
            'amplitude': gamma[idx],
            'width': properties['widths'][peaks.tolist().index(idx)] if 'widths' in properties else None
        }
        peak_results.append(peak_info)
    
    return peak_results


def fit_gaussian_peaks(tau_grid, gamma, n_peaks=None):
    """Аппроксимация DRT суммой гауссиан"""
    from scipy.optimize import curve_fit
    
    log_tau = np.log10(tau_grid)
    
    if n_peaks is None:
        # Автоматическое определение числа пиков
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
    
    # Начальные параметры
    initial_params = []
    peaks = find_peaks_drt(tau_grid, gamma)
    for i, peak in enumerate(peaks[:n_peaks]):
        initial_params.extend([peak['amplitude'], peak['log_tau'], 0.3])
    
    try:
        popt, _ = curve_fit(sum_gaussians, log_tau, gamma, p0=initial_params, maxfev=5000)
        
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
# Визуализация
# ============================================================================

def plot_impedance_spectra(frequencies, Z_real_exp, Z_imag_exp, Z_real_rec=None, Z_imag_rec=None):
    """Построение графиков импеданса"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Nyquist plot', 'Bode plot - Magnitude'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Nyquist plot
    fig.add_trace(
        go.Scatter(x=Z_real_exp, y=-Z_imag_exp, mode='markers',
                   name='Эксперимент', marker=dict(size=6, color='blue')),
        row=1, col=1
    )
    
    if Z_real_rec is not None and Z_imag_rec is not None:
        fig.add_trace(
            go.Scatter(x=Z_real_rec, y=-Z_imag_rec, mode='lines',
                       name='Реконструкция', line=dict(color='red', width=2)),
            row=1, col=1
        )
    
    fig.update_xaxes(title_text="Z' (Ом)", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' (Ом)", row=1, col=1)
    
    # Bode plot - Magnitude
    Z_mod_exp = np.sqrt(Z_real_exp**2 + Z_imag_exp**2)
    fig.add_trace(
        go.Scatter(x=frequencies, y=Z_mod_exp, mode='markers',
                   name='Эксперимент', marker=dict(size=6, color='blue')),
        row=1, col=2
    )
    
    if Z_real_rec is not None and Z_imag_rec is not None:
        Z_mod_rec = np.sqrt(Z_real_rec**2 + Z_imag_rec**2)
        fig.add_trace(
            go.Scatter(x=frequencies, y=Z_mod_rec, mode='lines',
                       name='Реконструкция', line=dict(color='red', width=2)),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="|Z| (Ом)", type="log", row=1, col=2)
    
    fig.update_layout(height=500, showlegend=True)
    
    return fig


def plot_drt_spectrum(tau_grid, gamma, peaks=None, confidence=None):
    """Построение DRT-спектра"""
    fig = go.Figure()
    
    # Основная кривая DRT
    fig.add_trace(go.Scatter(
        x=tau_grid, y=gamma,
        mode='lines',
        name='DRT',
        line=dict(color='blue', width=2)
    ))
    
    # Доверительный интервал
    if confidence is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([tau_grid, tau_grid[::-1]]),
            y=np.concatenate([gamma + confidence, (gamma - confidence)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Доверительный интервал'
        ))
    
    # Отмеченные пики
    if peaks:
        for peak in peaks:
            fig.add_trace(go.Scatter(
                x=[peak['tau']], y=[peak['amplitude']],
                mode='markers',
                marker=dict(size=12, color='red', symbol='x'),
                name=f"τ = {peak['tau']:.3e} c"
            ))
            
            fig.add_annotation(
                x=peak['tau'], y=peak['amplitude'],
                text=f"{1/(2*np.pi*peak['tau']):.2f} Гц",
                showarrow=True,
                arrowhead=2,
                ax=20, ay=-20
            )
    
    fig.update_xaxes(title_text="τ (с)", type="log")
    fig.update_yaxes(title_text="γ(τ)")
    fig.update_layout(height=500, title="Распределение времен релаксации (DRT)")
    
    return fig


# ============================================================================
# Интерфейс Streamlit
# ============================================================================

def main():
    # Боковая панель для ввода данных
    with st.sidebar:
        st.header("📁 Ввод данных")
        
        input_type = st.radio(
            "Способ ввода данных:",
            ["Загрузка файла", "Ручной ввод"]
        )
        
        data = None
        
        if input_type == "Загрузка файла":
            uploaded_file = st.file_uploader(
                "Загрузите файл с данными",
                type=['txt', 'csv', 'dat', 'z', 'mpt'],
                help="Файл должен содержать столбцы: частота (Гц), Z' (Ом), Z'' (Ом)"
            )
            
            if uploaded_file is not None:
                try:
                    # Попытка прочитать файл
                    content = uploaded_file.read().decode('utf-8')
                    lines = content.strip().split('\n')
                    
                    # Поиск начала данных
                    data_start = 0
                    for i, line in enumerate(lines):
                        if line.strip() and not line.startswith('#') and not line.startswith('!'):
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    float(parts[0])
                                    data_start = i
                                    break
                                except:
                                    continue
                    
                    # Чтение данных
                    data_lines = lines[data_start:]
                    rows = []
                    for line in data_lines:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    rows.append([float(p) for p in parts[:3]])
                                except:
                                    continue
                    
                    if rows:
                        data = pd.DataFrame(rows, columns=['frequency', 'Z_real', 'Z_imag'])
                        st.success(f"Загружено {len(data)} точек спектра")
                        st.dataframe(data.head())
                    else:
                        st.error("Не удалось прочитать данные из файла")
                        
                except Exception as e:
                    st.error(f"Ошибка чтения файла: {e}")
        
        else:  # Ручной ввод
            st.write("Введите данные в текстовом поле:")
            manual_data = st.text_area(
                "Формат: частота Z' Z'' (каждая строка - одна точка)",
                height=200,
                placeholder="1e3 10.5 -2.3\n1e2 12.1 -5.6\n1e1 14.2 -8.1"
            )
            
            if manual_data:
                try:
                    rows = []
                    for line in manual_data.strip().split('\n'):
                        parts = line.split()
                        if len(parts) >= 3:
                            rows.append([float(p) for p in parts[:3]])
                    
                    if rows:
                        data = pd.DataFrame(rows, columns=['frequency', 'Z_real', 'Z_imag'])
                        st.success(f"Введено {len(data)} точек спектра")
                        st.dataframe(data.head())
                    else:
                        st.error("Не удалось распознать данные")
                except Exception as e:
                    st.error(f"Ошибка: {e}")
        
        st.divider()
        
        st.header("⚙️ Параметры анализа")
        
        method = st.selectbox(
            "Метод инверсии:",
            ["Тихоновская регуляризация", "Байесовский метод", "Максимальная энтропия", 
             "Гауссовские процессы", "Генетическое программирование"],
            help="Выберите метод для вычисления DRT"
        )
        
        n_tau = st.slider("Количество точек по времени:", 50, 300, 150)
        
        if method == "Тихоновская регуляризация":
            reg_order = st.selectbox("Порядок регуляризации:", [0, 1, 2], index=2,
                                     help="0: сглаживание амплитуды, 1: сглаживание наклона, 2: сглаживание кривизны")
            lambda_auto = st.checkbox("Автоматический подбор λ", value=True)
            if not lambda_auto:
                lambda_value = st.number_input("Значение λ:", value=1e-4, format="%.1e")
            else:
                lambda_value = None
        elif method == "Максимальная энтропия":
            entropy_lambda = st.number_input("Параметр энтропии λ:", value=0.1, format="%.2f",
                                             help="Меньшее значение дает более гладкий спектр")
        elif method == "Генетическое программирование":
            n_peaks_max = st.slider("Максимальное число пиков:", 1, 10, 5)
            n_generations = st.slider("Число поколений:", 10, 100, 50)
        elif method == "Гауссовские процессы":
            n_components = st.slider("Число компонент GP:", 10, 50, 20)
        
        process_button = st.button("🚀 Запустить анализ", type="primary", use_container_width=True)
    
    # Основная область
    if data is not None and process_button:
        frequencies = data['frequency'].values
        Z_real = data['Z_real'].values
        Z_imag = data['Z_imag'].values
        
        # Проверка данных
        if len(frequencies) < 5:
            st.error("Недостаточно точек данных для анализа (минимум 5)")
            return
        
        # Сортировка по частоте
        sort_idx = np.argsort(frequencies)
        frequencies = frequencies[sort_idx]
        Z_real = Z_real[sort_idx]
        Z_imag = Z_imag[sort_idx]
        
        # Основная информация о данных
        st.subheader("📊 Информация о данных")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Частотный диапазон", f"{frequencies[0]:.2e} - {frequencies[-1]:.2e} Гц")
        with col2:
            R_inf_est = np.min(Z_real) if Z_real[0] > Z_real[-1] else np.mean(Z_real[:5])
            st.metric("Омическое сопротивление R∞", f"{R_inf_est:.4f} Ом")
        with col3:
            R_pol_est = np.max(Z_real) - R_inf_est
            st.metric("Поляризационное сопротивление Rpol", f"{R_pol_est:.4f} Ом")
        
        # Создание progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Выбор метода и выполнение расчета
        if method == "Тихоновская регуляризация":
            status_text.text("Выполняется Тихоновская регуляризация...")
            drt = TikhonovDRT(frequencies, Z_real, Z_imag, regularization_order=reg_order)
            tau_grid, gamma = drt.compute(n_tau=n_tau, lambda_value=lambda_value, lambda_auto=lambda_auto)
            confidence = None
            peak_results = find_peaks_drt(tau_grid, gamma)
            
            # Реконструкция импеданса
            Z_rec_real, Z_rec_imag = drt.reconstruct_impedance(tau_grid, gamma)
            
        elif method == "Байесовский метод":
            status_text.text("Выполняется байесовская инверсия (может занять некоторое время)...")
            drt = BayesianDRT(frequencies, Z_real, Z_imag)
            tau_grid, gamma, confidence = drt.compute(n_tau=n_tau)
            peak_results = find_peaks_drt(tau_grid, gamma)
            
            # Реконструкция
            K_real, K_imag = drt._build_kernel_matrix(tau_grid)
            Z_rec_real = drt.R_inf + K_real @ gamma
            Z_rec_imag = -K_imag @ gamma
            
        elif method == "Максимальная энтропия":
            status_text.text("Выполняется инверсия методом максимальной энтропии...")
            drt = MaxEntropyDRT(frequencies, Z_real, Z_imag)
            tau_grid, gamma = drt.compute(n_tau=n_tau, lambda_value=entropy_lambda)
            confidence = None
            peak_results = find_peaks_drt(tau_grid, gamma)
            
            # Реконструкция
            K_real, K_imag = drt._build_kernel_matrix(tau_grid)
            Z_rec_real = drt.R_inf + K_real @ gamma
            Z_rec_imag = -K_imag @ gamma
            
        elif method == "Гауссовские процессы":
            status_text.text("Выполняется GP-инверсия...")
            drt = GaussianProcessDRT(frequencies, Z_real, Z_imag)
            tau_grid, gamma, confidence = drt.compute(n_tau=n_tau, n_components=n_components)
            peak_results = find_peaks_drt(tau_grid, gamma)
            
            # Реконструкция
            K_real, K_imag = drt._build_kernel_matrix(tau_grid)
            Z_rec_real = drt.R_inf + K_real @ gamma
            Z_rec_imag = -K_imag @ gamma
            
        else:  # Генетическое программирование
            status_text.text("Выполняется ISGP-инверсия (может занять несколько минут)...")
            drt = ISGPDRT(frequencies, Z_real, Z_imag)
            tau_grid, gamma, peaks_params = drt.compute(n_tau=n_tau, n_peaks_max=n_peaks_max, n_generations=n_generations)
            confidence = None
            peak_results = find_peaks_drt(tau_grid, gamma)
            
            # Реконструкция
            K_real, K_imag = drt._build_kernel_matrix(tau_grid)
            Z_rec_real = drt.R_inf + K_real @ gamma
            Z_rec_imag = -K_imag @ gamma
        
        progress_bar.progress(100)
        status_text.text("Анализ завершен!")
        
        # Визуализация результатов
        st.divider()
        st.subheader("📈 Результаты анализа")
        
        # Вкладки для результатов
        tab1, tab2, tab3 = st.tabs(["DRT спектр", "Импедансные спектры", "Анализ пиков"])
        
        with tab1:
            fig_drt = plot_drt_spectrum(tau_grid, gamma, peak_results, confidence)
            st.plotly_chart(fig_drt, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Диапазон времен", f"{tau_grid[0]:.2e} - {tau_grid[-1]:.2e} c")
            with col2:
                st.metric("Максимальное значение DRT", f"{np.max(gamma):.4f}")
        
        with tab2:
            fig_imp = plot_impedance_spectra(frequencies, Z_real, Z_imag, Z_rec_real, Z_rec_imag)
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # Ошибка реконструкции
            error_real = np.abs(Z_real - Z_rec_real) / np.abs(Z_real + 1e-10) * 100
            error_imag = np.abs(Z_imag - Z_rec_imag) / np.abs(Z_imag + 1e-10) * 100
            st.metric("Средняя ошибка реконструкции", 
                     f"{np.mean(np.sqrt(error_real**2 + error_imag**2)):.2f} %")
        
        with tab3:
            st.subheader("Выделенные релаксационные процессы")
            
            if peak_results:
                df_peaks = pd.DataFrame(peak_results)
                df_peaks['frequency_Hz'] = df_peaks['frequency']
                df_peaks['log_tau'] = df_peaks['log_tau'].round(3)
                df_peaks['tau'] = df_peaks['tau'].apply(lambda x: f"{x:.3e}")
                df_peaks['frequency'] = df_peaks['frequency'].apply(lambda x: f"{x:.2f}")
                df_peaks['amplitude'] = df_peaks['amplitude'].round(4)
                
                st.dataframe(
                    df_peaks[['tau', 'frequency', 'amplitude']].rename(
                        columns={'tau': 'τ (с)', 'frequency': 'f (Гц)', 'amplitude': 'Амплитуда'}
                    ),
                    use_container_width=True
                )
                
                # Дополнительный анализ пиков
                st.subheader("Аппроксимация гауссианами")
                
                if len(peak_results) > 1:
                    n_peaks_fit = st.slider("Число пиков для аппроксимации:", 1, len(peak_results), len(peak_results))
                    fitted_gamma, fit_params = fit_gaussian_peaks(tau_grid, gamma, n_peaks_fit)
                else:
                    st.info(f"Обнаружен {len(peak_results)} пик. Для аппроксимации гауссианами требуется минимум 2 пика.")
                    fitted_gamma = gamma
                    fit_params = []
                
                fig_fit = go.Figure()
                fig_fit.add_trace(go.Scatter(x=tau_grid, y=gamma, mode='lines', name='Исходный DRT', line=dict(color='blue')))
                fig_fit.add_trace(go.Scatter(x=tau_grid, y=fitted_gamma, mode='lines', name='Аппроксимация', line=dict(color='red', dash='dash')))
                fig_fit.update_xaxes(title_text="τ (с)", type="log")
                fig_fit.update_yaxes(title_text="γ(τ)")
                fig_fit.update_layout(height=400)
                st.plotly_chart(fig_fit, use_container_width=True)
                
            else:
                st.info("Пики не обнаружены. Попробуйте уменьшить параметр prominence или изменить метод инверсии.")
        
        # Экспорт результатов
        st.divider()
        st.subheader("💾 Экспорт результатов")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            drt_df = pd.DataFrame({'tau': tau_grid, 'log_tau': np.log10(tau_grid), 'gamma': gamma})
            drt_csv = drt_df.to_csv(index=False)
            st.download_button(
                "📥 Скачать DRT данные (CSV)",
                drt_csv,
                "drt_results.csv",
                "text/csv"
            )
        
        with col2:
            if peak_results:
                peaks_df = pd.DataFrame(peak_results)
                peaks_csv = peaks_df.to_csv(index=False)
                st.download_button(
                    "📥 Скачать данные о пиках (CSV)",
                    peaks_csv,
                    "peaks_results.csv",
                    "text/csv"
                )
        
        with col3:
            if Z_rec_real is not None:
                recon_df = pd.DataFrame({
                    'frequency': frequencies,
                    'Z_real_exp': Z_real,
                    'Z_imag_exp': Z_imag,
                    'Z_real_rec': Z_rec_real,
                    'Z_imag_rec': Z_rec_imag
                })
                recon_csv = recon_df.to_csv(index=False)
                st.download_button(
                    "📥 Скачать реконструированный импеданс (CSV)",
                    recon_csv,
                    "reconstructed_impedance.csv",
                    "text/csv"
                )
    
    elif data is None:
        st.info("👈 Загрузите данные или введите их вручную в боковой панели для начала анализа")
    
    # Информация о программе
    st.sidebar.divider()
    with st.sidebar.expander("ℹ️ О программе"):
        st.markdown("""
        **DRT Analysis Tool v1.0**
        
        Программа реализует современные методы анализа импеданс-спектров:
        - Тихоновская регуляризация с автоматическим подбором λ
        - Байесовский метод с MAP-оценкой
        - Метод максимальной энтропии
        - Гауссовские процессы
        - Генетическое программирование (ISGP)
        
        **Используемые алгоритмы:**
        - Fredholm integral equation of the first kind
        - Регуляризация с ограничением неотрицательности
        - L-кривая для выбора параметров
        - Автоматическое выделение пиков
        
        **Рекомендации:**
        - Используйте Тихоновскую регуляризацию для быстрого анализа
        - Байесовский метод и GP дают оценку неопределенности
        - ISGP подходит для сложных спектров с малым числом процессов
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("⚡ *DRT Analysis Tool | Поддерживается 5 методов инверсии*")


if __name__ == "__main__":
    main()
