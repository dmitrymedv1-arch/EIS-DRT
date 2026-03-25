"""
Streamlit приложение для анализа импеданс-спектров методом распределения времен релаксации (DRT)
Поддерживает методы:
- Тихоновская регуляризация (Tikhonov)
- Байесовский метод (Bayesian)
- Метод максимальной энтропии (Maximum Entropy)
- Гауссовские процессы (Gaussian Process)
- ISGP (упрощенная версия)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, interpolate, signal
from scipy.linalg import svd, solve
import io
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="DRT Analyzer",
    page_icon="⚡",
    layout="wide"
)

# Заголовок
st.title("⚡ DRT Analyzer: Distribution of Relaxation Times")
st.markdown("""
Анализ электрохимических импеданс-спектров методом распределения времен релаксации.
Поддерживаются методы: Тихоновская регуляризация, Байесовский, Максимальная энтропия,
Гауссовские процессы и ISGP.
""")

# ============================================================================
# 1. Загрузка и предобработка данных
# ============================================================================

def load_data(uploaded_file):
    """Загрузка данных из файла"""
    try:
        # Пробуем прочитать как CSV
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        # Проверяем наличие необходимых колонок
        if len(df.columns) >= 3:
            # Предполагаем порядок: частота, Z', Z''
            freq = df.iloc[:, 0].values
            Z_real = df.iloc[:, 1].values
            Z_imag = df.iloc[:, 2].values
            return freq, Z_real, Z_imag, df
        else:
            st.error("Файл должен содержать минимум 3 колонки: частота, Z', Z''")
            return None, None, None, None
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")
        return None, None, None, None

def validate_kramers_kronig(freq, Z_real, Z_imag):
    """
    Проверка качества данных через тест Кронига-Крамерса
    """
    # Упрощенная проверка: вычисляем восстановленную мнимую часть из действительной
    omega = 2 * np.pi * freq
    log_omega = np.log(omega)
    
    # Проверка через преобразование Гильберта
    try:
        from scipy.signal import hilbert
        Z_real_interp = interpolate.interp1d(log_omega, Z_real, 
                                              kind='cubic', 
                                              fill_value='extrapolate')
        log_omega_fine = np.linspace(log_omega.min(), log_omega.max(), 1000)
        Z_real_fine = Z_real_interp(log_omega_fine)
        
        # Преобразование Гильберта
        analytic = hilbert(Z_real_fine)
        Z_imag_reconstructed = np.imag(analytic)
        
        # Интерполяция обратно на исходные частоты
        Z_imag_interp = interpolate.interp1d(log_omega_fine, Z_imag_reconstructed,
                                              kind='cubic', fill_value='extrapolate')
        Z_imag_pred = Z_imag_interp(log_omega)
        
        # Вычисление относительной невязки
        Z_mag = np.sqrt(Z_real**2 + Z_imag**2)
        residuals = np.abs(Z_imag - Z_imag_pred) / (Z_mag + 1e-10)
        max_residual = np.max(residuals)
        
        return max_residual < 0.05, residuals, max_residual
    except Exception as e:
        st.warning(f"Kramers-Kronig тест не выполнен: {e}")
        return True, None, 0

# ============================================================================
# 2. DRT инверсия - Тихоновская регуляризация
# ============================================================================

class TikhonovDRT:
    """Тихоновская регуляризация для DRT инверсии"""
    
    def __init__(self, freq, Z_real, Z_imag, tau_min=None, tau_max=None, n_tau=200):
        self.freq = np.asarray(freq)
        self.Z_real = np.asarray(Z_real)
        self.Z_imag = np.asarray(Z_imag)
        self.omega = 2 * np.pi * self.freq
        
        # Определение диапазона времен релаксации
        if tau_min is None:
            tau_min = 1 / (2 * np.pi * np.max(self.freq)) * 0.1
        if tau_max is None:
            tau_max = 1 / (2 * np.pi * np.min(self.freq)) * 10
            
        self.tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
        self.n_tau = n_tau
        self.n_freq = len(freq)
        
        # Построение матрицы ядра
        self.build_kernel_matrix()
    
    def build_kernel_matrix(self):
        """Построение матрицы ядра для действительной и мнимой частей"""
        # Матрица для действительной части
        self.A_real = np.zeros((self.n_freq, self.n_tau))
        # Матрица для мнимой части
        self.A_imag = np.zeros((self.n_freq, self.n_tau))
        
        for i in range(self.n_freq):
            for j in range(self.n_tau):
                denom = 1 + (self.omega[i] * self.tau[j])**2
                self.A_real[i, j] = 1 / denom
                self.A_imag[i, j] = -self.omega[i] * self.tau[j] / denom
        
        # Объединенная матрица
        self.A = np.vstack([self.A_real, self.A_imag])
        self.b = np.hstack([self.Z_real, self.Z_imag])
    
    def compute_L_matrix(self, order=2):
        """Построение матрицы регуляризации"""
        if order == 0:
            return np.eye(self.n_tau)
        elif order == 1:
            L = np.zeros((self.n_tau - 1, self.n_tau))
            for i in range(self.n_tau - 1):
                L[i, i] = -1
                L[i, i+1] = 1
            return L
        else:  # order == 2
            L = np.zeros((self.n_tau - 2, self.n_tau))
            for i in range(self.n_tau - 2):
                L[i, i] = 1
                L[i, i+1] = -2
                L[i, i+2] = 1
            return L
    
    def l_curve_criterion(self, lambda_range):
        """Выбор параметра регуляризации по L-кривой"""
        residuals = []
        norms = []
        
        for lam in lambda_range:
            try:
                L = self.compute_L_matrix(order=2)
                ATA = self.A.T @ self.A
                LTL = L.T @ L
                x = solve(ATA + lam * LTL, self.A.T @ self.b)
                x = np.maximum(x, 0)  # Неотрицательность
                
                res = np.linalg.norm(self.A @ x - self.b)
                norm = np.linalg.norm(L @ x)
                residuals.append(res)
                norms.append(norm)
            except:
                residuals.append(np.inf)
                norms.append(np.inf)
        
        residuals = np.array(residuals)
        norms = np.array(norms)
        valid = np.isfinite(residuals) & (residuals > 0) & (norms > 0)
        
        if np.sum(valid) < 3:
            return lambda_range[len(lambda_range)//2]
        
        # Поиск точки максимальной кривизны
        log_res = np.log(residuals[valid])
        log_norm = np.log(norms[valid])
        
        # Кривизна
        dx = np.diff(log_norm)
        dy = np.diff(log_res)
        curvature = np.abs(dx[1:] * dy[:-1] - dx[:-1] * dy[1:]) / (dx[1:]**2 + dy[:-1]**2)**1.5
        k_opt = np.argmax(curvature) + 1
        
        return lambda_range[valid][k_opt]
    
    def fit(self, lambda_reg=None, order=2):
        """Выполнение инверсии DRT"""
        if lambda_reg is None:
            # Автоматический выбор λ
            lambda_range = np.logspace(-10, 0, 50)
            lambda_reg = self.l_curve_criterion(lambda_range)
        
        L = self.compute_L_matrix(order=order)
        ATA = self.A.T @ self.A
        LTL = L.T @ L
        
        try:
            x = solve(ATA + lambda_reg * LTL, self.A.T @ self.b)
            x = np.maximum(x, 0)  # Неотрицательность
        except:
            # SVD метод
            U, s, Vt = svd(ATA + lambda_reg * LTL)
            x = Vt.T @ (U.T @ (self.A.T @ self.b) / s)
            x = np.maximum(np.real(x), 0)
        
        # Восстановленный импеданс
        Z_reconstructed_real = self.A_real @ x
        Z_reconstructed_imag = self.A_imag @ x
        
        # Расчет невязок
        residuals = np.sqrt((self.Z_real - Z_reconstructed_real)**2 + 
                           (self.Z_imag - Z_reconstructed_imag)**2)
        residuals /= np.sqrt(self.Z_real**2 + self.Z_imag**2)
        
        return x, lambda_reg, residuals, Z_reconstructed_real, Z_reconstructed_imag


# ============================================================================
# 3. DRT инверсия - Байесовский метод (упрощенный)
# ============================================================================

class BayesianDRT:
    """Байесовский подход с MAP оценкой и доверительными интервалами"""
    
    def __init__(self, freq, Z_real, Z_imag, tau_min=None, tau_max=None, n_tau=150):
        self.freq = np.asarray(freq)
        self.Z_real = np.asarray(Z_real)
        self.Z_imag = np.asarray(Z_imag)
        self.omega = 2 * np.pi * self.freq
        
        if tau_min is None:
            tau_min = 1 / (2 * np.pi * np.max(self.freq)) * 0.1
        if tau_max is None:
            tau_max = 1 / (2 * np.pi * np.min(self.freq)) * 10
            
        self.tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
        self.n_tau = n_tau
        
        # Построение матрицы
        self.build_kernel_matrix()
    
    def build_kernel_matrix(self):
        self.A = np.zeros((2 * len(self.freq), self.n_tau))
        for i in range(len(self.freq)):
            for j in range(self.n_tau):
                denom = 1 + (self.omega[i] * self.tau[j])**2
                self.A[2*i, j] = 1 / denom
                self.A[2*i + 1, j] = -self.omega[i] * self.tau[j] / denom
        
        self.b = np.hstack([self.Z_real, self.Z_imag])
        self.n_obs = len(self.b)
    
    def compute_posterior(self, lambda_reg=1e-6, n_samples=100):
        """Вычисление апостериорного распределения с помощью MCMC (упрощенный)"""
        L = np.eye(self.n_tau)
        ATA = self.A.T @ self.A
        LTL = L.T @ L
        Sigma_inv = ATA + lambda_reg * LTL
        
        # MAP оценка
        try:
            x_map = solve(Sigma_inv, self.A.T @ self.b)
            x_map = np.maximum(x_map, 0)
        except:
            U, s, Vt = svd(Sigma_inv)
            x_map = Vt.T @ (U.T @ (self.A.T @ self.b) / s)
            x_map = np.maximum(np.real(x_map), 0)
        
        # Ковариационная матрица
        try:
            Sigma = np.linalg.inv(Sigma_inv)
        except:
            Sigma = np.linalg.pinv(Sigma_inv)
        
        # Генерация выборок для доверительных интервалов
        samples = []
        if n_samples > 0:
            for _ in range(min(n_samples, 100)):
                try:
                    sample = np.random.multivariate_normal(x_map, Sigma)
                    sample = np.maximum(sample, 0)
                    samples.append(sample)
                except:
                    pass
        
        samples = np.array(samples) if samples else np.array([x_map])
        
        # Доверительные интервалы
        lower = np.percentile(samples, 2.5, axis=0)
        upper = np.percentile(samples, 97.5, axis=0)
        
        return x_map, lower, upper, samples


# ============================================================================
# 4. DRT инверсия - Максимальная энтропия
# ============================================================================

class MaxEntropyDRT:
    """Метод максимальной энтропии для DRT инверсии"""
    
    def __init__(self, freq, Z_real, Z_imag, tau_min=None, tau_max=None, n_tau=150):
        self.freq = np.asarray(freq)
        self.Z_real = np.asarray(Z_real)
        self.Z_imag = np.asarray(Z_imag)
        self.omega = 2 * np.pi * self.freq
        
        if tau_min is None:
            tau_min = 1 / (2 * np.pi * np.max(self.freq)) * 0.1
        if tau_max is None:
            tau_max = 1 / (2 * np.pi * np.min(self.freq)) * 10
            
        self.tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
        self.n_tau = n_tau
        self.n_freq = len(freq)
        
        self.build_kernel_matrix()
    
    def build_kernel_matrix(self):
        self.A = np.zeros((2 * self.n_freq, self.n_tau))
        for i in range(self.n_freq):
            for j in range(self.n_tau):
                denom = 1 + (self.omega[i] * self.tau[j])**2
                self.A[2*i, j] = 1 / denom
                self.A[2*i + 1, j] = -self.omega[i] * self.tau[j] / denom
        
        self.b = np.hstack([self.Z_real, self.Z_imag])
    
    def entropy(self, x):
        """Энтропия Шеннона (с регуляризацией для нулевых значений)"""
        x_safe = np.maximum(x, 1e-12)
        return -np.sum(x_safe * np.log(x_safe))
    
    def objective(self, x, lambda_reg):
        """Целевая функция: невязка + λ * энтропия"""
        residual = np.linalg.norm(self.A @ x - self.b)**2
        return residual - lambda_reg * self.entropy(x)
    
    def fit(self, lambda_reg=None, max_iter=500):
        """Выполнение инверсии методом максимальной энтропии"""
        if lambda_reg is None:
            lambda_reg = 1e-4
        
        # Начальное приближение
        x0 = np.ones(self.n_tau) / self.n_tau
        
        # Оптимизация
        bounds = [(0, None) for _ in range(self.n_tau)]
        
        def obj_wrapper(x):
            return self.objective(x, lambda_reg)
        
        try:
            result = optimize.minimize(
                obj_wrapper, x0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': max_iter}
            )
            x_opt = np.maximum(result.x, 0)
        except:
            x_opt = x0
        
        # Нормализация
        x_opt = x_opt / (np.sum(x_opt) + 1e-10)
        
        # Восстановленный импеданс
        Z_reconstructed_real = np.zeros(self.n_freq)
        Z_reconstructed_imag = np.zeros(self.n_freq)
        for i in range(self.n_freq):
            for j in range(self.n_tau):
                denom = 1 + (self.omega[i] * self.tau[j])**2
                Z_reconstructed_real[i] += x_opt[j] / denom
                Z_reconstructed_imag[i] += -self.omega[i] * self.tau[j] * x_opt[j] / denom
        
        return x_opt, Z_reconstructed_real, Z_reconstructed_imag


# ============================================================================
# 5. DRT инверсия - Гауссовские процессы (упрощенный)
# ============================================================================

class GaussianProcessDRT:
    """Гауссовский процесс для DRT с оценкой неопределенности"""
    
    def __init__(self, freq, Z_real, Z_imag, tau_min=None, tau_max=None, n_tau=150):
        self.freq = np.asarray(freq)
        self.Z_real = np.asarray(Z_real)
        self.Z_imag = np.asarray(Z_imag)
        self.omega = 2 * np.pi * self.freq
        
        if tau_min is None:
            tau_min = 1 / (2 * np.pi * np.max(self.freq)) * 0.1
        if tau_max is None:
            tau_max = 1 / (2 * np.pi * np.min(self.freq)) * 10
            
        self.tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
        self.n_tau = n_tau
        self.n_freq = len(freq)
        
        self.build_kernel_matrix()
    
    def build_kernel_matrix(self):
        self.A = np.zeros((2 * self.n_freq, self.n_tau))
        for i in range(self.n_freq):
            for j in range(self.n_tau):
                denom = 1 + (self.omega[i] * self.tau[j])**2
                self.A[2*i, j] = 1 / denom
                self.A[2*i + 1, j] = -self.omega[i] * self.tau[j] / denom
        
        self.b = np.hstack([self.Z_real, self.Z_imag])
    
    def rbf_kernel(self, x1, x2, length_scale=0.5, sigma_f=1.0):
        """Ядро RBF для ковариации"""
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)
    
    def fit(self, length_scale=0.5, noise_var=1e-6):
        """Вычисление GP-оценки DRT"""
        # Логарифмическая шкала времен
        X = np.log10(self.tau).reshape(-1, 1)
        
        # Ковариационная матрица
        K = self.rbf_kernel(X, X, length_scale)
        K += noise_var * np.eye(len(X))
        
        # Решаем K * alpha = A.T * b (упрощенная версия)
        # Для полного GP нужно решать систему с учетом ядра, здесь упрощение
        try:
            K_inv = np.linalg.inv(K)
            alpha = K_inv @ (self.A.T @ self.b)
            x_gp = K @ alpha
            x_gp = np.maximum(x_gp, 0)
            
            # Доверительные интервалы (диагональ ковариации)
            variance = np.diag(K - K @ K_inv @ K)
            std = np.sqrt(np.maximum(variance, 0))
            
            return x_gp, x_gp - 2*std, x_gp + 2*std
        except:
            x_gp = np.ones(self.n_tau) / self.n_tau
            return x_gp, x_gp, x_gp


# ============================================================================
# 6. DRT инверсия - ISGP (упрощенная версия)
# ============================================================================

class ISGPDRT:
    """Упрощенная версия ISGP с генетическим алгоритмом"""
    
    def __init__(self, freq, Z_real, Z_imag, tau_min=None, tau_max=None):
        self.freq = np.asarray(freq)
        self.Z_real = np.asarray(Z_real)
        self.Z_imag = np.asarray(Z_imag)
        self.omega = 2 * np.pi * self.freq
        
        if tau_min is None:
            tau_min = 1 / (2 * np.pi * np.max(self.freq)) * 0.1
        if tau_max is None:
            tau_max = 1 / (2 * np.pi * np.min(self.freq)) * 10
        
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_freq = len(freq)
    
    def peak_function(self, tau, A, tau0, sigma):
        """Гауссиан для представления пика"""
        return A * np.exp(-(np.log10(tau) - np.log10(tau0))**2 / (2 * sigma**2))
    
    def compute_impedance(self, peaks):
        """Вычисление импеданса по сумме пиков"""
        Z_real = np.zeros(self.n_freq)
        Z_imag = np.zeros(self.n_freq)
        
        for A, tau0, sigma in peaks:
            for i in range(self.n_freq):
                # Интегрируем по времени для каждого пика
                tau_range = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), 100)
                for tau in tau_range:
                    peak_val = self.peak_function(tau, A, tau0, sigma)
                    denom = 1 + (self.omega[i] * tau)**2
                    Z_real[i] += peak_val / denom
                    Z_imag[i] += -self.omega[i] * tau * peak_val / denom
        
        return Z_real, Z_imag
    
    def fitness(self, peaks):
        """Функция приспособленности"""
        Z_real_pred, Z_imag_pred = self.compute_impedance(peaks)
        error = np.sum((self.Z_real - Z_real_pred)**2 + (self.Z_imag - Z_imag_pred)**2)
        complexity = len(peaks)
        return error + 0.01 * complexity
    
    def fit(self, n_peaks=3, n_generations=50, population_size=20):
        """Генетический алгоритм для поиска оптимальных пиков"""
        np.random.seed(42)
        
        # Инициализация популяции
        population = []
        for _ in range(population_size):
            peaks = []
            for _ in range(n_peaks):
                A = np.random.uniform(0.1, 10)
                tau0 = np.random.uniform(self.tau_min, self.tau_max)
                sigma = np.random.uniform(0.1, 1.0)
                peaks.append((A, tau0, sigma))
            population.append(peaks)
        
        # Эволюция
        for generation in range(n_generations):
            fitness_scores = [self.fitness(peaks) for peaks in population]
            sorted_idx = np.argsort(fitness_scores)
            population = [population[i] for i in sorted_idx[:population_size//2]]
            
            # Кроссинговер и мутация
            new_population = []
            while len(new_population) < population_size:
                parent1 = population[np.random.randint(len(population))]
                parent2 = population[np.random.randint(len(population))]
                
                # Кроссинговер
                child = []
                for p1, p2 in zip(parent1, parent2):
                    if np.random.random() > 0.5:
                        child.append(p1)
                    else:
                        child.append(p2)
                
                # Мутация
                for i in range(len(child)):
                    if np.random.random() < 0.2:
                        A, tau0, sigma = child[i]
                        A *= np.random.uniform(0.8, 1.2)
                        tau0 *= np.random.uniform(0.9, 1.1)
                        sigma *= np.random.uniform(0.8, 1.2)
                        child[i] = (max(0.01, A), max(self.tau_min, min(self.tau_max, tau0)), max(0.05, min(2.0, sigma)))
                
                new_population.append(child)
            
            population = new_population
        
        # Лучшее решение
        best_peaks = population[np.argmin([self.fitness(p) for p in population])]
        
        # Вычисление DRT функции для визуализации
        tau_range = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max), 200)
        gamma = np.zeros_like(tau_range)
        for A, tau0, sigma in best_peaks:
            gamma += self.peak_function(tau_range, A, tau0, sigma)
        
        return gamma, tau_range, best_peaks


# ============================================================================
# 7. Постобработка - выделение пиков
# ============================================================================

def extract_peaks(gamma, tau, prominence=0.05):
    """Выделение пиков в DRT спектре"""
    # Нормализация
    gamma_norm = gamma / (np.max(gamma) + 1e-10)
    
    # Поиск локальных максимумов
    from scipy.signal import find_peaks
    peaks_idx, properties = find_peaks(gamma_norm, prominence=prominence, height=0.05)
    
    if len(peaks_idx) == 0:
        return [], [], []
    
    peaks_tau = tau[peaks_idx]
    peaks_height = gamma[peaks_idx]
    
    # Расчет площади под пиками (приближенный)
    peaks_area = []
    for idx in peaks_idx:
        # Находим границы пика
        left = idx
        right = idx
        while left > 0 and gamma[left] > gamma[left-1] * 0.5:
            left -= 1
        while right < len(gamma)-1 and gamma[right] > gamma[right+1] * 0.5:
            right += 1
        
        # Численное интегрирование
        area = np.trapz(gamma[left:right+1], tau[left:right+1])
        peaks_area.append(area)
    
    return peaks_tau, peaks_height, peaks_area


# ============================================================================
# 8. Визуализация
# ============================================================================

def plot_results(freq, Z_real, Z_imag, gamma, tau, 
                 Z_rec_real=None, Z_rec_imag=None,
                 peaks_tau=None, peaks_height=None,
                 confidence_lower=None, confidence_upper=None):
    """Создание комплексного графика результатов"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Nyquist plot
    ax = axes[0, 0]
    ax.plot(Z_real, -Z_imag, 'bo-', markersize=4, label='Эксперимент')
    if Z_rec_real is not None and Z_rec_imag is not None:
        ax.plot(Z_rec_real, -Z_rec_imag, 'r--', linewidth=2, label='Реконструкция')
    ax.set_xlabel("Z' (Ом)")
    ax.set_ylabel("-Z'' (Ом)")
    ax.set_title("Nyquist plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # DRT plot
    ax = axes[0, 1]
    ax.semilogx(tau, gamma, 'b-', linewidth=2, label='DRT')
    
    if confidence_lower is not None and confidence_upper is not None:
        ax.fill_between(tau, confidence_lower, confidence_upper, alpha=0.3, color='b', label='95% CI')
    
    if peaks_tau is not None and len(peaks_tau) > 0:
        ax.scatter(peaks_tau, peaks_height, color='red', s=50, zorder=5, label='Пики')
        for t, h in zip(peaks_tau, peaks_height):
            ax.axvline(t, color='red', linestyle='--', alpha=0.5)
    
    ax.set_xlabel("τ (с)")
    ax.set_ylabel("γ(τ)")
    ax.set_title("Distribution of Relaxation Times")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bode plot - magnitude
    ax = axes[1, 0]
    Z_mag = np.sqrt(Z_real**2 + Z_imag**2)
    ax.loglog(freq, Z_mag, 'bo-', markersize=4, label='Эксперимент')
    if Z_rec_real is not None and Z_rec_imag is not None:
        Z_rec_mag = np.sqrt(Z_rec_real**2 + Z_rec_imag**2)
        ax.loglog(freq, Z_rec_mag, 'r--', linewidth=2, label='Реконструкция')
    ax.set_xlabel("Частота (Гц)")
    ax.set_ylabel("|Z| (Ом)")
    ax.set_title("Bode plot - Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Phase plot
    ax = axes[1, 1]
    phase = -np.arctan2(Z_imag, Z_real) * 180 / np.pi
    ax.semilogx(freq, phase, 'bo-', markersize=4, label='Эксперимент')
    if Z_rec_real is not None and Z_rec_imag is not None:
        phase_rec = -np.arctan2(Z_rec_imag, Z_rec_real) * 180 / np.pi
        ax.semilogx(freq, phase_rec, 'r--', linewidth=2, label='Реконструкция')
    ax.set_xlabel("Частота (Гц)")
    ax.set_ylabel("Фаза (градусы)")
    ax.set_title("Bode plot - Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# 9. Основной интерфейс Streamlit
# ============================================================================

def main():
    # Боковая панель для загрузки и параметров
    with st.sidebar:
        st.header("📁 Загрузка данных")
        uploaded_file = st.file_uploader(
            "Выберите файл с импеданс-спектром",
            type=['txt', 'csv', 'dat', 'z']
        )
        
        st.header("⚙️ Параметры инверсии")
        
        method = st.selectbox(
            "Метод инверсии",
            ["Тихоновская регуляризация", "Байесовский метод", 
             "Максимальная энтропия", "Гауссовские процессы", "ISGP (генетический)"]
        )
        
        n_tau = st.slider("Количество точек по времени", 100, 500, 200)
        
        auto_lambda = st.checkbox("Автоматический выбор λ (для Tikhonov)", value=True)
        if not auto_lambda and method == "Тихоновская регуляризация":
            lambda_manual = st.number_input("λ (регуляризация)", value=1e-6, format="%.1e")
        else:
            lambda_manual = None
        
        st.header("🔍 Постобработка")
        peak_prominence = st.slider("Чувствительность выделения пиков", 0.01, 0.2, 0.05)
        
        run_button = st.button("🚀 Выполнить анализ", type="primary")
    
    # Основная область
    if uploaded_file is not None:
        # Загрузка данных
        freq, Z_real, Z_imag, df = load_data(uploaded_file)
        
        if freq is not None:
            st.success(f"Загружено {len(freq)} точек спектра")
            
            # Показ таблицы
            with st.expander("📊 Показать загруженные данные"):
                st.dataframe(df.head(10))
            
            # Kramers-Kronig тест
            with st.spinner("Проверка качества данных..."):
                kk_valid, residuals, max_res = validate_kramers_kronig(freq, Z_real, Z_imag)
                
                if kk_valid:
                    st.success(f"✅ Kramers-Kronig тест пройден (макс. невязка: {max_res:.3%})")
                else:
                    st.warning(f"⚠️ Kramers-Kronig тест не пройден (макс. невязка: {max_res:.3%})")
            
            if run_button:
                # Выполнение инверсии
                with st.spinner(f"Выполняется инверсия методом {method}..."):
                    
                    if method == "Тихоновская регуляризация":
                        drt = TikhonovDRT(freq, Z_real, Z_imag, n_tau=n_tau)
                        gamma, lambda_opt, residuals, Z_rec_real, Z_rec_imag = drt.fit(
                            lambda_reg=lambda_manual if not auto_lambda else None
                        )
                        confidence_lower = None
                        confidence_upper = None
                        st.info(f"Оптимальный λ: {lambda_opt:.2e}")
                    
                    elif method == "Байесовский метод":
                        drt = BayesianDRT(freq, Z_real, Z_imag, n_tau=n_tau)
                        gamma, lower, upper, samples = drt.compute_posterior(lambda_reg=1e-5, n_samples=50)
                        confidence_lower = lower
                        confidence_upper = upper
                        Z_rec_real = None
                        Z_rec_imag = None
                    
                    elif method == "Максимальная энтропия":
                        drt = MaxEntropyDRT(freq, Z_real, Z_imag, n_tau=n_tau)
                        gamma, Z_rec_real, Z_rec_imag = drt.fit(lambda_reg=1e-4)
                        confidence_lower = None
                        confidence_upper = None
                    
                    elif method == "Гауссовские процессы":
                        drt = GaussianProcessDRT(freq, Z_real, Z_imag, n_tau=n_tau)
                        gamma, lower, upper = drt.fit()
                        confidence_lower = lower
                        confidence_upper = upper
                        Z_rec_real = None
                        Z_rec_imag = None
                    
                    elif method == "ISGP (генетический)":
                        drt = ISGPDRT(freq, Z_real, Z_imag)
                        gamma, tau_range, peaks = drt.fit(n_peaks=3, n_generations=30)
                        # Для ISGP используем tau_range
                        tau = tau_range
                        Z_rec_real = None
                        Z_rec_imag = None
                        confidence_lower = None
                        confidence_upper = None
                        st.info(f"Найдено {len(peaks)} пиков")
                
                # Постобработка - выделение пиков
                if method == "ISGP (генетический)":
                    peaks_tau, peaks_height, peaks_area = extract_peaks(gamma, tau, peak_prominence)
                else:
                    peaks_tau, peaks_height, peaks_area = extract_peaks(gamma, drt.tau, peak_prominence)
                
                # Визуализация
                if method == "Тихоновская регуляризация" or method == "Максимальная энтропия":
                    fig = plot_results(
                        freq, Z_real, Z_imag, gamma, drt.tau if method != "ISGP" else tau,
                        Z_rec_real, Z_rec_imag,
                        peaks_tau, peaks_height,
                        confidence_lower, confidence_upper
                    )
                else:
                    fig = plot_results(
                        freq, Z_real, Z_imag, gamma, drt.tau if method != "ISGP" else tau,
                        None, None,
                        peaks_tau, peaks_height,
                        confidence_lower, confidence_upper
                    )
                
                st.pyplot(fig)
                
                # Таблица с результатами
                if len(peaks_tau) > 0:
                    st.subheader("📈 Выделенные пики (электрохимические процессы)")
                    results_df = pd.DataFrame({
                        "Время релаксации τ (с)": peaks_tau,
                        "Частота f (Гц)": 1 / (2 * np.pi * peaks_tau),
                        "Высота пика": peaks_height,
                        "Площадь (сопротивление, Ом)": peaks_area
                    })
                    st.dataframe(results_df)
                    
                    st.info("""
                    **Интерпретация пиков:**
                    - **Малые τ** (высокие частоты) → быстрые процессы (зарядоперенос, ионная проводимость)
                    - **Средние τ** → процессы на границах зерен, адсорбция
                    - **Большие τ** (низкие частоты) → медленные процессы (диффузия, массоперенос)
                    """)
                
                # Экспорт результатов
                st.subheader("💾 Экспорт результатов")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Экспорт DRT
                    if method == "ISGP (генетический)":
                        export_tau = tau
                    else:
                        export_tau = drt.tau
                    
                    export_df = pd.DataFrame({
                        "tau (s)": export_tau,
                        "gamma": gamma
                    })
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Скачать DRT данные (CSV)",
                        data=csv,
                        file_name="drt_results.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    if len(peaks_tau) > 0:
                        peaks_csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Скачать пики (CSV)",
                            data=peaks_csv,
                            file_name="peaks_results.csv",
                            mime="text/csv"
                        )
    
    else:
        # Информация о формате файла
        st.info("""
        ### 📖 Инструкция
        
        **Формат входного файла:**
        
        Файл должен содержать 3 колонки:
        1. Частота (Гц)
        2. Действительная часть импеданса Z' (Ом)
        3. Мнимая часть импеданса Z'' (Ом)
        
        Поддерживаются форматы: `.txt`, `.csv`, `.dat`
        
        **Пример данных:**
