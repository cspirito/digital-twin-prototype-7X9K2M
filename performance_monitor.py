import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StatisticalBaseline:
    mean: np.ndarray
    std: np.ndarray
    percentile_25: np.ndarray
    percentile_75: np.ndarray
    median: np.ndarray
    sample_size: int
    confidence_interval: Tuple[np.ndarray, np.ndarray]


class PerformanceMonitor:
    def __init__(self, window_size: int = 100, confidence_level: float = 0.95):
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.metrics_buffer = deque(maxlen=window_size)
        self.baseline: Optional[StatisticalBaseline] = None
        self.anomaly_scores = deque(maxlen=window_size)
        self.compliance_history = deque(maxlen=window_size * 10)

    def add_observation(self, metrics_vector: np.ndarray):
        self.metrics_buffer.append(metrics_vector)

        if len(self.metrics_buffer) >= self.window_size // 2:
            self.update_baseline()

    def update_baseline(self):
        if len(self.metrics_buffer) < 10:
            return

        data = np.array(self.metrics_buffer)

        z = stats.norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = np.mean(data, axis=0) - z * (np.std(data, axis=0) / np.sqrt(len(data)))
        ci_upper = np.mean(data, axis=0) + z * (np.std(data, axis=0) / np.sqrt(len(data)))

        self.baseline = StatisticalBaseline(
            mean=np.mean(data, axis=0),
            std=np.std(data, axis=0),
            percentile_25=np.percentile(data, 25, axis=0),
            percentile_75=np.percentile(data, 75, axis=0),
            median=np.median(data, axis=0),
            sample_size=len(data),
            confidence_interval=(ci_lower, ci_upper)
        )

    def calculate_mahalanobis_distance(self, observation: np.ndarray) -> float:
        if self.baseline is None or len(self.metrics_buffer) < 10:
            return 0.0

        data = np.array(self.metrics_buffer)
        mean = self.baseline.mean

        try:
            cov = np.cov(data, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-6

            diff = observation - mean
            inv_cov = np.linalg.pinv(cov)
            distance = np.sqrt(diff.T @ inv_cov @ diff)

            return distance
        except:
            return np.linalg.norm(observation - mean)

    def check_compliance(self, observation: np.ndarray,
                        tolerance_factor: float = 2.0) -> Dict[str, any]:
        if self.baseline is None:
            return {'compliant': True, 'score': 0.0, 'violations': []}

        violations = []
        scores = []

        for i, value in enumerate(observation):
            mean = self.baseline.mean[i]
            std = self.baseline.std[i]

            z_score = abs((value - mean) / (std + 1e-6))
            scores.append(z_score)

            if z_score > tolerance_factor:
                violations.append({
                    'metric_index': i,
                    'value': value,
                    'expected_mean': mean,
                    'std_deviations': z_score,
                    'severity': self._classify_severity(z_score)
                })

        mahalanobis = self.calculate_mahalanobis_distance(observation)
        self.anomaly_scores.append(mahalanobis)

        compliance_result = {
            'compliant': len(violations) == 0,
            'score': np.mean(scores),
            'mahalanobis_distance': mahalanobis,
            'violations': violations,
            'timestamp': np.datetime64('now')
        }

        self.compliance_history.append(compliance_result)
        return compliance_result

    def _classify_severity(self, z_score: float) -> str:
        if z_score < 2:
            return 'normal'
        elif z_score < 3:
            return 'warning'
        elif z_score < 4:
            return 'critical'
        else:
            return 'severe'

    def get_performance_trends(self, lookback: int = 50) -> Dict[str, any]:
        if len(self.metrics_buffer) < lookback:
            return {}

        recent_data = list(self.metrics_buffer)[-lookback:]
        data_array = np.array(recent_data)

        trends = {}
        for i in range(data_array.shape[1]):
            metric_values = data_array[:, i]
            x = np.arange(len(metric_values))

            slope, intercept = np.polyfit(x, metric_values, 1)
            r_value, p_value = stats.pearsonr(x, metric_values)

            trends[f'metric_{i}'] = {
                'slope': slope,
                'direction': 'increasing' if slope > 0 else 'decreasing',
                'strength': abs(r_value),
                'significance': p_value < 0.05,
                'current_value': metric_values[-1],
                'mean': np.mean(metric_values),
                'volatility': np.std(metric_values)
            }

        return trends


class TemporalComplianceTracker:
    def __init__(self, time_windows: List[int] = None):
        self.time_windows = time_windows or [10, 50, 100, 500]
        self.compliance_buffers = {
            window: deque(maxlen=window)
            for window in self.time_windows
        }
        self.non_compliance_events = []

    def update(self, is_compliant: bool, anomaly_score: float):
        for window in self.time_windows:
            self.compliance_buffers[window].append({
                'compliant': is_compliant,
                'score': anomaly_score,
                'timestamp': np.datetime64('now')
            })

    def get_temporal_compliance_rates(self) -> Dict[str, float]:
        rates = {}
        for window, buffer in self.compliance_buffers.items():
            if len(buffer) > 0:
                compliance_rate = sum(1 for x in buffer if x['compliant']) / len(buffer)
                rates[f'window_{window}'] = compliance_rate

        return rates

    def detect_temporal_anomaly(self, threshold: float = 0.7) -> Dict[str, any]:
        rates = self.get_temporal_compliance_rates()

        anomalies = []
        for window_name, rate in rates.items():
            if rate < threshold:
                anomalies.append({
                    'window': window_name,
                    'compliance_rate': rate,
                    'severity': self._classify_temporal_severity(rate)
                })

        is_temporal_anomaly = len(anomalies) > 0

        if is_temporal_anomaly:
            self.non_compliance_events.append({
                'timestamp': np.datetime64('now'),
                'windows_affected': anomalies,
                'min_compliance_rate': min(rates.values()) if rates else 1.0
            })

        return {
            'is_anomaly': is_temporal_anomaly,
            'affected_windows': anomalies,
            'overall_health': np.mean(list(rates.values())) if rates else 1.0
        }

    def _classify_temporal_severity(self, compliance_rate: float) -> str:
        if compliance_rate > 0.8:
            return 'normal'
        elif compliance_rate > 0.6:
            return 'degraded'
        elif compliance_rate > 0.4:
            return 'warning'
        elif compliance_rate > 0.2:
            return 'critical'
        else:
            return 'failed'