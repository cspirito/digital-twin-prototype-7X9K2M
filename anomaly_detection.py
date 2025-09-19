import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import time


@dataclass
class AnomalyEvent:
    timestamp: float
    duration: float
    severity: str
    confidence: float
    affected_metrics: List[str]
    pattern_type: str
    is_bounded: bool


class TemporalAnomalyDetector:
    def __init__(self, sensitivity: float = 0.95):
        self.sensitivity = sensitivity
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=1 - sensitivity,
            random_state=42
        )
        self.anomaly_buffer = deque(maxlen=1000)
        self.pattern_buffer = deque(maxlen=100)
        self.is_trained = False
        self.anomaly_events: List[AnomalyEvent] = []
        self.current_anomaly: Optional[AnomalyEvent] = None

    def train(self, normal_data: np.ndarray):
        if len(normal_data) < 10:
            return

        scaled_data = self.scaler.fit_transform(normal_data)
        self.isolation_forest.fit(scaled_data)
        self.is_trained = True

    def detect(self, observation: np.ndarray) -> Dict[str, any]:
        if not self.is_trained:
            return {'is_anomaly': False, 'confidence': 0.0}

        scaled_obs = self.scaler.transform(observation.reshape(1, -1))

        anomaly_score = self.isolation_forest.decision_function(scaled_obs)[0]
        prediction = self.isolation_forest.predict(scaled_obs)[0]

        is_anomaly = prediction == -1

        confidence = 1 / (1 + np.exp(anomaly_score))

        result = {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'score': anomaly_score,
            'timestamp': time.time()
        }

        self.anomaly_buffer.append(result)
        self._update_temporal_patterns(result)

        return result

    def _update_temporal_patterns(self, detection_result: Dict):
        if detection_result['is_anomaly']:
            if self.current_anomaly is None:
                self.current_anomaly = AnomalyEvent(
                    timestamp=detection_result['timestamp'],
                    duration=0,
                    severity='low',
                    confidence=detection_result['confidence'],
                    affected_metrics=[],
                    pattern_type='point',
                    is_bounded=False
                )
            else:
                self.current_anomaly.duration = (
                    detection_result['timestamp'] - self.current_anomaly.timestamp
                )
                self.current_anomaly.confidence = max(
                    self.current_anomaly.confidence,
                    detection_result['confidence']
                )
                self._classify_anomaly_pattern()
        else:
            if self.current_anomaly is not None:
                self.current_anomaly.is_bounded = True
                self.anomaly_events.append(self.current_anomaly)
                self.current_anomaly = None

    def _classify_anomaly_pattern(self):
        if self.current_anomaly is None:
            return

        duration = self.current_anomaly.duration

        if duration < 1:
            self.current_anomaly.pattern_type = 'spike'
            self.current_anomaly.severity = 'low'
        elif duration < 5:
            self.current_anomaly.pattern_type = 'transient'
            self.current_anomaly.severity = 'medium'
        elif duration < 20:
            self.current_anomaly.pattern_type = 'sustained'
            self.current_anomaly.severity = 'high'
        else:
            self.current_anomaly.pattern_type = 'persistent'
            self.current_anomaly.severity = 'critical'

    def get_temporal_summary(self) -> Dict[str, any]:
        if len(self.anomaly_buffer) == 0:
            return {}

        recent_anomalies = [x for x in self.anomaly_buffer if x['is_anomaly']]
        anomaly_rate = len(recent_anomalies) / len(self.anomaly_buffer)

        bounded_events = [e for e in self.anomaly_events if e.is_bounded]

        if bounded_events:
            avg_duration = np.mean([e.duration for e in bounded_events])
            pattern_distribution = {}
            for event in bounded_events:
                pattern_distribution[event.pattern_type] = (
                    pattern_distribution.get(event.pattern_type, 0) + 1
                )
        else:
            avg_duration = 0
            pattern_distribution = {}

        return {
            'anomaly_rate': anomaly_rate,
            'total_bounded_events': len(bounded_events),
            'average_duration': avg_duration,
            'pattern_distribution': pattern_distribution,
            'current_state': 'anomalous' if self.current_anomaly else 'normal'
        }


class AdaptiveThresholdDetector:
    def __init__(self, initial_threshold: float = 2.0, adaptation_rate: float = 0.1):
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.performance_buffer = deque(maxlen=100)
        self.false_positive_rate = 0.0
        self.false_negative_estimate = 0.0

    def detect_deviation(self, value: float, baseline: float, std: float) -> Tuple[bool, float]:
        z_score = abs((value - baseline) / (std + 1e-6))

        is_anomaly = z_score > self.threshold

        confidence = 1 - np.exp(-z_score / self.threshold)

        self.performance_buffer.append({
            'z_score': z_score,
            'detected': is_anomaly,
            'confidence': confidence
        })

        self._adapt_threshold()

        return is_anomaly, confidence

    def _adapt_threshold(self):
        if len(self.performance_buffer) < 50:
            return

        recent_scores = [x['z_score'] for x in self.performance_buffer]
        recent_detections = [x['detected'] for x in self.performance_buffer]

        score_variance = np.var(recent_scores)
        detection_rate = sum(recent_detections) / len(recent_detections)

        if detection_rate > 0.1:  # Too many detections
            self.threshold *= (1 + self.adaptation_rate)
        elif detection_rate < 0.01:  # Too few detections
            self.threshold *= (1 - self.adaptation_rate)

        self.threshold = np.clip(self.threshold, 1.5, 4.0)


class PatternRecognitionEngine:
    def __init__(self):
        self.known_patterns = {
            'resource_exhaustion': self._detect_resource_exhaustion,
            'performance_degradation': self._detect_performance_degradation,
            'intermittent_failure': self._detect_intermittent_failure,
            'cascade_effect': self._detect_cascade_effect
        }
        self.pattern_history = deque(maxlen=500)

    def analyze_patterns(self, metrics_history: List[np.ndarray]) -> Dict[str, any]:
        if len(metrics_history) < 10:
            return {'patterns': [], 'confidence': {}}

        detected_patterns = []
        confidence_scores = {}

        for pattern_name, detector_func in self.known_patterns.items():
            is_detected, confidence = detector_func(metrics_history)
            if is_detected:
                detected_patterns.append(pattern_name)
                confidence_scores[pattern_name] = confidence

        result = {
            'patterns': detected_patterns,
            'confidence': confidence_scores,
            'timestamp': time.time()
        }

        self.pattern_history.append(result)
        return result

    def _detect_resource_exhaustion(self, history: List[np.ndarray]) -> Tuple[bool, float]:
        if len(history) < 5:
            return False, 0.0

        cpu_memory_idx = [0, 1]  # CPU and Memory indices
        recent_values = np.array(history[-10:])[:, cpu_memory_idx]

        trend = np.mean(np.diff(recent_values, axis=0), axis=0)

        if np.all(trend > 0.5):  # Both increasing
            current = recent_values[-1]
            if np.any(current > 80):  # High utilization
                return True, min(1.0, np.mean(current) / 100)

        return False, 0.0

    def _detect_performance_degradation(self, history: List[np.ndarray]) -> Tuple[bool, float]:
        if len(history) < 10:
            return False, 0.0

        response_time_idx = 3
        throughput_idx = 2

        recent = np.array(history[-20:])
        response_times = recent[:, response_time_idx]
        throughput = recent[:, throughput_idx]

        response_trend = np.polyfit(range(len(response_times)), response_times, 1)[0]
        throughput_trend = np.polyfit(range(len(throughput)), throughput, 1)[0]

        if response_trend > 1 and throughput_trend < -1:
            confidence = min(1.0, (abs(response_trend) + abs(throughput_trend)) / 10)
            return True, confidence

        return False, 0.0

    def _detect_intermittent_failure(self, history: List[np.ndarray]) -> Tuple[bool, float]:
        if len(history) < 20:
            return False, 0.0

        error_rate_idx = 4
        error_rates = np.array(history[-30:])[:, error_rate_idx]

        variance = np.var(error_rates)
        spikes = np.sum(error_rates > 20)

        if variance > 50 and spikes > 3:
            return True, min(1.0, spikes / 10)

        return False, 0.0

    def _detect_cascade_effect(self, history: List[np.ndarray]) -> Tuple[bool, float]:
        if len(history) < 15:
            return False, 0.0

        recent = np.array(history[-15:])

        correlations = np.corrcoef(recent.T)
        high_correlations = np.sum(np.abs(correlations) > 0.8) - len(correlations)

        multi_metric_degradation = 0
        for i in range(recent.shape[1]):
            trend = np.polyfit(range(len(recent)), recent[:, i], 1)[0]
            if abs(trend) > 1:
                multi_metric_degradation += 1

        if high_correlations > 5 and multi_metric_degradation > 3:
            confidence = min(1.0, multi_metric_degradation / 5)
            return True, confidence

        return False, 0.0