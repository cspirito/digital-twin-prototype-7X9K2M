import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import time
from abc import ABC, abstractmethod


@dataclass
class PerformanceMetrics:
    timestamp: float
    cpu_usage: float
    memory_usage: float
    network_throughput: float
    response_time: float
    error_rate: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        base_metrics = [
            self.cpu_usage,
            self.memory_usage,
            self.network_throughput,
            self.response_time,
            self.error_rate
        ]
        custom_values = list(self.custom_metrics.values())
        return np.array(base_metrics + custom_values)


class SystemModel(ABC):
    @abstractmethod
    def simulate(self, input_state: Dict[str, Any], time_delta: float) -> PerformanceMetrics:
        pass

    @abstractmethod
    def get_expected_bounds(self, input_state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        pass


class HighFidelitySystemModel(SystemModel):
    def __init__(self, baseline_performance: Dict[str, Tuple[float, float]]):
        self.baseline = baseline_performance
        self.state_history = deque(maxlen=100)
        self.degradation_factor = 1.0

    def simulate(self, input_state: Dict[str, Any], time_delta: float) -> PerformanceMetrics:
        load_factor = input_state.get('load', 1.0)
        complexity = input_state.get('complexity', 1.0)

        base_cpu = 20 + (load_factor * 40) + np.random.normal(0, 2)
        base_memory = 30 + (load_factor * 20) + np.random.normal(0, 1.5)
        base_throughput = max(0, 100 - (load_factor * 30) + np.random.normal(0, 5))
        base_response = 50 + (load_factor * complexity * 100) + np.random.normal(0, 10)
        base_error = max(0, min(100, (load_factor - 0.8) * 50 + np.random.normal(0, 2)))

        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=base_cpu * self.degradation_factor,
            memory_usage=base_memory * self.degradation_factor,
            network_throughput=base_throughput / self.degradation_factor,
            response_time=base_response * self.degradation_factor,
            error_rate=base_error * self.degradation_factor
        )

        self.state_history.append(metrics)
        return metrics

    def get_expected_bounds(self, input_state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        load_factor = input_state.get('load', 1.0)

        lower_bounds = np.array([
            10 + (load_factor * 30),  # CPU
            25 + (load_factor * 15),  # Memory
            max(0, 70 - (load_factor * 40)),  # Throughput
            40 + (load_factor * 80),  # Response time
            max(0, (load_factor - 0.9) * 40)  # Error rate
        ])

        upper_bounds = np.array([
            30 + (load_factor * 50),  # CPU
            35 + (load_factor * 25),  # Memory
            min(100, 130 - (load_factor * 20)),  # Throughput
            60 + (load_factor * 120),  # Response time
            min(100, (load_factor - 0.7) * 60)  # Error rate
        ])

        return lower_bounds, upper_bounds


class PhysicalSystem:
    def __init__(self, name: str, inject_anomaly: bool = False):
        self.name = name
        self.inject_anomaly = inject_anomaly
        self.anomaly_start = None
        self.anomaly_duration = None
        self.anomaly_type = None

    def get_current_metrics(self, input_state: Dict[str, Any]) -> PerformanceMetrics:
        load_factor = input_state.get('load', 1.0)
        current_time = time.time()

        base_cpu = 22 + (load_factor * 38) + np.random.normal(0, 3)
        base_memory = 32 + (load_factor * 18) + np.random.normal(0, 2)
        base_throughput = max(0, 95 - (load_factor * 28) + np.random.normal(0, 6))
        base_response = 55 + (load_factor * 95) + np.random.normal(0, 12)
        base_error = max(0, min(100, (load_factor - 0.75) * 45 + np.random.normal(0, 3)))

        if self.inject_anomaly and self.anomaly_start:
            if current_time - self.anomaly_start < self.anomaly_duration:
                if self.anomaly_type == 'cpu_spike':
                    base_cpu = min(100, base_cpu * 2.5)
                elif self.anomaly_type == 'memory_leak':
                    elapsed = current_time - self.anomaly_start
                    base_memory = min(100, base_memory + (elapsed * 5))
                elif self.anomaly_type == 'network_degradation':
                    base_throughput = max(0, base_throughput * 0.3)
                    base_response = base_response * 3
                elif self.anomaly_type == 'cyber_effect':
                    base_cpu = min(100, base_cpu * 1.8)
                    base_response = base_response * 2.5
                    base_error = min(100, base_error + 25)

        return PerformanceMetrics(
            timestamp=current_time,
            cpu_usage=base_cpu,
            memory_usage=base_memory,
            network_throughput=base_throughput,
            response_time=base_response,
            error_rate=base_error
        )

    def inject_cyber_effect(self, effect_type: str, duration: float):
        self.anomaly_start = time.time()
        self.anomaly_duration = duration
        self.anomaly_type = effect_type


class DigitalTwin:
    def __init__(self, physical_system: PhysicalSystem, system_model: SystemModel):
        self.physical_system = physical_system
        self.system_model = system_model
        self.performance_history = deque(maxlen=1000)
        self.simulation_history = deque(maxlen=1000)
        self.divergence_history = deque(maxlen=1000)

    def update(self, input_state: Dict[str, Any], time_delta: float = 0.1):
        physical_metrics = self.physical_system.get_current_metrics(input_state)
        simulated_metrics = self.system_model.simulate(input_state, time_delta)

        self.performance_history.append(physical_metrics)
        self.simulation_history.append(simulated_metrics)

        divergence = self._calculate_divergence(physical_metrics, simulated_metrics)
        self.divergence_history.append({
            'timestamp': time.time(),
            'divergence': divergence,
            'physical': physical_metrics,
            'simulated': simulated_metrics
        })

        return physical_metrics, simulated_metrics, divergence

    def _calculate_divergence(self, physical: PerformanceMetrics,
                             simulated: PerformanceMetrics) -> float:
        physical_vec = physical.to_vector()
        simulated_vec = simulated.to_vector()

        return np.linalg.norm(physical_vec - simulated_vec)

    def get_recent_statistics(self, window_size: int = 50) -> Dict[str, Any]:
        if len(self.divergence_history) < window_size:
            return {}

        recent_divergences = [d['divergence'] for d in list(self.divergence_history)[-window_size:]]

        return {
            'mean_divergence': np.mean(recent_divergences),
            'std_divergence': np.std(recent_divergences),
            'max_divergence': np.max(recent_divergences),
            'min_divergence': np.min(recent_divergences),
            'trend': np.polyfit(range(len(recent_divergences)), recent_divergences, 1)[0]
        }