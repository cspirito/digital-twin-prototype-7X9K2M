import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import warnings
warnings.filterwarnings('ignore')

from digital_twin_core import (
    DigitalTwin, PhysicalSystem, HighFidelitySystemModel,
    PerformanceMetrics
)
from performance_monitor import (
    PerformanceMonitor, TemporalComplianceTracker
)
from anomaly_detection import (
    TemporalAnomalyDetector, PatternRecognitionEngine,
    AdaptiveThresholdDetector
)
from cyber_effect_assessment import ProbabilisticCyberEffectAssessment


class DigitalTwinDemonstration:
    def __init__(self):
        print("Initializing Digital Twin Prototype...")

        self.physical_system = PhysicalSystem("Industrial Control System", inject_anomaly=True)

        baseline_performance = {
            'cpu': (20, 60),
            'memory': (30, 50),
            'throughput': (40, 100),
            'response_time': (50, 150),
            'error_rate': (0, 20)
        }
        self.system_model = HighFidelitySystemModel(baseline_performance)

        self.digital_twin = DigitalTwin(self.physical_system, self.system_model)

        self.performance_monitor = PerformanceMonitor(window_size=50)
        self.temporal_tracker = TemporalComplianceTracker()
        self.anomaly_detector = TemporalAnomalyDetector(sensitivity=0.95)
        self.pattern_engine = PatternRecognitionEngine()
        self.cyber_assessment = ProbabilisticCyberEffectAssessment()

        self.metrics_history = []
        self.assessment_results = []

        print("Digital Twin initialized successfully!")

    def run_simulation(self, duration: int = 200, inject_attack_at: int = 80):
        print(f"\nRunning simulation for {duration} time steps...")
        print(f"Cyber effect will be injected at t={inject_attack_at}")

        for t in range(duration):
            load = 0.3 + 0.4 * np.sin(t * 0.1) + np.random.normal(0, 0.1)
            load = np.clip(load, 0, 1)
            input_state = {'load': load, 'complexity': 1.0}

            if t == inject_attack_at:
                print(f"\n[t={t}] INJECTING CYBER EFFECT: Network degradation attack")
                self.physical_system.inject_cyber_effect('network_degradation', duration=30)
            elif t == inject_attack_at + 50:
                print(f"\n[t={t}] INJECTING CYBER EFFECT: Resource hijacking")
                self.physical_system.inject_cyber_effect('cyber_effect', duration=20)

            physical_metrics, simulated_metrics, divergence = self.digital_twin.update(
                input_state, time_delta=0.1
            )

            metrics_vector = physical_metrics.to_vector()
            self.metrics_history.append(metrics_vector)

            self.performance_monitor.add_observation(metrics_vector)

            compliance = self.performance_monitor.check_compliance(
                metrics_vector, tolerance_factor=2.0
            )

            self.temporal_tracker.update(
                compliance['compliant'],
                compliance.get('mahalanobis_distance', 0)
            )

            if len(self.metrics_history) > 30:
                self.anomaly_detector.train(
                    np.array(self.metrics_history[-30:-10])
                )

            anomaly_result = self.anomaly_detector.detect(metrics_vector)

            temporal_anomaly = self.temporal_tracker.detect_temporal_anomaly(
                threshold=0.7
            )

            pattern_result = self.pattern_engine.analyze_patterns(
                self.metrics_history[-50:]
            )

            if anomaly_result['is_anomaly'] or not compliance['compliant']:
                observations = {
                    'anomaly_score': anomaly_result.get('score', 0),
                    'violations': compliance.get('violations', []),
                    'divergence': divergence
                }

                temporal_context = {
                    'pattern_type': self.anomaly_detector.current_anomaly.pattern_type
                    if self.anomaly_detector.current_anomaly else 'normal',
                    'compliance_rate': temporal_anomaly.get('overall_health', 1.0)
                }

                cyber_assessment = self.cyber_assessment.assess_cyber_effect_probability(
                    observations, temporal_context
                )

                if cyber_assessment['effects']:
                    top_effect = max(
                        cyber_assessment['effects'].items(),
                        key=lambda x: x[1]['probability']
                    )
                    if top_effect[1]['probability'] > 0.5:
                        print(f"[t={t}] ALERT: {top_effect[0]} detected "
                              f"(prob: {top_effect[1]['probability']:.2f}, "
                              f"severity: {top_effect[1]['severity']})")

                self.assessment_results.append({
                    'timestamp': t,
                    'assessment': cyber_assessment,
                    'divergence': divergence
                })

            if t % 20 == 0:
                stats = self.digital_twin.get_recent_statistics()
                if stats:
                    print(f"[t={t}] Divergence: {stats['mean_divergence']:.2f} "
                          f"(±{stats['std_divergence']:.2f}), "
                          f"Trend: {stats['trend']:.3f}")

        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        self._print_final_summary()

    def _print_final_summary(self):
        temporal_summary = self.anomaly_detector.get_temporal_summary()
        print("\n### Temporal Anomaly Summary ###")
        print(f"Total bounded anomaly events: {temporal_summary.get('total_bounded_events', 0)}")
        print(f"Average anomaly duration: {temporal_summary.get('average_duration', 0):.2f} time units")
        print(f"Pattern distribution: {temporal_summary.get('pattern_distribution', {})}")

        if self.assessment_results:
            high_prob_effects = []
            for result in self.assessment_results:
                for effect_name, details in result['assessment']['effects'].items():
                    if details['probability'] > 0.6:
                        high_prob_effects.append({
                            'name': effect_name,
                            'prob': details['probability'],
                            'time': result['timestamp']
                        })

            if high_prob_effects:
                print("\n### Detected Cyber Effects (>60% probability) ###")
                for effect in high_prob_effects[:5]:
                    print(f"- {effect['name']} at t={effect['time']} "
                          f"(probability: {effect['prob']:.2%})")

        root_causes = {}
        for result in self.assessment_results:
            for cause, prob in result['assessment']['root_cause_assessment'].items():
                if cause not in root_causes:
                    root_causes[cause] = []
                root_causes[cause].append(prob)

        if root_causes:
            print("\n### Root Cause Analysis ###")
            for cause, probs in root_causes.items():
                avg_prob = np.mean(probs)
                print(f"- {cause}: {avg_prob:.2%} average probability")

        print("\n### Performance Non-Compliance Summary ###")
        compliance_rates = self.temporal_tracker.get_temporal_compliance_rates()
        for window, rate in compliance_rates.items():
            print(f"- {window}: {rate:.2%} compliance rate")

    def visualize_results(self):
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Digital Twin Performance Analysis', fontsize=16, fontweight='bold')

        metrics_array = np.array(self.metrics_history)
        time_points = range(len(metrics_array))

        metric_names = ['CPU Usage', 'Memory Usage', 'Network Throughput',
                       'Response Time', 'Error Rate']

        ax = axes[0, 0]
        for i, name in enumerate(metric_names):
            ax.plot(time_points, metrics_array[:, i], label=name, alpha=0.7)
        ax.set_title('System Metrics Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Metric Value')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        divergences = [r['divergence'] for r in self.assessment_results]
        timestamps = [r['timestamp'] for r in self.assessment_results]
        ax.plot(timestamps, divergences, 'r-', linewidth=2)
        ax.fill_between(timestamps, divergences, alpha=0.3, color='red')
        ax.set_title('Digital Twin Divergence from Physical System')
        ax.set_xlabel('Time')
        ax.set_ylabel('Divergence')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        anomaly_scores = []
        for result in self.anomaly_detector.anomaly_buffer:
            anomaly_scores.append(result.get('score', 0))
        if anomaly_scores:
            ax.plot(anomaly_scores, 'b-', alpha=0.7)
            threshold = np.percentile(anomaly_scores, 95)
            ax.axhline(y=threshold, color='r', linestyle='--', label=f'95th percentile')
            ax.set_title('Anomaly Detection Scores')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Anomaly Score')
            ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        effect_probs = {}
        for result in self.assessment_results:
            for effect_name, details in result['assessment']['effects'].items():
                if effect_name not in effect_probs:
                    effect_probs[effect_name] = []
                effect_probs[effect_name].append(details['probability'])

        for effect_name, probs in effect_probs.items():
            ax.plot(probs[:100], label=effect_name, alpha=0.7)
        ax.set_title('Cyber Effect Probability Assessment')
        ax.set_xlabel('Assessment Point')
        ax.set_ylabel('Probability')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2, 0]
        compliance_rates = self.temporal_tracker.get_temporal_compliance_rates()
        if compliance_rates:
            windows = list(compliance_rates.keys())
            rates = list(compliance_rates.values())
            colors = ['green' if r > 0.8 else 'orange' if r > 0.5 else 'red' for r in rates]
            ax.bar(windows, rates, color=colors)
            ax.set_title('Temporal Compliance Rates')
            ax.set_xlabel('Time Window')
            ax.set_ylabel('Compliance Rate')
            ax.set_ylim(0, 1.1)
            ax.axhline(y=0.7, color='black', linestyle='--', alpha=0.5, label='Threshold')
            ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        if self.assessment_results:
            root_cause_summary = {}
            for result in self.assessment_results[-50:]:
                max_cause = max(
                    result['assessment']['root_cause_assessment'].items(),
                    key=lambda x: x[1]
                )
                cause_name = max_cause[0]
                if cause_name not in root_cause_summary:
                    root_cause_summary[cause_name] = 0
                root_cause_summary[cause_name] += 1

            if root_cause_summary:
                causes = list(root_cause_summary.keys())
                counts = list(root_cause_summary.values())
                colors_map = {
                    'cyber_attack': 'red',
                    'natural_degradation': 'yellow',
                    'system_failure': 'orange'
                }
                colors = [colors_map.get(c, 'gray') for c in causes]
                ax.pie(counts, labels=causes, colors=colors, autopct='%1.1f%%')
                ax.set_title('Root Cause Distribution (Last 50 Assessments)')

        plt.tight_layout()
        plt.show()


def main():
    print("="*60)
    print("DIGITAL TWIN PROTOTYPE - Performance Non-Compliance Detection")
    print("with Probabilistic Cyber Effect Assessment")
    print("="*60)

    demo = DigitalTwinDemonstration()

    demo.run_simulation(duration=200, inject_attack_at=80)

    print("\nGenerating visualization...")
    demo.visualize_results()

    system_state = {'cpu': 50, 'memory': 40, 'throughput': 80,
                   'response_time': 60, 'error_rate': 5}

    print("\n### Simulating Consequence Projection ###")
    for effect_name, effect in demo.cyber_assessment.effect_library.items():
        if effect.probability > 0.4:
            consequences = demo.cyber_assessment.simulate_consequences(
                effect, system_state, time_horizon=50
            )
            print(f"\n{effect_name.upper()} Consequences:")
            print(f"  Peak Impact: {consequences['peak_impact']:.2f}")
            print(f"  Recovery Time: {consequences['recovery_time']} time units")
            print(f"  Severity: {consequences['severity_assessment']}")


if __name__ == "__main__":
    main()