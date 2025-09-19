import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class CyberEffect:
    name: str
    probability: float
    confidence: float
    impact_metrics: List[str]
    expected_signature: Dict[str, float]
    temporal_pattern: str
    severity: str


@dataclass
class BayesianNode:
    name: str
    prior: float
    likelihood_given_evidence: Dict[str, float] = field(default_factory=dict)
    posterior: float = 0.0


class ProbabilisticCyberEffectAssessment:
    def __init__(self):
        self.effect_library = self._initialize_effect_library()
        self.bayesian_network = self._initialize_bayesian_network()
        self.evidence_buffer = defaultdict(list)
        self.assessment_history = []

    def _initialize_effect_library(self) -> Dict[str, CyberEffect]:
        return {
            'denial_of_service': CyberEffect(
                name='denial_of_service',
                probability=0.0,
                confidence=0.0,
                impact_metrics=['cpu', 'memory', 'response_time'],
                expected_signature={
                    'cpu_spike': 0.8,
                    'memory_spike': 0.6,
                    'response_degradation': 0.9,
                    'throughput_drop': 0.9
                },
                temporal_pattern='sustained',
                severity='high'
            ),
            'resource_hijacking': CyberEffect(
                name='resource_hijacking',
                probability=0.0,
                confidence=0.0,
                impact_metrics=['cpu', 'network'],
                expected_signature={
                    'cpu_spike': 0.9,
                    'network_anomaly': 0.7,
                    'irregular_pattern': 0.8
                },
                temporal_pattern='persistent',
                severity='critical'
            ),
            'data_manipulation': CyberEffect(
                name='data_manipulation',
                probability=0.0,
                confidence=0.0,
                impact_metrics=['error_rate', 'response_time'],
                expected_signature={
                    'error_spike': 0.7,
                    'inconsistent_responses': 0.8,
                    'validation_failures': 0.9
                },
                temporal_pattern='intermittent',
                severity='high'
            ),
            'performance_degradation': CyberEffect(
                name='performance_degradation',
                probability=0.0,
                confidence=0.0,
                impact_metrics=['response_time', 'throughput'],
                expected_signature={
                    'gradual_slowdown': 0.8,
                    'throughput_decline': 0.7,
                    'queue_buildup': 0.6
                },
                temporal_pattern='gradual',
                severity='medium'
            ),
            'control_flow_hijacking': CyberEffect(
                name='control_flow_hijacking',
                probability=0.0,
                confidence=0.0,
                impact_metrics=['cpu', 'memory', 'error_rate'],
                expected_signature={
                    'unexpected_behavior': 0.9,
                    'irregular_execution': 0.8,
                    'memory_corruption': 0.6
                },
                temporal_pattern='spike',
                severity='critical'
            )
        }

    def _initialize_bayesian_network(self) -> Dict[str, BayesianNode]:
        nodes = {
            'cyber_attack': BayesianNode(
                name='cyber_attack',
                prior=0.1,
                likelihood_given_evidence={
                    'anomaly_detected': 0.7,
                    'pattern_match': 0.8,
                    'temporal_correlation': 0.6,
                    'multi_metric_deviation': 0.75
                }
            ),
            'natural_degradation': BayesianNode(
                name='natural_degradation',
                prior=0.3,
                likelihood_given_evidence={
                    'gradual_change': 0.8,
                    'single_metric': 0.7,
                    'predictable_pattern': 0.9,
                    'no_correlation': 0.6
                }
            ),
            'system_failure': BayesianNode(
                name='system_failure',
                prior=0.2,
                likelihood_given_evidence={
                    'sudden_change': 0.8,
                    'cascade_effect': 0.9,
                    'complete_failure': 0.95,
                    'recovery_pattern': 0.4
                }
            )
        }
        return nodes

    def assess_cyber_effect_probability(self,
                                       observations: Dict[str, any],
                                       temporal_context: Dict[str, any]) -> Dict[str, any]:
        evidence = self._extract_evidence(observations, temporal_context)

        for effect_name, effect in self.effect_library.items():
            probability, confidence = self._calculate_effect_probability(
                effect, evidence, observations
            )
            effect.probability = probability
            effect.confidence = confidence

        root_cause = self._bayesian_inference(evidence)

        assessment = {
            'timestamp': time.time(),
            'effects': {
                name: {
                    'probability': effect.probability,
                    'confidence': effect.confidence,
                    'severity': effect.severity,
                    'pattern': effect.temporal_pattern
                }
                for name, effect in self.effect_library.items()
                if effect.probability > 0.3
            },
            'root_cause_assessment': root_cause,
            'confidence_level': self._calculate_overall_confidence(evidence),
            'recommended_actions': self._generate_recommendations(root_cause)
        }

        self.assessment_history.append(assessment)
        return assessment

    def _extract_evidence(self, observations: Dict, temporal: Dict) -> Dict[str, float]:
        evidence = {}

        if 'anomaly_score' in observations:
            evidence['anomaly_detected'] = min(1.0, observations['anomaly_score'] / 5)

        if 'pattern_type' in temporal:
            pattern = temporal['pattern_type']
            if pattern in ['sustained', 'persistent']:
                evidence['temporal_correlation'] = 0.8
            elif pattern == 'spike':
                evidence['sudden_change'] = 0.9
            else:
                evidence['gradual_change'] = 0.7

        if 'violations' in observations:
            violations = observations.get('violations', [])
            if len(violations) > 3:
                evidence['multi_metric_deviation'] = min(1.0, len(violations) / 5)
            elif len(violations) == 1:
                evidence['single_metric'] = 0.8

        if 'compliance_rate' in temporal:
            rate = temporal['compliance_rate']
            if rate < 0.3:
                evidence['complete_failure'] = 0.9
            elif rate < 0.6:
                evidence['pattern_match'] = 0.7

        return evidence

    def _calculate_effect_probability(self, effect: CyberEffect,
                                     evidence: Dict[str, float],
                                     observations: Dict) -> Tuple[float, float]:
        signature_matches = 0
        total_signatures = len(effect.expected_signature)

        for signature, expected_prob in effect.expected_signature.items():
            if signature in evidence:
                if evidence[signature] >= expected_prob * 0.7:
                    signature_matches += 1

        base_probability = signature_matches / total_signatures if total_signatures > 0 else 0

        if 'divergence' in observations:
            divergence_factor = min(1.0, observations['divergence'] / 100)
            base_probability *= (1 + divergence_factor * 0.5)

        temporal_modifier = 1.0
        if 'temporal_pattern' in observations:
            if observations['temporal_pattern'] == effect.temporal_pattern:
                temporal_modifier = 1.3

        probability = min(1.0, base_probability * temporal_modifier)

        confidence = min(1.0, signature_matches / total_signatures * 0.8 +
                        len(evidence) / 10 * 0.2)

        return probability, confidence

    def _bayesian_inference(self, evidence: Dict[str, float]) -> Dict[str, float]:
        posteriors = {}

        for node_name, node in self.bayesian_network.items():
            likelihood = 1.0
            for evidence_type, evidence_strength in evidence.items():
                if evidence_type in node.likelihood_given_evidence:
                    likelihood *= (
                        node.likelihood_given_evidence[evidence_type] * evidence_strength +
                        (1 - node.likelihood_given_evidence[evidence_type]) * (1 - evidence_strength)
                    )

            node.posterior = (node.prior * likelihood) / (
                node.prior * likelihood + (1 - node.prior) * (1 - likelihood) + 1e-10
            )
            posteriors[node_name] = node.posterior

        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v/total for k, v in posteriors.items()}

        return posteriors

    def _calculate_overall_confidence(self, evidence: Dict[str, float]) -> float:
        if not evidence:
            return 0.0

        evidence_strength = np.mean(list(evidence.values()))

        evidence_count_factor = min(1.0, len(evidence) / 5)

        return evidence_strength * 0.7 + evidence_count_factor * 0.3

    def _generate_recommendations(self, root_cause: Dict[str, float]) -> List[str]:
        recommendations = []

        max_cause = max(root_cause.items(), key=lambda x: x[1])
        cause_name, probability = max_cause

        if probability < 0.3:
            recommendations.append("Continue monitoring - no clear threat detected")
            recommendations.append("Maintain current baselines and thresholds")
        elif cause_name == 'cyber_attack' and probability > 0.6:
            recommendations.append("ALERT: High probability of cyber attack")
            recommendations.append("Initiate incident response procedures")
            recommendations.append("Isolate affected components if possible")
            recommendations.append("Collect forensic data for analysis")
            recommendations.append("Review recent system changes and access logs")
        elif cause_name == 'system_failure' and probability > 0.5:
            recommendations.append("System failure likely - initiate recovery procedures")
            recommendations.append("Check hardware health and resource availability")
            recommendations.append("Review system logs for error patterns")
            recommendations.append("Consider failover to backup systems")
        elif cause_name == 'natural_degradation' and probability > 0.5:
            recommendations.append("Natural performance degradation detected")
            recommendations.append("Schedule maintenance window")
            recommendations.append("Review capacity planning")
            recommendations.append("Consider resource scaling")
        else:
            recommendations.append("Anomaly detected - investigate further")
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Review correlation with external events")

        return recommendations

    def simulate_consequences(self, effect: CyberEffect,
                            system_state: Dict[str, float],
                            time_horizon: int = 100) -> Dict[str, any]:
        projected_states = []
        current_state = system_state.copy()

        for t in range(time_horizon):
            impact_factor = 1.0 + (effect.probability * 0.5)

            for metric in effect.impact_metrics:
                if metric in current_state:
                    if effect.temporal_pattern == 'gradual':
                        current_state[metric] *= (1 + 0.01 * impact_factor)
                    elif effect.temporal_pattern == 'sustained':
                        current_state[metric] *= impact_factor
                    elif effect.temporal_pattern == 'spike':
                        if t < 10:
                            current_state[metric] *= (impact_factor * 2)
                        else:
                            current_state[metric] /= 1.1

                    current_state[metric] = np.clip(current_state[metric], 0, 100)

            projected_states.append(current_state.copy())

        return {
            'effect_name': effect.name,
            'projected_timeline': projected_states,
            'peak_impact': max([max(state.values()) for state in projected_states]),
            'recovery_time': self._estimate_recovery_time(projected_states),
            'severity_assessment': effect.severity
        }

    def _estimate_recovery_time(self, states: List[Dict[str, float]]) -> int:
        if not states:
            return 0

        baseline = np.mean([list(state.values()) for state in states[:10]], axis=0)

        for i, state in enumerate(states[10:], 10):
            current = np.array(list(state.values()))
            if np.all(np.abs(current - baseline) < baseline * 0.1):
                return i

        return len(states)