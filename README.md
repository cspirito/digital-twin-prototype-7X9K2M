# Digital Twin Prototype for Performance Non-Compliance Detection

## Concept

This prototype demonstrates a digital twin system designed to detect performance non-compliance rather than simple binary performance degradation. The system treats anomalies as temporally bounded phenomena with probabilistic characteristics, enabling sophisticated assessment of cyber effects on physical systems.

The core insight is that performance degradation exists on a continuum - systems don't simply flip from "working" to "broken" but exhibit temporal patterns of non-compliance that can be statistically characterized. By maintaining high-fidelity simulation models alongside real-world system monitoring, the digital twin can probabilistically assess whether observed behaviors are consequences of cyber effects, natural degradation, or system failures.

The framework establishes statistical boundaries around expected performance, then uses Bayesian inference to evaluate the likelihood that deviations represent cyber attacks versus other causes. This approach enables defenders to make informed decisions about incident response based on probabilistic assessments rather than binary alerts.

## Core Components

1. **digital_twin_core.py** - Digital twin framework with physical system simulation and high-fidelity models
2. **performance_monitor.py** - Statistical baseline tracking and temporal compliance monitoring
3. **anomaly_detection.py** - Temporal anomaly detection with pattern recognition and adaptive thresholds
4. **cyber_effect_assessment.py** - Bayesian probabilistic assessment of cyber effects with consequence simulation
5. **demonstration.py** - Full demonstration with visualization

## Key Features

- **Performance non-compliance detection** using statistical baselines and Mahalanobis distance
- **Temporal bounded anomaly detection** (not just binary states) with pattern classification
- **Probabilistic cyber effect assessment** using Bayesian inference
- **Consequence simulation** for different attack scenarios
- **Real-time divergence monitoring** between physical and digital twin
- **Multi-window temporal compliance tracking**
- **Adaptive thresholding** for reducing false positives

## Installation & Usage

```bash
pip install -r requirements.txt
python demonstration.py
```

The demonstration simulates 200 time steps with cyber effects injected at specific points, showing how the system detects and assesses these anomalies probabilistically within temporal boundaries.

## How It Works

1. **Baseline Establishment**: The system continuously monitors performance metrics and establishes statistical baselines for normal operation
2. **Divergence Detection**: Real-time comparison between physical system behavior and digital twin predictions identifies divergences
3. **Temporal Analysis**: Anomalies are classified by temporal patterns (spike, transient, sustained, persistent) rather than binary states
4. **Probabilistic Assessment**: Bayesian networks evaluate the probability of cyber effects versus natural causes
5. **Consequence Projection**: The system simulates potential future states under different effect scenarios

## Output

The demonstration provides:
- Real-time alerts for detected cyber effects with probability scores
- Temporal compliance rates across multiple time windows
- Root cause analysis distinguishing between cyber attacks, system failures, and natural degradation
- Visualizations of system metrics, divergence patterns, and probability assessments