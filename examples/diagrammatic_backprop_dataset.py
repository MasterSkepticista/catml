"""Synthetic patient-state dataset for the diagrammatic backprop example."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAUSES = ("viral", "bacterial", "allergic")
SEVERITIES = ("mild", "moderate", "severe")
SYSTEMS = ("respiratory", "gastro", "skin")
DIAGNOSES = (
    "cold",
    "bronchitis",
    "strep",
    "food_poisoning",
    "allergy",
    "eczema",
    "migraine",
)


@dataclass(frozen=True)
class Sample:
    syndrome: tuple[str, str, str]
    evidence: np.ndarray
    latent: np.ndarray
    diagnosis: np.ndarray


def build_dataset() -> list[Sample]:
    """Create a tiny synthetic clinic with a meaningful causal middle layer."""
    samples = []
    for cause_index, cause in enumerate(CAUSES):
        for severity_index, severity in enumerate(SEVERITIES):
            for system_index, system in enumerate(SYSTEMS):
                syndrome = (cause, severity, system)
                evidence = _encode_evidence(cause_index, severity_index, system_index)
                latent = _encode_latent(cause_index, severity_index, system_index)
                diagnosis = _encode_diagnosis(_diagnosis_index(cause_index, severity_index, system_index))
                samples.append(Sample(syndrome, evidence, latent, diagnosis))
    return samples


def split_direct_supervision(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    """Hold out non-moderate cases from the direct evidence->diagnosis edge."""
    direct_train = [sample for sample in samples if sample.syndrome[1] == "moderate"]
    held_out = [sample for sample in samples if sample.syndrome[1] != "moderate"]
    return direct_train, held_out


def _encode_evidence(cause_index: int, severity_index: int, system_index: int) -> np.ndarray:
    x = np.zeros((3, 6))

    symptom_a = [0, 0, 1][system_index]
    symptom_b = [1, 1, 0][system_index]
    x[0, symptom_a] = 1.0
    x[0, symptom_b] = 1.0
    x[0, 3 + cause_index] = 1.0

    x[1, severity_index] = 1.0
    x[1, 3 + cause_index] = 1.0
    if system_index == 0:
        x[1, 5] = 1.0

    x[2, cause_index] = 1.0
    x[2, 3 + severity_index] = 1.0
    if system_index == 1:
        x[2, 4] = 1.0
    if system_index == 2:
        x[2, 5] = 1.0
    return x


def _encode_latent(cause_index: int, severity_index: int, system_index: int) -> np.ndarray:
    x = np.zeros((3, 3))
    x[0, cause_index] = 1.0
    x[1, severity_index] = 1.0
    x[2, system_index] = 1.0
    return x


def _diagnosis_index(cause_index: int, severity_index: int, system_index: int) -> int:
    if cause_index == 2 and system_index == 2:
        return 5
    if cause_index == 2:
        return 4
    if system_index == 1:
        return 3
    if cause_index == 1 and severity_index == 2:
        return 2
    if cause_index == 0 and severity_index == 0:
        return 0
    if system_index == 0:
        return 1
    return 6


def _encode_diagnosis(index: int) -> np.ndarray:
    x = np.zeros((1, len(DIAGNOSES)))
    x[0, index] = 1.0
    return x
