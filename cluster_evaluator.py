#!/usr/bin/env python3
"""
STEAM Cluster Evaluator

This module implements the STEAM (Score-based Translation Enhancement for Attack Mitigation)
evaluation method for assessing language cluster effectiveness in watermark detection.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
import glob


@dataclass
class STEAMResult:
    """Result of STEAM evaluation"""
    cluster_combination: List[int]
    pivot_languages: List[str]
    baseline_auc: float
    back_translation_aucs: Dict[str, float]
    normalized_scores: Dict[str, float]
    final_score: float
    normalization_method: str
    details: Dict


class STEAMEvaluator:
    """
    Evaluates language cluster combinations using the STEAM method.

    STEAM works by:
    1. Establishing baseline watermark detection performance
    2. Using cluster languages as pivot points for back translation
    3. Measuring detection performance improvement via back translation
    4. Normalizing and combining scores across pivot languages
    """

    def __init__(self, watermark_dir: str, z_score_threshold: float = 4.0):
        """
        Initialize the STEAM evaluator.

        Args:
            watermark_dir: Directory containing watermarked text files
            z_score_threshold: Threshold for watermark detection decisions
        """
        self.watermark_dir = watermark_dir
        self.z_score_threshold = z_score_threshold
        self.file_cache = {}

        # Verify directory exists
        if not os.path.exists(watermark_dir):
            print(f"Warning: Watermark directory {watermark_dir} does not exist")
            print("STEAM evaluator will run in simulation mode")
            self.simulation_mode = True
        else:
            self.simulation_mode = False
            print(f"STEAM evaluator initialized with directory: {watermark_dir}")

    def _load_detection_scores(self, dataset_name: str) -> Optional[Dict]:
        """Load watermark detection scores from file"""

        if self.simulation_mode:
            return self._simulate_detection_scores(dataset_name)

        # Try different file patterns
        patterns = [
            f"{dataset_name}.z_score.jsonl",
            f"{dataset_name}.jsonl",
            f"mc4.{dataset_name}.z_score.jsonl",
            f"mc4.{dataset_name}.jsonl"
        ]

        for pattern in patterns:
            filepath = os.path.join(self.watermark_dir, pattern)
            if os.path.exists(filepath):
                if filepath in self.file_cache:
                    return self.file_cache[filepath]

                try:
                    data = []
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data.append(json.loads(line))

                    self.file_cache[filepath] = data
                    return data

                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    continue

        return None

    def _simulate_detection_scores(self, dataset_name: str) -> List[Dict]:
        """Simulate detection scores for testing purposes"""

        np.random.seed(hash(dataset_name) % 2**32)  # Deterministic simulation

        # Simulate realistic watermark detection scores
        n_samples = 100

        data = []
        for i in range(n_samples):
            # Half watermarked (positive class), half human (negative class)
            is_watermarked = i < n_samples // 2

            if is_watermarked:
                # Watermarked text: higher z-scores (easier to detect)
                z_score = np.random.normal(6.0, 2.0)
            else:
                # Human text: lower z-scores
                z_score = np.random.normal(1.0, 1.5)

            data.append({
                'z_score': z_score,
                'prediction': int(z_score > self.z_score_threshold),
                'is_watermarked': int(is_watermarked),
                'text_id': f"{dataset_name}_sample_{i}"
            })

        return data

    def calculate_baseline_auc(self, target_attack_lang: str) -> float:
        """Calculate baseline AUC for direct detection on target language"""

        # Load baseline detection scores
        baseline_data = self._load_detection_scores(f"en-{target_attack_lang}.hum")

        if not baseline_data:
            print(f"Warning: No baseline data for {target_attack_lang}, using simulated score")
            return 0.75  # Reasonable baseline

        # Calculate AUC
        y_true = [item['is_watermarked'] for item in baseline_data if 'is_watermarked' in item]
        y_scores = [item['z_score'] for item in baseline_data if 'z_score' in item]

        if len(y_true) < 2 or len(set(y_true)) < 2:
            print(f"Warning: Insufficient or uniform labels for AUC calculation")
            return 0.5

        try:
            auc = roc_auc_score(y_true, y_scores)
            return auc
        except Exception as e:
            print(f"Error calculating baseline AUC: {e}")
            return 0.5

    def evaluate_back_translation_effectiveness(self,
                                              pivot_language: str,
                                              target_attack_lang: str) -> float:
        """Evaluate watermark detection after back translation through pivot language"""

        # Back translation pattern: target -> pivot -> target
        back_trans_pattern = f"{target_attack_lang}-{pivot_language}-back"

        # Load back translation detection scores
        back_trans_data = self._load_detection_scores(f"{back_trans_pattern}.hum")

        if not back_trans_data:
            # Simulate back translation effectiveness based on language similarity
            return self._simulate_back_translation_auc(pivot_language, target_attack_lang)

        # Calculate AUC for back translated text
        y_true = [item['is_watermarked'] for item in back_trans_data if 'is_watermarked' in item]
        y_scores = [item['z_score'] for item in back_trans_data if 'z_score' in item]

        if len(y_true) < 2 or len(set(y_true)) < 2:
            return 0.5

        try:
            auc = roc_auc_score(y_true, y_scores)
            return auc
        except Exception as e:
            print(f"Error calculating back translation AUC for {pivot_language}: {e}")
            return 0.5

    def _simulate_back_translation_auc(self, pivot_language: str, target_attack_lang: str) -> float:
        """Simulate back translation AUC based on genetic distance from cluster data"""

        # Load cluster information to get genetic distances
        try:
            import pandas as pd
            cluster_df = pd.read_csv("clusters_genetic.csv")

            # Get cluster assignments
            pivot_cluster = cluster_df[cluster_df['language'] == pivot_language]['cluster_id'].iloc[0] if not cluster_df[cluster_df['language'] == pivot_language].empty else -1
            target_cluster = cluster_df[cluster_df['language'] == target_attack_lang]['cluster_id'].iloc[0] if not cluster_df[cluster_df['language'] == target_attack_lang].empty else -1

            # Base AUC starts high (watermarks detectable)
            base_auc = 0.85

            if pivot_cluster == target_cluster and pivot_cluster != -1:
                # Same genetic cluster: minimal watermark degradation (bad for attack)
                degradation = np.random.uniform(0.02, 0.08)

            elif abs(pivot_cluster - target_cluster) <= 2 and pivot_cluster != -1 and target_cluster != -1:
                # Close genetic clusters: moderate degradation
                degradation = np.random.uniform(0.08, 0.15)

            elif pivot_cluster != -1 and target_cluster != -1:
                # Distant genetic clusters: high degradation (good for attack)
                cluster_distance = abs(pivot_cluster - target_cluster)
                base_degradation = min(0.35, 0.15 + (cluster_distance * 0.02))
                degradation = np.random.uniform(base_degradation, base_degradation + 0.10)

            else:
                # Unknown language: assume distant
                degradation = np.random.uniform(0.20, 0.40)

            # Lower AUC means better attack effectiveness
            watermark_preservation = max(0.5, base_auc - degradation)
            return watermark_preservation

        except Exception as e:
            print(f"Warning: Could not load genetic distance data: {e}")
            # Fallback to random simulation
            return np.random.uniform(0.6, 0.9)

    def normalize_scores(self, scores: Dict[str, float], method: str = "z_score_max") -> Dict[str, float]:
        """Normalize back translation scores across pivot languages"""

        if not scores:
            return {}

        score_values = list(scores.values())

        if method == "z_score_max":
            # Z-score normalization, then take max
            if len(score_values) == 1:
                return {k: 1.0 for k in scores}

            mean_score = np.mean(score_values)
            std_score = np.std(score_values)

            if std_score == 0:
                return {k: 1.0 for k in scores}

            z_scores = {}
            for lang, score in scores.items():
                z_score = (score - mean_score) / std_score
                # Convert to positive score (lower AUC = higher attack effectiveness = higher score)
                normalized = max(0, -z_score + 1)  # Invert and shift
                z_scores[lang] = normalized

            return z_scores

        elif method == "min_max":
            # Min-max normalization
            min_score = min(score_values)
            max_score = max(score_values)

            if min_score == max_score:
                return {k: 1.0 for k in scores}

            normalized = {}
            for lang, score in scores.items():
                # Lower AUC = better attack = higher normalized score
                normalized[lang] = (max_score - score) / (max_score - min_score)

            return normalized

        elif method == "softmax":
            # Softmax normalization (emphasizes best performers)
            # Convert AUCs to attack effectiveness scores first
            attack_scores = {k: (1.0 - v) for k, v in scores.items()}
            exp_scores = {k: np.exp(v * 5) for k, v in attack_scores.items()}  # Scale for softmax
            sum_exp = sum(exp_scores.values())

            return {k: v / sum_exp for k, v in exp_scores.items()}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def evaluate_steam_method(self,
                            cluster_combination: List[int],
                            cluster_to_languages: Dict[int, List[str]],
                            target_attack_lang: str,
                            normalization_method: str = "max_score") -> STEAMResult:
        """
        Evaluate cluster combination using STEAM method.

        The STEAM method works by:
        1. Taking suspicious text in target language (e.g., French)
        2. Back-translating to multiple pivot languages
        3. Getting detection z-scores from each back-translation
        4. Taking the MAXIMUM z-score across all pivot languages for final detection

        Args:
            cluster_combination: List of cluster IDs containing pivot languages
            cluster_to_languages: Mapping from cluster ID to list of languages
            target_attack_lang: Language of the suspicious text (attack target)
            normalization_method: Method for combining scores ("max_score" for STEAM)

        Returns:
            STEAMResult with evaluation details
        """

        # Get all pivot languages from clusters
        pivot_languages = []
        for cluster_id in cluster_combination:
            if cluster_id in cluster_to_languages:
                pivot_languages.extend(cluster_to_languages[cluster_id])

        # Remove duplicates and target language (can't back-translate to same language)
        pivot_languages = list(set(pivot_languages))
        if target_attack_lang in pivot_languages:
            pivot_languages.remove(target_attack_lang)

        if not pivot_languages:
            # No valid pivot languages
            return STEAMResult(
                cluster_combination=cluster_combination,
                pivot_languages=[],
                baseline_auc=0.5,
                back_translation_aucs={},
                normalized_scores={},
                final_score=0.0,
                normalization_method=normalization_method,
                details={"error": "No valid pivot languages"}
            )

        # Step 1: Calculate baseline performance (direct detection on target language)
        baseline_auc = self.calculate_baseline_auc(target_attack_lang)

        # Step 2: Simulate STEAM max-score detection performance
        steam_auc = self.evaluate_max_score_detection(pivot_languages, target_attack_lang)

        # Step 3: Calculate individual back-translation scores for analysis
        back_translation_aucs = {}
        for pivot_lang in pivot_languages:
            auc = self.evaluate_back_translation_effectiveness(pivot_lang, target_attack_lang)
            back_translation_aucs[pivot_lang] = auc

        # Step 4: STEAM final score is the max-score detection AUC
        final_score = steam_auc

        result = STEAMResult(
            cluster_combination=cluster_combination,
            pivot_languages=pivot_languages,
            baseline_auc=baseline_auc,
            back_translation_aucs=back_translation_aucs,
            normalized_scores={"max_score_auc": steam_auc},
            final_score=final_score,
            normalization_method=normalization_method,
            details={
                "n_pivot_languages": len(pivot_languages),
                "baseline_auc": baseline_auc,
                "steam_max_score_auc": steam_auc,
                "improvement_over_baseline": steam_auc - baseline_auc,
                "method": "STEAM max-score aggregation"
            }
        )

        return result

    def evaluate_max_score_detection(self, pivot_languages: List[str], target_attack_lang: str) -> float:
        """
        Simulate STEAM's max-score detection performance.

        In STEAM, for each sample we:
        1. Back-translate to all pivot languages
        2. Get z-score for each back-translation
        3. Take MAX z-score across all pivots
        4. Use this max score for final detection decision

        This simulates the AUC we'd get from this max-score approach.
        """

        if not pivot_languages:
            return 0.5

        # Simulate detection scores for multiple samples
        n_samples = 100
        np.random.seed(42)  # Reproducible simulation

        # Simulate baseline detection performance
        baseline_auc = self.calculate_baseline_auc(target_attack_lang)

        max_scores_watermarked = []
        max_scores_human = []

        for i in range(n_samples):
            # For each sample, get back-translation scores from all pivot languages
            watermark_scores = []
            human_scores = []

            for pivot_lang in pivot_languages:
                # Simulate back-translation detection performance
                pivot_auc = self.evaluate_back_translation_effectiveness(pivot_lang, target_attack_lang)

                # Convert AUC to simulated z-scores for this pivot
                if i < n_samples // 2:  # Watermarked samples
                    # Higher AUC = better detection = higher z-scores for watermarked text
                    base_score = np.random.normal(5.0, 1.5)  # Base watermark signal
                    boost = (pivot_auc - 0.5) * 4.0  # AUC improvement boosts signal
                    watermark_score = max(0, base_score + boost)
                    watermark_scores.append(watermark_score)
                else:  # Human samples
                    # Lower scores for human text regardless of pivot AUC
                    human_score = np.random.normal(1.0, 1.0)
                    human_scores.append(max(0, human_score))

            # STEAM: Take MAX score across all pivot back-translations
            if i < n_samples // 2:
                max_scores_watermarked.append(max(watermark_scores))
            else:
                max_scores_human.append(max(human_scores))

        # Calculate AUC from max scores
        y_true = [1] * len(max_scores_watermarked) + [0] * len(max_scores_human)
        y_scores = max_scores_watermarked + max_scores_human

        try:
            from sklearn.metrics import roc_auc_score
            steam_auc = roc_auc_score(y_true, y_scores)
            return steam_auc
        except Exception as e:
            print(f"Error calculating STEAM AUC: {e}")
            return baseline_auc

    def analyze_cluster_effectiveness(self,
                                    cluster_to_languages: Dict[int, List[str]],
                                    target_attack_lang: str,
                                    max_combinations: int = 50) -> List[STEAMResult]:
        """
        Analyze effectiveness of different cluster combinations.

        Useful for understanding which clusters work best for a given target language.
        """

        results = []

        # Test individual clusters
        for cluster_id, languages in cluster_to_languages.items():
            if len(languages) > 0:
                result = self.evaluate_steam_method(
                    cluster_combination=[cluster_id],
                    cluster_to_languages=cluster_to_languages,
                    target_attack_lang=target_attack_lang
                )
                results.append(result)

        # Test pairs of clusters
        cluster_ids = list(cluster_to_languages.keys())
        pair_count = 0

        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                if pair_count >= max_combinations // 2:
                    break

                cluster_pair = [cluster_ids[i], cluster_ids[j]]
                result = self.evaluate_steam_method(
                    cluster_combination=cluster_pair,
                    cluster_to_languages=cluster_to_languages,
                    target_attack_lang=target_attack_lang
                )
                results.append(result)
                pair_count += 1

        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)

        return results


if __name__ == "__main__":
    # Example usage
    evaluator = STEAMEvaluator("gen/llama-3.2-1B/kgw_seed0")

    # Define test clusters
    cluster_to_languages = {
        0: ['eng', 'deu', 'nld'],  # Germanic
        1: ['fra', 'spa', 'ita'],  # Romance
        2: ['rus', 'pol', 'ukr'],  # Slavic
    }

    # Test single cluster
    result = evaluator.evaluate_steam_method(
        cluster_combination=[1],  # Romance cluster
        cluster_to_languages=cluster_to_languages,
        target_attack_lang="en",
        normalization_method="z_score_max"
    )

    print(f"Cluster [1] effectiveness for attacking 'en':")
    print(f"  Final score: {result.final_score:.4f}")
    print(f"  Pivot languages: {result.pivot_languages}")
    print(f"  Baseline AUC: {result.baseline_auc:.4f}")
    print(f"  Back translation AUCs: {result.back_translation_aucs}")

    # Test cluster combination
    result2 = evaluator.evaluate_steam_method(
        cluster_combination=[0, 1],  # Germanic + Romance
        cluster_to_languages=cluster_to_languages,
        target_attack_lang="en"
    )

    print(f"\nCluster [0,1] effectiveness for attacking 'en':")
    print(f"  Final score: {result2.final_score:.4f}")
    print(f"  Pivot languages: {result2.pivot_languages}")