import logging
import os
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any

import psutil
import torch

logger = logging.getLogger(__name__)

class BaseInferenceEngine(ABC):
    """Base class for all inference engines"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log_hardware_specs()
        self.model = self._load_model()
        logger.info(f"Initialized {self.__class__.__name__} on {self.device}")

    def _log_hardware_specs(self):
        """Logs CPU, RAM, and GPU specifications."""
        logger.info("--- Hardware Specifications ---")
        logger.info(f"CPU Cores (Physical/Logical): {psutil.cpu_count(logical=False)}/{psutil.cpu_count(logical=True)}")
        ram_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f"Total RAM: {ram_gb:.2f} GB")
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory_gb:.2f} GB")
        else:
            logger.info("GPU: Not available (running on CPU)")
        logger.info("-----------------------------")

    def _load_model(self) -> torch.nn.Module:
        """Load model with proper error handling and device mapping"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Use the robust loader from inference.py
            return self._load_real_model()
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {str(e)}")
            # Return a mock model for development/testing
            return self._get_mock_model()

    def _load_real_model(self):
        # Placeholder: override in subclasses if needed
        return self._get_mock_model()

    @abstractmethod
    def _get_mock_model(self) -> torch.nn.Module:
        """Return a mock model for testing"""
        pass

    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using the model"""
        pass

    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        return True

    def _get_explanation(self, input_data: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Generates a mock XAI explanation."""
        return "This prediction is based on the key features in the input data, processed by the model's learned patterns. Confidence is high due to strong feature alignment."

    def _apply_speculative_decoding(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Mocks the effect of speculative decoding by slightly altering results and adding a note."""
        result["note"] = "Speculative decoding applied, potentially reducing latency."
        # Example: slightly boost confidence as a mock effect
        if "confidence" in result:
            result["confidence"] = min(100.0, result.get("confidence", 0) * 1.05)
        if "predicted_properties" in result and "confidence_score" in result["predicted_properties"]:
             prediction = result.get("predicted_properties", {})
             prediction["confidence_score"] = min(100.0, prediction.get("confidence_score", 0) * 1.05)
             result["predicted_properties"] = prediction
        return result

class BiologyInferenceEngine(BaseInferenceEngine):
    """
    NexaBio_1: Predicts secondary protein structure (H/E/C)
    NexaBio_2: Predicts tertiary protein structure (3D coordinates, mock)
    """
    def _get_mock_model(self) -> torch.nn.Module:
        # Use different mock models for secondary/tertiary
        if "1" in os.path.basename(self.model_path):
            # Secondary structure: 20 input (AA), 3 output (H/E/C)
            return torch.nn.Sequential(
                torch.nn.Linear(20, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 3)
            )
        else:
            # Tertiary structure: 20 input (AA), 60 output (3D coords for 20 AA)
            return torch.nn.Sequential(
                torch.nn.Linear(20, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 60)
            )

    def get_secondary_structure(self, sequence: str) -> str:
        # Mock: repeat H/E/C for the sequence length
        pattern = "HEC"
        return "".join(pattern[i % 3] for i in range(len(sequence)))

    def get_tertiary_coordinates(self, sequence: str) -> list:
        # Mock: generate a list of [x, y, z] for each residue
        return [[round(i * 1.1, 2), round(i * 1.2, 2), round(i * 1.3, 2)] for i in range(len(sequence))]

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self._validate_input(input_data):
                raise ValueError("Invalid input data")
            sequence = input_data.get("sequence", "")
            confidence_threshold = input_data.get("confidence_threshold", 0.8)
            explain = input_data.get("explain", False)
            mode = input_data.get("mode")

            model_name = os.path.basename(self.model_path)
            timestamp = datetime.now().isoformat()

            result = {}
            if "1" in model_name:
                # NexaBio_1: Secondary structure
                structure = self.get_secondary_structure(sequence)
                confidence = 0.92
                if confidence < confidence_threshold:
                    structure = "U" * len(sequence)
                result = {
                    "sequence": sequence,
                    "secondary_structure": structure,
                    "confidence": round(confidence * 100, 2),
                    "timestamp": timestamp
                }
            else:
                # NexaBio_2: Tertiary structure
                coords = self.get_tertiary_coordinates(sequence)
                confidence = 0.89
                if confidence < confidence_threshold:
                    coords = []
                result = {
                    "sequence": sequence,
                    "tertiary_coordinates": coords,
                    "confidence": round(confidence * 100, 2),
                    "timestamp": timestamp
                }

            if mode == "speculative":
                result = self._apply_speculative_decoding(result)

            if explain:
                result["explanation"] = self._get_explanation(input_data, result)

            return result
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def generate_dataset(self, num_candidates: int, sequence_length: int = 12) -> list:
        dataset = []
        for _ in range(num_candidates):
            seq = ''.join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=sequence_length))
            pred = self.predict({"sequence": seq, "confidence_threshold": 0.8})
            dataset.append(pred)
        return dataset

    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate biology input data"""
        required_fields = ["sequence"]
        return all(field in input_data for field in required_fields)

class MaterialsInferenceEngine(BaseInferenceEngine):
    """
    NexaMat_1: Battery ion prediction (mock)
    NexaMat_2: GNN+VAE battery ion prediction (mock)
    """
    def _get_mock_model(self) -> torch.nn.Module:
        if "1" in os.path.basename(self.model_path):
            # Simple battery ion prediction
            return torch.nn.Sequential(
                torch.nn.Linear(100, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            )
        else:
            # GNN+VAE battery ion prediction
            return torch.nn.Sequential(
                torch.nn.Linear(100, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1)
            )

    def get_material_prediction(self, structure: str) -> dict:
        # Mock: return a dict with all required prediction labels
        return {
            "formation_energy_per_atom": -0.4063791,
            "energy_per_atom": -0.027647773,
            "density": -0.6608056,
            "volume": 0.12958337,
            "n_elements": 5.313307,
            "li_fraction": 0.10428204,
            "predicted_band_gap": 1.515275,
            "confidence_score": 99.9999951393316
        }

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self._validate_input(input_data):
                raise ValueError("Invalid input data")
            structure = input_data.get("structure", "")
            energy_threshold = input_data.get("energy_threshold", 0.5)
            explain = input_data.get("explain", False)
            mode = input_data.get("mode")

            model_name = os.path.basename(self.model_path)
            timestamp = datetime.now().isoformat()
            prediction = self.get_material_prediction(structure)
            confidence = prediction["confidence_score"] / 100.0

            if confidence < energy_threshold:
                prediction = {k: None for k in prediction}

            result = {
                "input_structure": structure,
                "predicted_properties": prediction,
                "timestamp": timestamp
            }

            if mode == "speculative":
                result = self._apply_speculative_decoding(result)

            if explain:
                result["explanation"] = self._get_explanation(input_data, result)

            return result
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def generate_dataset(self, num_candidates: int, structure_length: int = 10) -> list:
        dataset = []
        for _ in range(num_candidates):
            struct = ''.join(random.choices("LiNaKMgAlSiO", k=structure_length))
            pred = self.predict({"structure": struct, "energy_threshold": 0.5})
            dataset.append(pred)
        return dataset

    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate materials input data"""
        required_fields = ["structure"]
        return all(field in input_data for field in required_fields)

class LLMInferenceEngine(BaseInferenceEngine):
    """ Mock LLM Engine for text generation """
    def _get_mock_model(self) -> torch.nn.Module:
        return torch.nn.Identity() # No real model needed for mock

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        max_tokens = input_data.get("max_tokens", 100)

        start_time = time.perf_counter()

        # Simulate token generation
        time_to_first_token = random.uniform(0.05, 0.15) # Simulate TTFT
        time.sleep(time_to_first_token)

        # Simulate rest of the generation
        time.sleep(0.005 * max_tokens) # Simulate work per token
        words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]
        response_text = " ".join(random.choices(words, k=max_tokens))

        end_time = time.perf_counter()

        total_latency_ms = (end_time - start_time) * 1000
        ttft_ms = time_to_first_token * 1000
        tokens_per_second = max_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else float('inf')

        return {
            "response": response_text,
            "tokens_generated": max_tokens,
            "metrics": {
                "total_latency_ms": round(total_latency_ms, 2),
                "time_to_first_token_ms": round(ttft_ms, 2),
                "tokens_per_second": round(tokens_per_second, 2)
            }
        }

    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        return "prompt" in input_data
