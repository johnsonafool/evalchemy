import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


class BFCLBenchmark(BaseBenchmark):
    """
    BFCL Benchmark for evaluating function calling capabilities.
    """

    def __init__(
        self,
        debug: bool = False,
        seed: int = 1234,
        max_tokens: int = 32768,
        test_categories: List[str] = None,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize BFCL benchmark.

        Args:
            debug: If set, only evaluate on a small subset of examples
            seed: Random seed for reproducibility
            max_tokens: Maximum number of tokens for generation
            test_categories: List of test categories to run (e.g., ['simple', 'parallel'])
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = "BFCL"
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.test_categories = test_categories or [
            "simple"
        ]  # Default to simple category

        # Set up paths
        self.bfcl_root = Path(__file__).parent / "berkeley-function-call-leaderboard"
        self.data_path = self.bfcl_root / "bfcl_eval" / "data"

        # Add BFCL to Python path for imports
        if str(self.bfcl_root) not in sys.path:
            sys.path.insert(0, str(self.bfcl_root))

    def _get_model_name(self, model: LM) -> str:
        """Extract a clean model name for file naming."""
        if hasattr(model, "model_identifier"):
            return (
                model.model_identifier.split("=")[1]
                .split(",")[0]
                .split("__")[-1]
                .replace("-", "_")
                .lower()
                .replace(".", "")
            )
        else:
            return model.__class__.__name__.lower()

    def _load_test_data(self, category: str) -> List[Dict]:
        """Load test data for a specific category."""
        test_file = self.data_path / f"BFCL_v3_{category}.json"

        if not test_file.exists():
            self.logger.warning(f"Test file not found: {test_file}")
            return []

        test_cases = []
        with open(test_file, "r") as f:
            for line in f:
                test_cases.append(json.loads(line.strip()))

        if self.debug:
            test_cases = test_cases[:10]  # Limit to 10 cases for debugging

        return test_cases

    def _prepare_function_calling_prompt(self, test_case: Dict) -> str:
        """Prepare the prompt for function calling."""
        # Extract the user question
        question = test_case["question"][0][0]["content"]

        # Format function definitions
        functions = test_case.get("function", [])
        if not functions:
            return question

        # Create function calling prompt
        function_descriptions = []
        for func in functions:
            func_desc = f"Function: {func['name']}\n"
            func_desc += f"Description: {func['description']}\n"

            if "parameters" in func:
                params = func["parameters"]
                if "properties" in params:
                    func_desc += "Parameters:\n"
                    for param_name, param_info in params["properties"].items():
                        param_type = param_info.get("type", "unknown")
                        param_desc = param_info.get("description", "")
                        required = param_name in params.get("required", [])
                        req_str = " (required)" if required else " (optional)"
                        func_desc += (
                            f"  - {param_name} ({param_type}){req_str}: {param_desc}\n"
                        )

            function_descriptions.append(func_desc)

        # Combine into final prompt
        prompt = "You are a helpful assistant that can call functions to help answer questions.\n\n"
        prompt += "Available functions:\n"
        prompt += "\n".join(function_descriptions)
        prompt += f"\n\nUser question: {question}\n\n"
        prompt += (
            "Please provide the appropriate function call(s) to answer this question. "
        )
        prompt += "Format your response as a JSON object with the function name and parameters."

        return prompt

    def generate_responses(self, model: LM) -> List[Dict[str, Any]]:
        """
        Generate model responses for BFCL test cases.

        Args:
            model: Language model instance

        Returns:
            List of dictionaries containing model outputs and metadata
        """
        self.logger.info("Generating responses for BFCL...")

        # model_name = self._get_model_name(model)
        model_name = model
        all_results = []

        for category in self.test_categories:
            self.logger.info(f"Processing category: {category}")
            test_cases = self._load_test_data(category)

            if not test_cases:
                self.logger.warning(f"No test cases found for category: {category}")
                continue

            # Prepare instances for batch processing
            instances = []
            for i, test_case in enumerate(test_cases):
                prompt = self._prepare_function_calling_prompt(test_case)

                messages = [{"role": "user", "content": prompt}]
                templated_messages = self._prepare_messages(messages, model)

                instance = Instance(
                    "generate_until",
                    test_case,
                    (
                        templated_messages,
                        {
                            "max_new_tokens": self.max_new_tokens,
                            "do_sample": False,
                            "temperature": 0.0,
                            "seed": self.seed,
                        },
                    ),
                    i,
                )
                instances.append(instance)

            # Generate responses
            if instances:
                self.logger.info(
                    f"Generating {len(instances)} responses for {category}"
                )
                outputs = self.compute(model, instances)

                # Process outputs
                for i, (test_case, output) in enumerate(zip(test_cases, outputs)):
                    result = {
                        "id": test_case["id"],
                        "category": category,
                        "question": test_case["question"],
                        "function": test_case.get("function", []),
                        "model_response": output.strip(),
                        "model_name": model_name,
                    }
                    all_results.append(result)

        return all_results

    def _run_bfcl_evaluation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run the official BFCL evaluation using their evaluation scripts.
        """
        model_name = results[0]["model_name"] if results else "unknown_model"

        # Create temporary directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            result_dir = temp_path / "result" / model_name
            result_dir.mkdir(parents=True, exist_ok=True)

            # Group results by category and save in BFCL format
            category_results = {}
            for result in results:
                category = result["category"]
                if category not in category_results:
                    category_results[category] = []

                # Format result in BFCL expected format
                bfcl_result = {"id": result["id"], "result": result["model_response"]}
                category_results[category].append(bfcl_result)

            # Save results to files
            for category, cat_results in category_results.items():
                result_file = result_dir / f"BFCL_v3_{category}_result.json"
                with open(result_file, "w") as f:
                    for res in cat_results:
                        f.write(json.dumps(res) + "\n")

            # Run BFCL evaluation
            try:
                # Change to BFCL directory and run evaluation
                original_cwd = os.getcwd()
                os.chdir(self.bfcl_root)

                # Prepare evaluation command
                eval_cmd = (
                    [
                        sys.executable,
                        "-m",
                        "bfcl_eval.eval_checker.eval_runner",
                        "--model",
                        model_name,
                        "--test-category",
                    ]
                    + self.test_categories
                    + [
                        "--result-dir",
                        str(temp_path / "result"),
                        "--score-dir",
                        str(temp_path / "score"),
                    ]
                )

                # Run evaluation
                result = subprocess.run(
                    eval_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )

                if result.returncode != 0:
                    self.logger.error(f"BFCL evaluation failed: {result.stderr}")
                    return {"error": f"Evaluation failed: {result.stderr}"}

                # Load evaluation results
                score_dir = temp_path / "score" / model_name
                evaluation_results = {}

                for category in self.test_categories:
                    score_file = score_dir / f"BFCL_v3_{category}_score.json"
                    if score_file.exists():
                        with open(score_file, "r") as f:
                            lines = f.readlines()
                            if lines:
                                # First line contains summary statistics
                                summary = json.loads(lines[0])
                                evaluation_results[category] = {
                                    "accuracy": summary.get("accuracy", 0.0),
                                    "correct_count": summary.get("correct_count", 0),
                                    "total_count": summary.get("total_count", 0),
                                }

                return evaluation_results

            except subprocess.TimeoutExpired:
                self.logger.error("BFCL evaluation timed out")
                return {"error": "Evaluation timed out"}
            except Exception as e:
                self.logger.error(f"Error running BFCL evaluation: {str(e)}")
                return {"error": f"Evaluation error: {str(e)}"}
            finally:
                os.chdir(original_cwd)

    def evaluate_responses(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the generated responses using BFCL evaluation metrics.

        Args:
            results: List of dictionaries containing model outputs

        Returns:
            Dictionary containing evaluation metrics
        """
        if not results:
            return {"error": "No results to evaluate"}

        self.logger.info("Evaluating BFCL responses...")

        # Run official BFCL evaluation
        evaluation_results = self._run_bfcl_evaluation(results)

        if "error" in evaluation_results:
            return evaluation_results

        # Calculate overall metrics
        metrics = {}
        total_correct = 0
        total_count = 0

        for category, cat_results in evaluation_results.items():
            accuracy = cat_results.get("accuracy", 0.0)
            correct = cat_results.get("correct_count", 0)
            count = cat_results.get("total_count", 0)

            metrics[f"{category}_accuracy"] = accuracy * 100  # Convert to percentage
            metrics[f"{category}_correct"] = correct
            metrics[f"{category}_total"] = count

            total_correct += correct
            total_count += count

        # Overall accuracy
        if total_count > 0:
            metrics["overall_accuracy"] = (total_correct / total_count) * 100
        else:
            metrics["overall_accuracy"] = 0.0

        metrics["total_correct"] = total_correct
        metrics["total_count"] = total_count

        result_dict = {
            "metrics": metrics,
            "num_questions": total_count,
            "benchmark_version": "BFCL_v3",
        }

        return result_dict

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete BFCL benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Starting BFCL benchmark evaluation")
        try:
            generation_results = self.generate_responses(model)

            if not generation_results:
                return {"error": "No generation results"}

            evaluation_results = self.evaluate_responses(generation_results)
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running BFCL benchmark: {str(e)}")
            return {"error": str(e)}


# Legacy function interface for backward compatibility
def eval_instruct(model: LM) -> Dict[str, Any]:
    """
    Legacy function interface for BFCL evaluation.

    Args:
        model: Language model instance

    Returns:
        Dictionary containing model outputs and identifier
    """
    benchmark = BFCLBenchmark()
    return benchmark.generate_responses(model)


def evaluate(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Legacy function interface for BFCL evaluation.

    Args:
        results: Dictionary containing model outputs

    Returns:
        Dictionary containing evaluation metrics
    """
    benchmark = BFCLBenchmark()
    return benchmark.evaluate_responses(results)
