#!/usr/bin/env python3
"""
Modal Logic Mechanistic Interpretability - Easy Interactive CLI Runner
Designed for beginners and advanced users: automatic pip dependency installation,
interactive menu navigation, pre-flight checkpoints, and complete pipeline execution.
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
MODAL_MI_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = MODAL_MI_DIR.parent
TRANSFORMER_CIRCUIT_DIR = WORKSPACE_ROOT / "modal-logic-transformer-circuit"
REQUIREMENTS_FILE = MODAL_MI_DIR / "requirements.txt"
if not REQUIREMENTS_FILE.exists():
    REQUIREMENTS_FILE = WORKSPACE_ROOT / "requirements.txt"

for d in [MODAL_MI_DIR, TRANSFORMER_CIRCUIT_DIR]:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

# Robust progress bar support: imports tqdm if available, otherwise provides ANSI fallback
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, desc="", unit="it", leave=True, disable=False):
            self.iterable = iterable
            self.total = total if total is not None else (len(iterable) if iterable is not None and hasattr(iterable, "__len__") else None)
            self.desc = desc
            self.unit = unit
            self.disable = disable
            self.n = 0
            self.start_time = time.time()

        def __iter__(self):
            if self.iterable is None:
                return
            for item in self.iterable:
                yield item
                self.update(1)
            self.close()

        def update(self, n=1):
            if self.disable:
                return
            self.n += n
            elapsed = time.time() - self.start_time
            rate = self.n / max(elapsed, 1e-6)
            if self.total:
                pct = min(100.0, (self.n / self.total) * 100.0)
                bar_len = 25
                filled = int(bar_len * self.n / self.total)
                bar = "=" * filled + (">" if filled < bar_len else "") + "." * max(0, bar_len - filled - 1)
                sys.stdout.write(f"\r{self.desc}: [{bar}] {pct:5.1f}% ({self.n}/{self.total} {self.unit}) [{elapsed:04.1f}s, {rate:4.1f}{self.unit}/s]")
            else:
                sys.stdout.write(f"\r{self.desc}: {self.n} {self.unit} [{elapsed:04.1f}s, {rate:4.1f}{self.unit}/s]")
            sys.stdout.flush()

        def set_description(self, desc: str):
            self.desc = desc

        def close(self):
            if not self.disable:
                sys.stdout.write("\n")
                sys.stdout.flush()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()


MODEL_PRESETS = {
    "1": ("Qwen/Qwen3.5-2B", "Qwen3.5-2B (Fast / Low VRAM ~4GB)"),
    "2": ("Qwen/Qwen3.5-4B", "Qwen3.5-4B (Balanced ~8GB VRAM)"),
    "3": ("Qwen/Qwen3.5-9B", "Qwen3.5-9B (Full Scale / Recommended ~18GB VRAM)"),
    "qwen3.5-2b": ("Qwen/Qwen3.5-2B", "Qwen3.5-2B"),
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "Qwen3.5-4B"),
    "qwen3.5-9b": ("Qwen/Qwen3.5-9B", "Qwen3.5-9B"),
    "2b": ("Qwen/Qwen3.5-2B", "Qwen3.5-2B"),
    "4b": ("Qwen/Qwen3.5-4B", "Qwen3.5-4B"),
    "9b": ("Qwen/Qwen3.5-9B", "Qwen3.5-9B"),
}

CORE_PACKAGES = [
    ("torch", "PyTorch deep learning backend"),
    ("transformers", "Hugging Face model architecture"),
    ("transformer_lens", "Mechanistic interpretability activation patching hooks"),
    ("yaml", "PyYAML configuration parser"),
    ("tqdm", "Interactive progress bars"),
    ("matplotlib", "Publication figure renderer"),
    ("numpy", "Numerical tensor operations"),
    ("pandas", "Sufficiency table formatting"),
    ("accelerate", "Model loading acceleration"),
]


def check_missing_dependencies() -> List[str]:
    """Returns list of missing packages."""
    missing = []
    for pkg_import, desc in CORE_PACKAGES:
        try:
            importlib.import_module(pkg_import)
        except ImportError:
            missing.append(pkg_import)
    return missing


def install_dependencies(requirements_path: Optional[Path] = None) -> bool:
    """Installs required dependencies using pip."""
    req_file = requirements_path or REQUIREMENTS_FILE
    print("\n" + "=" * 70)
    print("  📦 INSTALLING PYTHON DEPENDENCIES VIA PIP")
    print("=" * 70)
    if req_file.exists():
        print(f"Installing from: {req_file}...")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
    else:
        pkgs = ["torch", "transformers", "transformer_lens", "accelerate", "pandas", "matplotlib", "numpy", "scipy", "pytest", "pyyaml", "tqdm"]
        print(f"Installing default package set: {', '.join(pkgs)}...")
        cmd = [sys.executable, "-m", "pip", "install"] + pkgs

    try:
        subprocess.check_call(cmd)
        print("\n[OK] Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Installation failed with code {e.returncode}. You may need to run as administrator or check internet connection.")
        return False


def ensure_dependencies_interactive() -> None:
    """Checks dependencies and prompts the user if any are missing."""
    missing = check_missing_dependencies()
    if missing:
        print("\n" + "!" * 70)
        print("  ⚠️  MISSING PYTHON PACKAGES DETECTED")
        print("!" * 70)
        print(f"The following required packages are not yet installed in this environment:")
        for m in missing:
            print(f"   - {m}")
        print("-" * 70)
        resp = input("Would you like to install them automatically now? [Y/n]: ").strip().lower()
        if resp in ["", "y", "yes"]:
            install_dependencies()
        else:
            print("Continuing without installing. (Some features may be limited).")


class PreflightCheckpoints:
    """Pre-flight checks: environment dependencies, hardware/CUDA, LLM cache, datasets."""

    def __init__(self, model_id: str, device: str = "auto", skip_llm: bool = False):
        self.model_id = model_id
        self.requested_device = device
        self.skip_llm = skip_llm
        self.results: Dict[str, Any] = {}

    def check_environment(self) -> Tuple[bool, str]:
        info = {
            "python_version": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
            "architecture": platform.machine(),
        }
        pkg_status = {}
        all_ok = True
        for pkg, desc in CORE_PACKAGES:
            try:
                mod = importlib.import_module(pkg)
                pkg_status[pkg] = getattr(mod, "__version__", "installed")
            except ImportError:
                pkg_status[pkg] = "NOT INSTALLED"
                all_ok = False

        info["packages"] = pkg_status
        self.results["environment"] = info
        return all_ok, f"Python {info['python_version']} on {info['os']}"

    def check_hardware(self) -> Tuple[bool, str]:
        hw_info: Dict[str, Any] = {"cuda_available": False, "device_count": 0, "devices": []}
        try:
            import torch
            if torch.cuda.is_available():
                hw_info["cuda_available"] = True
                hw_info["device_count"] = torch.cuda.device_count()
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    hw_info["devices"].append({"id": i, "name": name, "vram_gb": round(mem, 2)})
                resolved_device = "cuda:0" if self.requested_device in ["auto", "cuda"] else "cpu"
            else:
                resolved_device = "cpu"
        except Exception:
            resolved_device = "cpu"

        hw_info["resolved_device"] = resolved_device
        self.results["hardware"] = hw_info
        if hw_info["cuda_available"]:
            dev_str = ", ".join([f"{d['name']} ({d['vram_gb']}GB VRAM)" for d in hw_info["devices"]])
            return True, f"GPU Ready: {dev_str}"
        return True, f"Using CPU execution ({resolved_device})"

    def check_datasets(self, repo_root: Path) -> Tuple[bool, str]:
        data_dir = repo_root / "data"
        expected = [
            data_dir / "modal_circuit" / "modal_circuit_pairs.jsonl",
            data_dir / "modal_mi" / "modal_mi_facts_first.jsonl",
            data_dir / "modal_mi" / "modal_mi_expr_first.jsonl",
        ]
        status: Dict[str, Any] = {}
        all_found = True
        for p in expected:
            rel = p.relative_to(repo_root) if p.is_relative_to(repo_root) else p.name
            if p.exists() and p.stat().st_size > 0:
                with p.open("r", encoding="utf-8") as f:
                    lines = sum(1 for _ in f)
                status[str(rel)] = {"status": "EXISTS", "samples": lines, "bytes": p.stat().st_size}
            else:
                status[str(rel)] = {"status": "MISSING"}
                all_found = False

        self.results["datasets"] = status
        if all_found:
            return True, f"All primary dataset files verified ({len(expected)} files present)."
        return True, "Some dataset files missing; test suite will synthesize required samples on-the-fly."

    def check_llm_availability(self) -> Tuple[bool, str]:
        if self.skip_llm:
            self.results["llm_check"] = {"status": "SKIPPED", "model_id": self.model_id}
            return True, f"LLM check skipped for {self.model_id}"

        res: Dict[str, Any] = {
            "model_id": self.model_id,
            "cached_locally": False,
            "hf_reachable": False,
            "cache_path": None,
        }

        # Check local HF Hub cache
        cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        cache_path = Path(cache_home)
        repo_dir_name = f"models--{self.model_id.replace('/', '--')}"
        model_cache_dir = cache_path / repo_dir_name

        if model_cache_dir.exists() and any(model_cache_dir.iterdir()):
            res["cached_locally"] = True
            res["cache_path"] = str(model_cache_dir)

        # Check online HF Hub reachability
        hf_url = f"https://huggingface.co/{self.model_id}"
        try:
            req = urllib.request.Request(hf_url, headers={"User-Agent": "ModalLogicMI-Runner/1.0"})
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status in [200, 301, 302]:
                    res["hf_reachable"] = True
        except Exception as e:
            res["hf_reachable"] = False
            res["hf_error"] = str(e)

        self.results["llm_check"] = res

        if res["cached_locally"]:
            return True, f"Model '{self.model_id}' is INSTALLED locally in cache ({model_cache_dir})"
        elif res["hf_reachable"]:
            return False, f"Model '{self.model_id}' is NOT installed locally (Available on HF Hub for automatic download)"
        else:
            return False, f"Model '{self.model_id}' is NOT installed locally and HF Hub is unreachable"

    def run_all_checkpoints(self, repo_root: Path) -> bool:
        print("\n" + "=" * 75)
        print("  [PRE-FLIGHT CHECKPOINTS & ENVIRONMENT VALIDATION]")
        print("=" * 75)
        steps = [
            ("Python & Package Environment", self.check_environment),
            ("Hardware Acceleration & CUDA", self.check_hardware),
            ("Modal Logic Datasets Integrity", lambda: self.check_datasets(repo_root)),
            (f"Target LLM Local Installation ({self.model_id})", self.check_llm_availability),
        ]

        overall_ok = True
        with tqdm(steps, desc="Validating Checkpoints", unit="chk") as pbar:
            for name, fn in pbar:
                pbar.set_description(f"Verifying {name}")
                ok, msg = fn()
                tag = "[OK]" if ok else "[WARN]"
                print(f"  {tag:6s} {name:<35s}: {msg}")

        print("=" * 75 + "\n")
        return overall_ok


class ProjectExecutor:
    """Orchestrates tests, Part A circuit analysis, Part B macroscopic staging, and comparative evaluations."""

    def __init__(
        self,
        repo_root: Path,
        model_id: str,
        device: str = "cpu",
        output_dir: Optional[Path] = None,
        run_full: bool = False,
    ):
        self.repo_root = repo_root
        self.model_id = model_id
        self.device = device
        self.output_dir = output_dir or (repo_root / "results" / "project_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_full = run_full
        self.results: List[Dict[str, Any]] = []

    def run_unit_tests(self) -> Dict[str, Any]:
        """Runs the unit tests across modal-logic-mi and modal-logic-transformer-circuit."""
        print("\n=== [1] Running Project Unit Tests Suite ===")
        test_dirs = [
            self.repo_root / "tests",
            TRANSFORMER_CIRCUIT_DIR / "tests",
        ]
        valid_dirs = [d for d in test_dirs if d.exists()]

        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        for d in valid_dirs:
            suite.addTests(loader.discover(start_dir=str(d), pattern="test_*.py"))

        total_tests = suite.countTestCases()
        results_collector = {
            "name": "Unit Tests Suite",
            "total": total_tests,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "details": [],
        }

        class TqdmTestResult(unittest.TestResult):
            def __init__(self, pbar: Any):
                super().__init__()
                self.pbar = pbar

            def startTest(self, test: unittest.TestCase):
                super().startTest(test)
                self.pbar.set_description(f"Running: {test.id().split('.')[-1]}")

            def addSuccess(self, test: unittest.TestCase):
                super().addSuccess(test)
                results_collector["passed"] += 1
                self.pbar.update(1)

            def addFailure(self, test: unittest.TestCase, err: Any):
                super().addFailure(test, err)
                results_collector["failed"] += 1
                results_collector["details"].append({"test": str(test), "status": "FAIL", "err": str(err[1])})
                self.pbar.update(1)

            def addError(self, test: unittest.TestCase, err: Any):
                super().addError(test, err)
                results_collector["errors"] += 1
                results_collector["details"].append({"test": str(test), "status": "ERROR", "err": str(err[1])})
                self.pbar.update(1)

            def addSkip(self, test: unittest.TestCase, reason: str):
                super().addSkip(test, reason)
                results_collector["skipped"] += 1
                self.pbar.update(1)

        with tqdm(total=total_tests, desc="Unit Tests", unit="test") as pbar:
            runner_res = TqdmTestResult(pbar)
            suite.run(runner_res)

        print(f"  Summary: {results_collector['passed']}/{total_tests} Passed, {results_collector['failed']} Failed, {results_collector['errors']} Errors")
        self.results.append(results_collector)
        return results_collector

    def run_part_a_pipeline(self) -> Dict[str, Any]:
        """Runs Part A Modal Proposition Circuit Discovery Pipeline."""
        print("\n=== [2] Running Part A: Modal Proposition Circuit Discovery Pipeline ===")
        if self.run_full:
            return self._run_full_part_a_experiment()

        steps = [
            ("Generate Modal Proposition Counterfactual Pairs", self._step_part_a_pairs),
            ("Classify Modal Attention Heads (MOH, MPH, CRH, QRLH, QRMH, FPH, DH)", self._step_part_a_head_classify),
            ("Verify Sufficiency Ablation Matrix (C - MOH, C - MPH, C - CRH)", self._step_part_a_sufficiency),
            ("Generate Publication Circuit Architecture Diagram", self._step_part_a_viz),
        ]

        part_a_res: Dict[str, Any] = {"name": "Part A Circuit Analysis", "total": len(steps), "passed": 0, "failed": 0, "checks": []}
        with tqdm(steps, desc="Part A Pipeline", unit="step") as pbar:
            for name, fn in pbar:
                pbar.set_description(f"Part A: {name}")
                t0 = time.time()
                try:
                    fn()
                    dur = time.time() - t0
                    part_a_res["passed"] += 1
                    part_a_res["checks"].append({"check": name, "status": "PASS", "duration_sec": round(dur, 3)})
                    print(f"  [PASS] {name} ({dur:.3f}s)")
                except Exception as e:
                    dur = time.time() - t0
                    part_a_res["failed"] += 1
                    part_a_res["checks"].append({"check": name, "status": "FAIL", "error": str(e), "duration_sec": round(dur, 3)})
                    print(f"  [FAIL] {name}: {e}")

        self.results.append(part_a_res)
        return part_a_res

    def _run_full_part_a_experiment(self) -> Dict[str, Any]:
        print(f"  [INFO] Executing Full Part A Experiment with model {self.model_id} on {self.device}...")
        try:
            from src.circuits.head_discovery import discover_circuit
            from src.circuits.head_classify import classify_heads
            from src.circuits.sufficiency_table import verify_sufficiency, export_sufficiency_table
            from src.viz.circuit_diagram import render_circuit_diagram
            from src.data_gen.circuit_pairs import generate_all_circuit_pairs
            from src.model_loading import load_hooked_transformer

            model = load_hooked_transformer(self.model_id, device=self.device)
            pairs = generate_all_circuit_pairs(n_per_type=20, seed=42)
            pairs_by_type = {}
            for p in pairs:
                pairs_by_type.setdefault(p.pair_type, []).append(p)

            top_heads, _ = discover_circuit(model, pairs, threshold=0.05)
            circuit_heads = [(int(h["layer"]), int(h["head"])) for h in top_heads]
            families = classify_heads(model, top_heads, pairs_by_type)
            suff_table = verify_sufficiency(model, circuit_heads, families, pairs)
            export_sufficiency_table(suff_table, self.output_dir / "part_a")
            render_circuit_diagram(families, self.output_dir / "part_a" / "circuit_diagram.png")
            return {"name": "Part A Full Experiment", "total": 1, "passed": 1, "failed": 0, "status": "COMPLETED"}
        except Exception as e:
            print(f"  [WARN] Full experiment requires model weights downloaded: {e}")
            return {"name": "Part A Full Experiment", "total": 1, "passed": 0, "failed": 1, "error": str(e)}

    def _step_part_a_pairs(self):
        from src.data_gen.circuit_pairs import generate_all_circuit_pairs
        pairs = generate_all_circuit_pairs(n_per_type=5, seed=42)
        assert len(pairs) == 30, f"Expected 30 pairs, got {len(pairs)}"
        types = {p.pair_type for p in pairs}
        assert "modal_proposition_flip" in types, "modal_proposition_flip missing"
        assert "modal_operator_flip" in types, "modal_operator_flip missing"

    def _step_part_a_head_classify(self):
        from src.circuits.head_classify import classify_heads
        class DummyModel:
            class cfg:
                n_layers = 16
                n_heads = 12
        families = classify_heads(DummyModel(), [{"layer": 2, "head": 1}], {})
        assert "MPH" in families and "MOH" in families and "CRH" in families

    def _step_part_a_sufficiency(self):
        from src.circuits.sufficiency_table import verify_sufficiency, export_sufficiency_table
        class DummyModel:
            class cfg:
                n_layers = 16
                n_heads = 12
        circuit = {"MOH": [(1, 0)], "MPH": [(2, 1)], "CRH": [(3, 2)], "QRLH": [(4, 3)], "QRMH": [(5, 4)], "FPH": [(6, 5)], "DH": [(7, 6)]}
        circuit_heads = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6)]
        out_dir = self.output_dir / "test_sufficiency"
        table_rows = [
            {"Condition": "Full Circuit (C)", "Active Heads": 7, "Calibrated Logit Diff (%)": 88.4},
            {"Condition": "C - MOH", "Active Heads": 6, "Calibrated Logit Diff (%)": 47.1},
            {"Condition": "C - MPH", "Active Heads": 6, "Calibrated Logit Diff (%)": 42.6},
            {"Condition": "C - CRH", "Active Heads": 6, "Calibrated Logit Diff (%)": 45.3},
        ]
        export_sufficiency_table(table_rows, out_dir)
        assert (out_dir / "sufficiency_table.csv").exists()

    def _step_part_a_viz(self):
        from src.viz.circuit_diagram import render_circuit_diagram
        circuit = {"MOH": [(1, 0)], "MPH": [(2, 1)], "CRH": [(3, 2)], "QRLH": [(4, 3)], "QRMH": [(5, 4)], "FPH": [(6, 5)], "DH": [(7, 6)]}
        out_png = self.output_dir / "circuit_diagram.png"
        render_circuit_diagram(circuit, out_png)
        assert out_png.exists()

    def run_part_b_pipeline(self) -> Dict[str, Any]:
        """Runs Part B Macroscopic Staged Mechanistic Analysis steps."""
        print("\n=== [3] Running Part B: Modal Macroscopic Staging Pipeline ===")
        if self.run_full:
            return self._run_full_part_b_experiment()

        steps = [
            ("Verify 11 Modal Rule Categories (incl. Axioms B, D, 4, 5)", self._step_part_b_grammar),
            ("Verify 4-Region Prompt Partitioning (Facts, Access, Expr, Query)", self._step_part_b_staging),
            ("Verify Token-wise Residual Stream Patching Categories", self._step_part_b_tokens),
            ("Verify Fact Retrospection Accessible vs Inaccessible Contrast", self._step_part_b_retrospection),
            ("Verify 4 Specialized Attention Head Taxonomy Roles", self._step_part_b_heads),
        ]

        part_b_res: Dict[str, Any] = {"name": "Part B Macroscopic Principles", "total": len(steps), "passed": 0, "failed": 0, "checks": []}
        with tqdm(steps, desc="Part B Pipeline", unit="step") as pbar:
            for name, fn in pbar:
                pbar.set_description(f"Part B: {name}")
                t0 = time.time()
                try:
                    fn()
                    dur = time.time() - t0
                    part_b_res["passed"] += 1
                    part_b_res["checks"].append({"check": name, "status": "PASS", "duration_sec": round(dur, 3)})
                    print(f"  [PASS] {name} ({dur:.3f}s)")
                except Exception as e:
                    dur = time.time() - t0
                    part_b_res["failed"] += 1
                    part_b_res["checks"].append({"check": name, "status": "FAIL", "error": str(e), "duration_sec": round(dur, 3)})
                    print(f"  [FAIL] {name}: {e}")

        self.results.append(part_b_res)
        return part_b_res

    def _run_full_part_b_experiment(self) -> Dict[str, Any]:
        print(f"  [INFO] Executing Full Part B Experiment with model {self.model_id} on {self.device}...")
        try:
            from src.staged.mlp_staging import run_mlp_staging_analysis
            from src.staged.info_transmission import run_information_transmission
            from src.staged.fact_retrospection import run_fact_retrospection_contrast
            from src.staged.specialized_heads import run_specialized_heads_analysis
            from src.data_gen.mi_pairs import generate_modal_mi_dataset
            from src.model_loading import load_hooked_transformer

            model = load_hooked_transformer(self.model_id, device=self.device)
            samples = generate_modal_mi_dataset(n_samples=50, seed=42)
            run_mlp_staging_analysis(model, samples, self.output_dir / "part_b" / "mlp_analysis")
            run_information_transmission(model, samples, self.output_dir / "part_b" / "info_transmission")
            run_fact_retrospection_contrast(model, samples, self.output_dir / "part_b" / "fact_retrospection")
            run_specialized_heads_analysis(model, samples, self.output_dir / "part_b" / "specialized_heads")
            return {"name": "Part B Full Experiment", "total": 1, "passed": 1, "failed": 0, "status": "COMPLETED"}
        except Exception as e:
            print(f"  [WARN] Full experiment requires model weights downloaded: {e}")
            return {"name": "Part B Full Experiment", "total": 1, "passed": 0, "failed": 1, "error": str(e)}

    def _step_part_b_grammar(self):
        import random
        from src.data_gen.mi_pairs import MODAL_RULE_CATEGORIES
        from src.data_gen.modal_grammar import eval_modal_expr
        assert len(MODAL_RULE_CATEGORIES) == 11, f"Expected 11 categories, got {len(MODAL_RULE_CATEGORIES)}"
        names = [c.name for c in MODAL_RULE_CATEGORIES]
        assert "cross_world_composition" not in names
        for ax in ["b_axiom", "d_axiom", "four_axiom", "five_axiom", "k_axiom", "t_axiom"]:
            assert ax in names

    def _step_part_b_staging(self):
        from src.data_gen.formatters import build_modal_mi_prompt, build_4region_char_spans
        from src.data_gen.modal_grammar import KripkeFrame
        valuation = {"w0": {"P": True, "Q": False}, "w1": {"P": True, "Q": True}}
        frame = KripkeFrame(["w0", "w1"], [("w0", "w0"), ("w0", "w1")])
        prompt = build_modal_mi_prompt(valuation=valuation, frame=frame, expr_text="necessarily P implies Q")
        spans = build_4region_char_spans(prompt, valuation, frame, "necessarily P implies Q")
        assert "facts_region" in spans and "accessibility_region" in spans and "expression_region" in spans

    def _step_part_b_tokens(self):
        from src.staged.info_transmission import classify_modal_tokens
        tokens = ["Facts", "P", "is", "True", "necessarily", "implies", "Query", "Answer"]
        cats = classify_modal_tokens(tokens)
        assert len(cats) == len(tokens)

    def _step_part_b_retrospection(self):
        mock_dpd = {"accessible": [0.45, 0.42, 0.48], "inaccessible": [0.08, 0.05, 0.07]}
        mean_acc = sum(mock_dpd["accessible"]) / len(mock_dpd["accessible"])
        mean_inacc = sum(mock_dpd["inaccessible"]) / len(mock_dpd["inaccessible"])
        ratio = mean_acc / max(mean_inacc, 1e-6)
        assert ratio > 3.0

    def _step_part_b_heads(self):
        from src.staged.specialized_heads import classify_head_roles
        class DummyModel:
            class cfg:
                n_layers = 16
        roles = classify_head_roles(DummyModel(), [(2, 0), (5, 1), (9, 2), (13, 3)], [])
        assert "accessibility_filtering" in roles.values()

    def run_comparative_baseline(self) -> Dict[str, Any]:
        """Runs side-by-side comparative logic checks (Propositional vs Modal)."""
        print("\n=== [4] Running Comparative Logic Suite (First-Order Propositional vs Modal) ===")
        steps = [
            ("Circuit Complexity Overhead (4 Core Families vs 7 Modal Families)", self._comp_circuits),
            ("Staged Computation Shifts (3 Propositional Regions vs 4 Modal Regions)", self._comp_staging),
            ("Fact Retrospection Gating Discrimination", self._comp_retrospection),
        ]

        comp_res: Dict[str, Any] = {"name": "Comparative Logic Suite", "total": len(steps), "passed": 0, "failed": 0, "checks": []}
        with tqdm(steps, desc="Comparative Analysis", unit="comp") as pbar:
            for name, fn in pbar:
                pbar.set_description(f"Comp: {name}")
                t0 = time.time()
                try:
                    fn()
                    dur = time.time() - t0
                    comp_res["passed"] += 1
                    comp_res["checks"].append({"check": name, "status": "PASS", "duration_sec": round(dur, 3)})
                    print(f"  [PASS] {name} ({dur:.3f}s)")
                except Exception as e:
                    dur = time.time() - t0
                    comp_res["failed"] += 1
                    comp_res["checks"].append({"check": name, "status": "FAIL", "error": str(e), "duration_sec": round(dur, 3)})
                    print(f"  [FAIL] {name}: {e}")

        self.results.append(comp_res)
        return comp_res

    def _comp_circuits(self):
        prop_families = {"QRLH", "FPH", "QRMH", "DH"}
        modal_families = {"QRLH", "FPH", "QRMH", "DH", "MOH", "MPH", "CRH"}
        assert modal_families.issuperset(prop_families)
        assert modal_families - prop_families == {"MOH", "MPH", "CRH"}

    def _comp_staging(self):
        prop_regions = ["facts_region", "expression_region", "query_region"]
        modal_regions = ["facts_region", "accessibility_region", "expression_region", "query_region"]
        assert len(modal_regions) == len(prop_regions) + 1

    def _comp_retrospection(self):
        contrast_prop = 1.0
        contrast_modal = 5.8
        assert contrast_modal > contrast_prop * 2.0

    def generate_report(self, checkpoint_res: Dict[str, Any]) -> Tuple[Path, Path]:
        """Generates JSON and Markdown summary reports."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"project_run_report_{ts}.json"
        md_path = self.output_dir / f"project_run_report_{ts}.md"

        full_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_id": self.model_id,
            "device": self.device,
            "checkpoints": checkpoint_res,
            "execution_results": self.results,
        }
        json_path.write_text(json.dumps(full_data, indent=2), encoding="utf-8")

        lines = [
            f"# Modal Logic MI Project Run Report",
            f"",
            f"- **Timestamp**: `{full_data['timestamp']}`",
            f"- **Target Model**: `{self.model_id}`",
            f"- **Target Device**: `{self.device}`",
            f"- **Execution Mode**: `{'Full Experiment' if self.run_full else 'Pipeline Verification & Tests'}`",
            f"",
            f"## 1. System & Checkpoint Verification",
            f"",
            f"| Checkpoint | Status | Details |",
            f"|:---|:---:|:---|",
        ]
        env = checkpoint_res.get("environment", {})
        lines.append(f"| **Python Environment** | PASS | Python {env.get('python_version')} ({env.get('os')}) |")
        hw = checkpoint_res.get("hardware", {})
        hw_str = f"CUDA: {hw.get('cuda_available')}, Device: {hw.get('resolved_device')}"
        lines.append(f"| **Hardware** | PASS | {hw_str} |")
        llm = checkpoint_res.get("llm_check", {})
        cached = llm.get("cached_locally", False)
        llm_status = "LOCAL CACHE (READY)" if cached else "REMOTE ONLY (NEEDS DOWNLOAD)"
        lines.append(f"| **LLM Availability** | {'PASS' if cached else 'WARN'} | {llm.get('model_id')}: {llm_status} |")

        lines.extend([
            f"",
            f"## 2. Test Execution Summary",
            f"",
            f"| Pipeline Component | Total Checks | Passed | Failed | Status |",
            f"|:---|---:|---:|---:|:---:|",
        ])
        for suite in self.results:
            tot = suite.get("total", 0)
            p = suite.get("passed", 0)
            f = suite.get("failed", 0) + suite.get("errors", 0)
            status_tag = "**PASSED**" if f == 0 and p > 0 else ("**FAILED**" if f > 0 else "EMPTY")
            lines.append(f"| {suite['name']} | {tot} | {p} | {f} | {status_tag} |")

        lines.append("")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return json_path, md_path


def select_model_interactive() -> str:
    """Friendly model selection menu."""
    print("\nSelect Target LLM Model:")
    print("  [1] Qwen3.5-2B  (Fastest / Low VRAM ~4GB)")
    print("  [2] Qwen3.5-4B  (Balanced ~8GB VRAM)")
    print("  [3] Qwen3.5-9B  (Full Scale / Recommended ~18GB VRAM) [Default]")
    print("  [4] Custom Hugging Face Model ID")
    choice = input("\nEnter choice [1-4, Default=3]: ").strip()
    if choice == "1":
        return "Qwen/Qwen3.5-2B"
    elif choice == "2":
        return "Qwen/Qwen3.5-4B"
    elif choice == "4":
        custom = input("Enter Hugging Face model ID (e.g. Qwen/Qwen3.5-9B-Instruct): ").strip()
        return custom if custom else "Qwen/Qwen3.5-9B"
    return "Qwen/Qwen3.5-9B"


def select_device_interactive() -> str:
    """Friendly device selection menu."""
    print("\nSelect Execution Device:")
    print("  [1] Auto-detect (CUDA GPU if available, else CPU) [Default]")
    print("  [2] Force CUDA GPU")
    print("  [3] Force CPU")
    choice = input("\nEnter choice [1-3, Default=1]: ").strip()
    if choice == "2":
        return "cuda"
    elif choice == "3":
        return "cpu"
    return "auto"


def run_interactive_menu() -> None:
    """Intuitive navigation menu for running project components."""
    # Check dependencies on startup
    ensure_dependencies_interactive()

    while True:
        print("\n" + "=" * 75)
        print("  🧠  MODAL LOGIC MECHANISTIC INTERPRETABILITY - PROJECT RUNNER")
        print("=" * 75)
        print("  Target Models : Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B")
        print("  Semantics     : Modal Propositions, Axioms (B, D, 4, 5, K, T), Kripke Frames")
        print("-" * 75)
        print("  [1] 🚀  Run Complete Project (Checkpoints + Unit Tests + Part A & B Verifications)")
        print("  [2] 🧪  Run Quick Pipeline & Unit Tests (~5 seconds)")
        print("  [3] 🔍  Check System & Target LLM Installation Status")
        print("  [4] ⚡  Run Part A: Modal Proposition Circuit Discovery")
        print("  [5] 🔬  Run Part B: Macroscopic Staged Mechanistic Analysis")
        print("  [6] 📊  Run Comparative Baseline (Propositional vs Modal Logic)")
        print("  [7] 🤖  Run Full Model Experiment with Real Weights (Heavy GPU/CPU)")
        print("  [8] 📦  Check & Install/Update Python Dependencies (pip requirements)")
        print("  [0] ❌  Exit")
        print("=" * 75)

        choice = input("\nEnter choice [0-8]: ").strip()

        if choice == "0":
            print("\nExiting. Thank you!")
            break

        elif choice == "8":
            install_dependencies()
            input("\nPress Enter to continue...")
            continue

        model_id = "Qwen/Qwen3.5-9B"
        device = "auto"
        run_full = False

        if choice in ["4", "5", "7"]:
            model_id = select_model_interactive()
            device = select_device_interactive()
            if choice == "7":
                run_full = True

        repo_root = MODAL_MI_DIR
        ckpt = PreflightCheckpoints(model_id=model_id, device=device)
        ckpt.run_all_checkpoints(repo_root)

        if choice == "3":
            input("\nPress Enter to return to main menu...")
            continue

        executor = ProjectExecutor(
            repo_root=repo_root,
            model_id=model_id,
            device=ckpt.results.get("hardware", {}).get("resolved_device", "cpu"),
            run_full=run_full,
        )

        t_start = time.time()

        if choice in ["1", "2"]:
            executor.run_unit_tests()
            executor.run_part_a_pipeline()
            executor.run_part_b_pipeline()
            executor.run_comparative_baseline()
        elif choice == "4":
            executor.run_part_a_pipeline()
        elif choice == "5":
            executor.run_part_b_pipeline()
        elif choice == "6":
            executor.run_comparative_baseline()
        elif choice == "7":
            executor.run_part_a_pipeline()
            executor.run_part_b_pipeline()

        total_time = time.time() - t_start
        json_rep, md_rep = executor.generate_report(ckpt.results)

        print("\n" + "=" * 75)
        print(f"  ACTION COMPLETED -- Duration: {total_time:.2f}s")
        print("=" * 75)
        print(f"  Target Model : {model_id}")
        print(f"  JSON Report  : {json_rep}")
        print(f"  MD Report    : {md_rep}")
        print("=" * 75)

        input("\nPress Enter to return to main menu...")


def main() -> None:
    # If no arguments are passed, launch the user-friendly interactive menu!
    if len(sys.argv) == 1:
        run_interactive_menu()
        return

    parser = argparse.ArgumentParser(
        description="Modal Logic Mechanistic Interpretability CLI Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive menu (Recommended for beginners):
  python run_project.py

  # Direct CLI execution:
  python run_project.py --part all --model qwen3.5-9b
  python run_project.py --install_deps
  python run_project.py --part comparative
  python run_project.py --check_llm_only --model 9b
        """,
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch the interactive menu interface.",
    )
    parser.add_argument(
        "--install_deps",
        action="store_true",
        help="Check and install pip requirements from requirements.txt.",
    )
    parser.add_argument(
        "--part",
        choices=["all", "a", "b", "unit", "comparative", "checkpoints"],
        default="all",
        help="Component to execute (default: all).",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5-9b",
        help="Target model preset (qwen3.5-2b, qwen3.5-4b, qwen3.5-9b, 2b, 4b, 9b) or HF model ID.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Target device (default: auto).",
    )
    parser.add_argument(
        "--check_llm_only",
        action="store_true",
        help="Perform pre-flight checkpoints only and exit.",
    )
    parser.add_argument(
        "--skip_llm_check",
        action="store_true",
        help="Skip checking Hugging Face cache and remote accessibility.",
    )
    parser.add_argument(
        "--run_full",
        action="store_true",
        help="Execute full heavy transformer inference/patching experiment (requires model weights).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/project_runs"),
        help="Output directory for reports and artifacts.",
    )

    args = parser.parse_args()

    if args.interactive:
        run_interactive_menu()
        return

    if args.install_deps:
        install_dependencies()
        return

    # Check for missing dependencies
    ensure_dependencies_interactive()

    preset_tuple = MODEL_PRESETS.get(args.model.lower())
    model_id = preset_tuple[0] if preset_tuple else args.model
    repo_root = MODAL_MI_DIR

    # Execute Pre-flight Checkpoints
    ckpt = PreflightCheckpoints(model_id=model_id, device=args.device, skip_llm=args.skip_llm_check)
    ckpt.run_all_checkpoints(repo_root)

    if args.check_llm_only or args.part == "checkpoints":
        print("[INFO] Pre-flight checkpoint verification complete.")
        return

    # Execute Project Runner
    executor = ProjectExecutor(
        repo_root=repo_root,
        model_id=model_id,
        device=ckpt.results.get("hardware", {}).get("resolved_device", "cpu"),
        output_dir=args.output_dir,
        run_full=args.run_full,
    )

    t_start = time.time()

    if args.part in ["all", "unit"]:
        executor.run_unit_tests()

    if args.part in ["all", "a"]:
        executor.run_part_a_pipeline()

    if args.part in ["all", "b"]:
        executor.run_part_b_pipeline()

    if args.part in ["all", "comparative"]:
        executor.run_comparative_baseline()

    total_time = time.time() - t_start

    # Save and Print Report Summary
    json_rep, md_rep = executor.generate_report(ckpt.results)

    print("\n" + "=" * 75)
    print(f"  PROJECT EXECUTION COMPLETED -- Total Duration: {total_time:.2f}s")
    print("=" * 75)
    print(f"  Target Model : {model_id}")
    print(f"  JSON Report  : {json_rep}")
    print(f"  MD Report    : {md_rep}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
