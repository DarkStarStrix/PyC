from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "infra" / "nexa_insight" / "app.py"


def load_app_module():
    spec = importlib.util.spec_from_file_location("nexa_insight_app", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        return None
    return module


def test_parse_ncu_csv_groups_metrics_per_kernel():
    app = load_app_module()
    if app is None:
        return

    csv_text = "\n".join(
        [
            '"ID","Process ID","Process Name","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"',
            '"1","100","python","kernelA","LaunchStats","gpu__time_duration.sum","nsecond","91000000"',
            '"1","100","python","kernelA","SpeedOfLight","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","76.5"',
            '"1","100","python","kernelA","MemoryWorkloadAnalysis","l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum","sector","12345"',
            '"2","100","python","kernelB","LaunchStats","gpu__time_duration.sum","nsecond","1500000"',
            '"2","100","python","kernelB","SpeedOfLight","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","12.0"',
        ]
    )

    records = app.parse_ncu_csv(csv_text)

    assert len(records) == 2
    assert records[0].kernel_name == "kernelA"
    assert records[0].total_time_ms == 91.0
    assert records[0].throughput_pct == 76.5
    assert records[0].global_loads == 12345.0


def test_append_history_deduplicates_and_keeps_latest(tmp_path: Path):
    app = load_app_module()
    if app is None:
        return

    history_path = tmp_path / "history.json"
    history = ["python train.py", "./bench"]

    history = app.append_history(history_path, history, "python train.py")

    assert history[-1] == "python train.py"
    assert history.count("python train.py") == 1


def _static_markup_map(source_path: Path) -> dict[str, bool]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    mapping: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "Static":
            continue
        panel_id = None
        markup = True
        for keyword in node.keywords:
            if keyword.arg == "id" and isinstance(keyword.value, ast.Constant):
                panel_id = str(keyword.value.value)
            if keyword.arg == "markup" and isinstance(keyword.value, ast.Constant):
                markup = bool(keyword.value.value)
        if panel_id:
            mapping[panel_id] = markup
    return mapping


def test_local_tui_raw_panels_disable_markup():
    source = ROOT / "infra" / "nexa_insight" / "local_tui.py"
    markup = _static_markup_map(source)
    assert markup["summary"] is False
    assert markup["progress"] is False
    assert markup["events"] is False
    assert markup["tail"] is False
    assert markup["result_summary"] is False
    assert markup["profile_summary"] is False


def test_app_kernel_detail_modal_disables_markup():
    app = load_app_module()
    if app is None:
        return

    source = ROOT / "infra" / "nexa_insight" / "app.py"
    markup = _static_markup_map(source)
    assert markup["kernel-detail-body"] is False
