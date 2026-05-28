"""Build an integrated real/synthetic story report for the Bayesian mainline."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_search = load_json(Path(args.real_search_report))
    real_final = load_json(Path(args.real_final_report))
    synthetic = load_json(Path(args.synthetic_report))
    payload = {
        "inputs": {
            "real_search_report": str(Path(args.real_search_report).resolve()),
            "real_final_report": str(Path(args.real_final_report).resolve()),
            "synthetic_report": str(Path(args.synthetic_report).resolve()),
        },
        "real_search_summary": real_search.get("final_summary", {}),
        "real_final_summary": real_final.get("final_summary", {}),
        "synthetic_summary": synthetic.get("final_summary", {}),
        "synthetic_truth": synthetic.get("synthetic_truth", {}),
    }
    (output_dir / "integrated_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "integrated_report.html").write_text(
        render_integrated_report(real_search, real_final, synthetic, payload["inputs"]),
        encoding="utf-8",
    )
    print("Integrated Bayesian calibration story report complete.")
    print(f"  output: {(output_dir / 'integrated_report.html').resolve()}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def render_integrated_report(
    real_search: dict[str, Any],
    real_final: dict[str, Any],
    synthetic: dict[str, Any],
    inputs: dict[str, str],
) -> str:
    real_search_summary = real_search["final_summary"]
    real_final_summary = real_final["final_summary"]
    synthetic_summary = synthetic["final_summary"]
    truth = synthetic.get("synthetic_truth", {})
    sigma_mm = real_final.get("settings", {}).get("statistical_residual", {}).get("noise_std_mm", 0.06)
    fixed_ratio = real_final_summary.get("fine_tune_lambda_ratio", float("nan"))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Bayesian Calibration Integrated Real/Synthetic Story</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 28px; color: #1f2933; line-height: 1.55; }}
    h1, h2, h3 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    code {{ background: #f0f4f8; padding: 1px 4px; border-radius: 3px; }}
    .note {{ background: #f8fafc; border-left: 4px solid #486581; padding: 10px 14px; margin: 12px 0 20px; }}
    .warn {{ color: #9a6700; font-weight: 700; }}
    .bad {{ color: #b42318; font-weight: 700; }}
    .ok {{ color: #0b6b3a; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>璐濆彾鏂潪鍑犱綍鏍囧畾锛氱湡瀹炴暟鎹笌浠跨湡鏁版嵁鏁村悎鎶ュ憡</h1>
  <div class="note">
    鏈姤鍛婃暣鍚堜笁浠朵簨锛氱湡瀹炴暟鎹笂鐨?Stage 4 姝ｅ垯姣斾緥鎼滅储銆佸浐鍖栨瘮渚嬪悗鐨勭湡瀹炴暟鎹富绾垮璺戙€佷互鍙?54 缁磋宸ā鍨嬬敓鎴愮殑浠跨湡 A/B/C 楠岃瘉銆?    涓荤嚎鍥哄畾涓?<b>S1 鍑犱綍閿氱偣 + Bayesian basis 闈炲嚑浣曟畫宸鲸璇?+ 33 鍑犱綍鍙傛暟鍙楁帶 MAP 寰皟</b>銆?    Stage 1 鍜?Stage 4 閮戒娇鐢ㄦ湭闄?sigma 鐨勪綅缃畫宸紱<code>sigma={float(sigma_mm):.3g} mm</code> 鍙綔涓鸿礉鍙舵柉娈嬪樊鍣０鍏堥獙鍜屽彈鎺ч槇鍊煎昂搴︺€?  </div>

  <h2>绠楁硶鏁呬簨绾?/h2>
  <p>娴嬮噺妯″瀷鍐欎负 <code>y_k = p(q_k; theta) + Phi(x_k) beta + eps_k</code>銆?  绗竴闃舵鐢?S1 鍦ㄥ嚑浣?33 鍙傛暟涓緱鍒板彈鎺ч敋鐐?<code>theta_anchor</code>锛?  绗簩銆佷笁闃舵鐢ㄦ湁闄愩€佸彲瑙ｉ噴鐨?Bayesian basis 鍑芥暟搴撴嫙鍚堥敋鐐瑰悗鐨勬畫宸粨鏋勶紱
  绗洓闃舵閲婃斁鍏ㄩ儴 33 涓嚑浣曞弬鏁板仛灏忚寖鍥?MAP 寰皟锛?/p>
  <p><code>min ||y - p(q;theta) - Phi(x)beta||^2 + lambda_theta ||D^-1(theta-theta_anchor)||^2 + beta^T Lambda beta</code></p>
  <p>鍏朵腑 <code>lambda_theta = lambda_stage1 * ratio</code>銆傛湰杞湡瀹炴暟鎹悳绱㈠悗锛岄粯璁ゅ浐鍖栨瘮渚嬩负 <code>{fmt(fixed_ratio)}</code>銆?/p>

  <h2>鐪熷疄鏁版嵁锛氭瘮渚嬫悳绱㈢粨鏋?/h2>
  <p>姣斾緥鎼滅储鍙墽琛屼竴娆★紝鐢ㄦ潵鍐冲畾鍚庣画榛樿姣斾緥锛涜〃涓?peeled 鎸囧墺绂?<code>Phi beta</code> 鍚庡彧鐪嬪嚑浣曟湰浣撱€?/p>
  {table(real_search.get("bayesian_residual", {}).get("fine_tune", {}).get("search_rows", []),
         ["ratio_to_stage1_lambda", "lambda_theta", "post_A_train_rmse_mm", "post_B_train_rmse_mm", "post_A_C_rmse_mm", "post_B_C_rmse_mm", "post_C_all_rmse_mm", "peeled_C_all_rmse_mm", "peeled_C_all_delta_vs_anchor_mm", "allowed_peeled_delta_mm", "theta_update_scaled_l2", "controlled_geometry"])}

  <h2>鐪熷疄鏁版嵁锛氬浐鍖栨瘮渚嬪悗鐨勪富绾?/h2>
  {summary_table("鐪熷疄鏁版嵁鏈€缁堜富绾?, real_final_summary)}
  <h3>Stage 1 鍑犱綍娑堣瀺</h3>
  {table(real_final.get("stage1_geometry", {}).get("split_metrics", []),
         ["method", "label", "lambda", "active_count", "A_train_rmse_mm", "B_train_rmse_mm", "A_C_rmse_mm", "B_C_rmse_mm", "C_all_rmse_mm"])}
  <h3>Bayesian 鍊欓€夊嚱鏁颁笌閫変腑鍑芥暟</h3>
  {table(real_final.get("bayesian_residual", {}).get("candidate_library", []), None)}
  {table(real_final.get("bayesian_residual", {}).get("models", {}).get("bayesian_basis", {}).get("selected_groups", []),
         ["name", "label", "formula", "columns", "prior_std_mm"])}
  <h3>A/B/C 闃舵绮惧害</h3>
  {table(real_final.get("bayesian_stage_metrics", []), None)}

  <h2>浠跨湡鏁版嵁锛?4 缁磋宸ā鍨嬩笌 A/B/C 鏋勯€?/h2>
  <p>浠跨湡鐢ㄤ簬楠岃瘉鏁呬簨绾挎槸鍚﹁嚜娲斤細鍑犱綍璇樊鍦?A/B/C 涓畬鍏ㄧ浉鍚岋紝闈炲嚑浣曡宸湪涓嶅悓瀛愮┖闂翠腑涓嶅悓锛涘洜姝ゅ悎鐞嗙畻娉曞簲鍏堟壘鍒板叡浜嚑浣曢敋鐐癸紝鍐嶇敤缁熻娈嬪樊瑙ｉ噴瀛愮┖闂寸浉鍏崇殑闈炲嚑浣曠粨鏋勩€?/p>
  {table([truth], ["truth_model", "geometric_parameter_rule", "nongeometric_parameter_rule", "geometric_max_abs_delta_across_spaces", "noise_std_mm", "payload_kg"])}
  <h3>闈炲嚑浣曠湡鍊煎樊寮傛鏌?/h3>
  {table(truth.get("nongeometric_pair_differences", []), None)}

  <h2>浠跨湡鏁版嵁锛氫富绾跨粨鏋?/h2>
  {summary_table("浠跨湡鏈€缁堜富绾?, synthetic_summary)}
  <h3>Stage 1 鍑犱綍娑堣瀺</h3>
  {table(synthetic.get("stage1_geometry", {}).get("split_metrics", []),
         ["method", "label", "lambda", "active_count", "A_train_rmse_mm", "B_train_rmse_mm", "A_C_rmse_mm", "B_C_rmse_mm", "C_all_rmse_mm"])}
  <h3>Bayesian 閫変腑鍑芥暟</h3>
  {table(synthetic.get("bayesian_residual", {}).get("models", {}).get("bayesian_basis", {}).get("selected_groups", []),
         ["name", "label", "formula", "columns", "prior_std_mm"])}
  <h3>A/B/C 闃舵绮惧害</h3>
  {table(synthetic.get("bayesian_stage_metrics", []), None)}

  <h2>鍏卞悓缁撹涓庤竟鐣?/h2>
  <ul>
    <li>鑻?<code>Bayesian basis</code> 鐩告瘮 <code>geometry_anchor</code> 鏄庢樉闄嶄綆 A/B/C RMSE锛岃鏄庢畫宸腑瀛樺湪鍙敱鍙楁帶缁熻鍩哄嚱鏁拌В閲婄殑缁撴瀯銆?/li>
    <li>鑻?<code>Bayesian + fine-tune</code> 缁х画鏀瑰杽鑱斿悎妯″瀷锛屼絾 <code>fine_tuned_geometry_only</code> 鍙樺樊锛岀粨璁哄簲鍐欐垚鈥滆仈鍚堟ā鍨嬫洿濂解€濓紝涓嶈兘鍐欐垚鈥滅函鍑犱綍鍙傛暟鏇村ソ鈥濄€?/li>
    <li>浠跨湡涓嚑浣曠湡鍊煎叡浜€侀潪鍑犱綍鐪熷€煎垎绌洪棿鍙樺寲锛屾槸瀵圭湡瀹炴暟鎹晠浜嬬嚎鐨勫彲鎺ч獙璇侊紱鐪熷疄鏁版嵁浠嶅彈閲囨牱鍒嗗竷銆佹祴閲忓櫔澹板拰鏈缓妯＄墿鐞嗗奖鍝嶃€?/li>
    <li>C 闆嗘槸 held-out validation锛屽苟鍙備笌绛栫暐鍒ゆ柇锛涗笉鑳界瓑鍚屼簬鏈€缁堝畬鍏ㄧ嫭绔嬫祴璇曢泦銆?/li>
  </ul>

  <h2>杈撳叆鏂囦欢</h2>
  {table([inputs], None)}
</body>
</html>"""


def summary_table(title: str, summary: dict[str, Any]) -> str:
    rows = [
        {"metric": "title", "value": title},
        {"metric": "stage1_anchor_method", "value": summary.get("stage1_anchor_method")},
        {"metric": "stage1_anchor_lambda", "value": summary.get("stage1_anchor_lambda")},
        {"metric": "fine_tune_lambda_ratio", "value": summary.get("fine_tune_lambda_ratio")},
        {"metric": "fine_tune_lambda_theta", "value": summary.get("fine_tune_lambda_theta")},
        {"metric": "selected_basis_groups", "value": ", ".join(summary.get("selected_basis_groups", []))},
        {"metric": "anchor_C_all_rmse_mm", "value": summary.get("stage1_anchor_C_all_rmse_mm")},
        {"metric": "bayesian_C_all_rmse_mm", "value": summary.get("bayesian_before_finetune_C_all_rmse_mm")},
        {"metric": "fine_tuned_C_all_rmse_mm", "value": summary.get("bayesian_finetuned_C_all_rmse_mm")},
        {"metric": "peeled_geometry_C_all_rmse_mm", "value": summary.get("peeled_geometry_C_all_rmse_mm")},
        {"metric": "peeled_geometry_improved", "value": summary.get("peeled_geometry_improved")},
    ]
    return table(rows, ["metric", "value"])


def table(rows: list[dict[str, Any]] | dict[str, Any], keys: list[str] | None) -> str:
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        return "<p><em>鏃犳暟鎹?/em></p>"
    if keys is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
    header = "".join(f"<th>{escape(str(key))}</th>" for key in keys)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{escape(fmt(row.get(key)))}</td>" for key in keys)
            + "</tr>"
        )
    return f"<table><tr>{header}</tr>{''.join(body)}</table>"


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "nan"
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(fmt(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}={fmt(val)}" for key, val in value.items())
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-search-report",
        default="data/reports/bayesian_calibration_pipeline/real_world200_to_normal50_unscaled_ratio_search/report.json",
    )
    parser.add_argument(
        "--real-final-report",
        default="data/reports/bayesian_calibration_pipeline/real_world200_to_normal50_unscaled_fixed_ratio/report.json",
    )
    parser.add_argument(
        "--synthetic-report",
        default="data/reports/bayesian_calibration_pipeline/synthetic_abc_unscaled_fixed_ratio/report.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/reports/bayesian_calibration_pipeline/integrated_real_synthetic_story",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

