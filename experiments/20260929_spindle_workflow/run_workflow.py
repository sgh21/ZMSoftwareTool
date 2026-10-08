"""Replay real packages through an empty store; never read research checkpoints."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from core.services.spindle_monitoring_service import SpindleMonitoringService  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', type=Path, default=PROJECT / 'data/processed/spindle_daily_packages')
    parser.add_argument('--output', type=Path, default=PROJECT / 'data/reports/spindle_workflow' / datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    if (args.output / 'store/state.json').exists():
        raise ValueError('全流程验证必须使用空目录，请另选 --output')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    service = SpindleMonitoringService(args.output / 'store')
    assert not service.runs and not service.models
    def progress(percent, message):
        print(f'{percent:3d}% {message}', flush=True)
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    initial = [args.packages / f'spindle_202607{day}.zip' for day in (22, 23, 24, 25)]
    service.import_packages(initial, 'initial', progress)
    service.set_labels(list(service.runs), 'healthy', '流程验证：模拟工作人员确认初始整批正常数据')
    assert len(service.runs) == 31 and len(service.training_candidates()) == 31
    assert all(run['manual_label'] == 'healthy' for run in service.runs.values())
    assert all(entry['score'] is None for entry in service.state['results'])
    first = service.train(progress, {'epochs': args.epochs})
    assert first['initialization'] == 'random' and first['reanalysis_status'] == 'complete'
    service.set_thresholds(warning=2, fault=4)  # 人工设置接口演练值，不是工程阈值。
    for day in ('20260726', '20260823'):
        service.import_packages([args.packages / f'spindle_{day}.zip'], 'daily', progress)
    assert len(service.runs) == 49 and len(service.training_candidates()) == 31
    before = deepcopy(service.state['results'])
    first_scores = [{**entry, 'condition': service.runs[entry['run_id']]['experiment_condition']}
                    for entry in before if entry['model_version'] == first['version']]
    save('initial_model_results.json', first_scores)
    healthy = [run for run in service.list_runs() if run['captured_date'] == '2026-07-26'][:3]
    for run in healthy:
        service.set_label(run['run_id'], 'healthy', '流程验证：模拟工作人员确认7月正常数据', True)
    abnormal = [run for run in service.list_runs() if run['experiment_condition'].startswith('unbalance')]
    for run in abnormal:
        service.set_label(run['run_id'], 'abnormal', '流程验证：按试验工况模拟人工异常标注', True)
    assert len(service.training_candidates()) == 34
    labels = {key: deepcopy(run['label_history']) for key, run in service.runs.items()}
    second = service.train(progress, {'epochs': args.epochs})
    assert second['version'] != first['version'] and second['initialization'] == 'random'
    assert second['reanalysis_status'] == 'complete'
    assert service.thresholds_review_required
    assert all(service.latest_result(key)['assessment']['status'] == 'review_required' for key in service.runs)
    assert service.state['results'][:len(before)] == before
    assert {key: run['label_history'] for key, run in service.runs.items()} == labels
    used = set(second['training_run_ids'] + second['calibration_run_ids'])
    assert len(used) == 34 and not used.intersection(run['run_id'] for run in abnormal)
    assert not set(second['training_run_ids']).intersection(second['calibration_run_ids'])
    for run_id in service.runs:
        result = service.latest_result(run_id)
        assert result['model_version'] == second['version'] and result['score'] is not None
    # 阈值按模型版本重新人工确认；旧判定及旧阈值仍保留。
    service.set_thresholds(warning=2, fault=4)
    for run_id in service.runs:
        service.evaluate(run_id, reason='threshold_confirmation')
    assert not service.thresholds_review_required
    # 用实际推理分数核验三种分支；仅隔离验证，不作为推荐阈值。
    example = healthy[0]['run_id']
    score = service.latest_result(example)['score']
    for warning, fault, expected in ((score*2, score*3, 'normal'),
                                      (score*.5, score*2, 'warning'),
                                      (score*.25, score*.5, 'fault')):
        service.set_thresholds(warning, fault)
        assert service.evaluate(example, reason='threshold_branch_test')['assessment']['status'] == expected
    service.set_thresholds(2, 4)
    service.evaluate(example, reason='threshold_confirmation')
    old_count = len(service.runs), len(service.state['results'])
    service.import_packages([args.packages / 'spindle_20260823.zip'])
    assert old_count == (len(service.runs), len(service.state['results']))
    restored = SpindleMonitoringService(service.root)
    assert restored.current_model['version'] == second['version']
    assert {key: run['label_history'] for key, run in restored.runs.items()} == labels
    latest = [{**restored.latest_result(run['run_id']), 'condition': run['experiment_condition']}
              for run in restored.list_runs()]
    columns = ('run_id','captured_date','model_version','role','condition','score','normalized_mse',
               'xy_rms_mm_s','temperature_c','actual_speed_rpm','current_a','assessment')
    save('updated_model_results.json', [{key: row[key] for key in columns} for row in latest])
    report = {
        'completed_at': datetime.now().isoformat(), 'elapsed_seconds': time.perf_counter()-started,
        'store': str(service.root), 'epochs_per_training': args.epochs,
        'initial_runs': 31, 'daily_runs': 18, 'total_runs': 49, 'packages': 6,
        'model_versions': [first['version'], second['version']],
        'training_candidates': [31,34], 'human_label_simulation': {'healthy':34,'abnormal':4,'unconfirmed':11},
        'initial_model': {key:first[key] for key in ('device','train_window_count','validation_window_count','healthy_reference_mse','history')},
        'updated_model': {key:second[key] for key in ('device','train_window_count','validation_window_count','healthy_reference_mse','history')},
        'checks': ['empty_start','initial_batch_confirmed_healthy','random_initialization_twice','run_separated_calibration',
                   'daily_not_auto_enrolled','abnormal_excluded','automatic_full_history_reanalysis',
                   'old_models_and_results_preserved','labels_preserved','manual_threshold_three_branches',
                   'restart_persistence','duplicate_import_no_overwrite','threshold_reconfirmation_after_new_model'],
        'thresholds': {'warning':2,'fault':4,'purpose':'workflow_simulation_only_not_engineering_limits'},
        'evaluation_count':len(service.state['results']),
        'limitations':[f'{args.epochs} epochs is workflow validation, not convergence certification',
                       'experiment labels are not measured geometric error or certified repair labels',
                       'August runs are never used for either model training or calibration'],
    }
    save('report.json',report)
    print('VALIDATED '+str(args.output),flush=True)


if __name__ == '__main__':
    main()
