import json
import time
import numpy as np
from context_marl_ac.marl.marl_env import MARLEnv
from maddpg.trainer import StageConditionedMADDPGTrainer, TrainerConfig
from maddpg.evaluate_maddpg import _src_precision_recall
from maddpg.context_engineering_block import CEB_STATE_DIM, build_ceb_features

benchmark = [json.loads(l) for l in open('maddpg/results/benchmark_splits/eval60.jsonl', encoding='utf-8')]

print("=== CEB Policy (60 questions) ===")
tcfg = TrainerConfig(state_dim=CEB_STATE_DIM, hidden_dim=128, device='cpu')
trainer = StageConditionedMADDPGTrainer(tcfg)
trainer.load_checkpoint('maddpg/results/maddpg_v4/checkpoints/ep_0200.pt')
env = MARLEnv()

ceb_p, ceb_r = [], []
for i, q_dict in enumerate(benchmark):
    state = env.reset(q_dict, index=i+1)
    obs = np.array(build_ceb_features(state), dtype=np.float32)
    _, params, discrete = trainer.select_action('retriever', obs, ['sparse_retrieve', 'hybrid_rerank', 'dense_retrieve'], explore=False)
    state, _, _, _ = env.step('retriever', discrete, params=params)
    
    obs = np.array(build_ceb_features(state), dtype=np.float32)
    _, params, discrete = trainer.select_action('grader', obs, ['strict_filter', 'keep_all'], explore=False)
    
    if discrete == 'strict_filter':
        time.sleep(4.0) # Conservative sleep for parallel runs
    state, _, _, _ = env.step('grader', discrete, params=params)
    
    expected = q_dict.get('source_file', [])
    if isinstance(expected, str): expected = [expected]
    
    p, r = _src_precision_recall(state.selected_evidence, expected)
    ceb_p.append(p)
    ceb_r.append(r)

print(f'\n[FINAL] CEB Mean Source Precision: {sum(ceb_p)/len(ceb_p):.4f}')
print(f'[FINAL] CEB Mean Source Recall: {sum(ceb_r)/len(ceb_r):.4f}')
