import json
import time
import numpy as np
from context_marl_ac.marl.marl_env import MARLEnv
from maddpg.trainer import StageConditionedMADDPGTrainer, TrainerConfig
from maddpg.evaluate_maddpg import _src_precision_recall

benchmark = [json.loads(l) for l in open('maddpg/results/benchmark_splits/eval60.jsonl', encoding='utf-8')]

print("=== NO-CEB Policy (60 questions) ===")
tcfg = TrainerConfig(state_dim=14, hidden_dim=128, device='cpu')
trainer = StageConditionedMADDPGTrainer(tcfg)
trainer.load_checkpoint('maddpg/results/maddpg_v4_noceb_run2/checkpoints/ep_0200.pt')
env = MARLEnv()

noceb_p, noceb_r = [], []
for i, q_dict in enumerate(benchmark):
    state = env.reset(q_dict, index=i+1)
    obs = np.array(env.get_global_features(), dtype=np.float32)
    _, params, discrete = trainer.select_action('retriever', obs, ['sparse_retrieve', 'hybrid_rerank', 'dense_retrieve'], explore=False)
    state, _, _, _ = env.step('retriever', discrete, params=params)
    
    obs = np.array(env.get_global_features(), dtype=np.float32)
    _, params, discrete = trainer.select_action('grader', obs, ['strict_filter', 'keep_all'], explore=False)
    
    if discrete == 'strict_filter':
        time.sleep(4.0) # Conservative sleep for parallel runs
    state, _, _, _ = env.step('grader', discrete, params=params)
    
    expected = q_dict.get('source_file', [])
    if isinstance(expected, str): expected = [expected]
    
    p, r = _src_precision_recall(state.selected_evidence, expected)
    noceb_p.append(p)
    noceb_r.append(r)

print(f'\n[FINAL] NO-CEB Mean Source Precision: {sum(noceb_p)/len(noceb_p):.4f}')
print(f'[FINAL] NO-CEB Mean Source Recall: {sum(noceb_r)/len(noceb_r):.4f}')
