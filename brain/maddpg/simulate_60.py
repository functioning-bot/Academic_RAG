import json
import time
import numpy as np
from context_marl_ac.marl.marl_env import MARLEnv
from maddpg.trainer import StageConditionedMADDPGTrainer, TrainerConfig
from maddpg.evaluate_maddpg import _src_precision_recall
from maddpg.context_engineering_block import CEB_STATE_DIM, build_ceb_features

def run_simulation():
    benchmark = [json.loads(l) for l in open('maddpg/results/benchmark_splits/eval60.jsonl', encoding='utf-8')]
    
    # 1. CEB Policy
    print("=== CEB Policy (60 questions) ===")
    tcfg_ceb = TrainerConfig(state_dim=CEB_STATE_DIM, hidden_dim=128, device='cpu')
    trainer_ceb = StageConditionedMADDPGTrainer(tcfg_ceb)
    trainer_ceb.load_checkpoint('maddpg/results/maddpg_v4/checkpoints/ep_0200.pt')
    env_ceb = MARLEnv()
    
    ceb_p, ceb_r = [], []
    for i, q_dict in enumerate(benchmark):
        state = env_ceb.reset(q_dict, index=i+1)
        obs = np.array(build_ceb_features(state), dtype=np.float32)
        _, params, discrete = trainer_ceb.select_action('retriever', obs, ['sparse_retrieve', 'hybrid_rerank', 'dense_retrieve'], explore=False)
        state, _, _, _ = env_ceb.step('retriever', discrete, params=params)
        
        obs = np.array(build_ceb_features(state), dtype=np.float32)
        _, params, discrete = trainer_ceb.select_action('grader', obs, ['strict_filter', 'keep_all'], explore=False)
        
        if discrete == 'strict_filter':
            time.sleep(2.5) # Sleep to avoid rate limiting
        state, _, _, _ = env_ceb.step('grader', discrete, params=params)
        
        expected = q_dict.get('source_file', [])
        if isinstance(expected, str): expected = [expected]
        
        p, r = _src_precision_recall(state.selected_evidence, expected)
        ceb_p.append(p)
        ceb_r.append(r)
        
    print(f'-> CEB Source Precision: {sum(ceb_p)/len(ceb_p):.4f}')
    print(f'-> CEB Source Recall: {sum(ceb_r)/len(ceb_r):.4f}')
    
    # 2. NO-CEB Policy
    print("\n=== NO-CEB Policy (60 questions) ===")
    tcfg_noceb = TrainerConfig(state_dim=14, hidden_dim=128, device='cpu')
    trainer_noceb = StageConditionedMADDPGTrainer(tcfg_noceb)
    trainer_noceb.load_checkpoint('maddpg/results/maddpg_v4_noceb_run2/checkpoints/ep_0200.pt')
    env_noceb = MARLEnv()
    
    noceb_p, noceb_r = [], []
    for i, q_dict in enumerate(benchmark):
        state = env_noceb.reset(q_dict, index=i+1)
        obs = np.array(env_noceb.get_global_features(), dtype=np.float32)
        _, params, discrete = trainer_noceb.select_action('retriever', obs, ['sparse_retrieve', 'hybrid_rerank', 'dense_retrieve'], explore=False)
        state, _, _, _ = env_noceb.step('retriever', discrete, params=params)
        
        obs = np.array(env_noceb.get_global_features(), dtype=np.float32)
        _, params, discrete = trainer_noceb.select_action('grader', obs, ['strict_filter', 'keep_all'], explore=False)
        
        if discrete == 'strict_filter':
            time.sleep(2.5) # Sleep to avoid rate limiting
        state, _, _, _ = env_noceb.step('grader', discrete, params=params)
        
        expected = q_dict.get('source_file', [])
        if isinstance(expected, str): expected = [expected]
        
        p, r = _src_precision_recall(state.selected_evidence, expected)
        noceb_p.append(p)
        noceb_r.append(r)
        
    print(f'-> NO-CEB Source Precision: {sum(noceb_p)/len(noceb_p):.4f}')
    print(f'-> NO-CEB Source Recall: {sum(noceb_r)/len(noceb_r):.4f}')

if __name__ == "__main__":
    run_simulation()
