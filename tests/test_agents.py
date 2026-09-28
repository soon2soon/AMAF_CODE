import numpy as np
import torch
from amaf.agents.td3 import TD3Agent
from amaf.agents.sac import SACAgent
from amaf.agents.dqn import DQNAgent


def _fill(agent, n=16, state_dim=5, action_dim=2):
    for _ in range(n):
        s=np.random.randn(state_dim).astype('float32'); a=np.tanh(np.random.randn(action_dim)).astype('float32'); ns=np.random.randn(state_dim).astype('float32')
        agent.add_transition(s,a,ns,float(np.random.randn()),0.0)


def test_td3_variants_one_train_step():
    for critic_type in ['vanilla','legacy_dueling','legacy_amaf_qfusion','paper_dueling','paper_amaf']:
        cfg={'critic_type':critic_type,'n_heads':3,'replay':{'capacity':100,'batch_size':8},'policy_freq':2}
        a=TD3Agent(5,2,1.0,cfg,torch.device('cpu')); _fill(a)
        out=a.train_step(); assert out is not None and np.isfinite(out['critic_loss'])


def test_sac_one_train_step():
    cfg={'replay':{'capacity':100,'batch_size':8}}
    a=SACAgent(5,2,1.0,cfg,torch.device('cpu')); _fill(a)
    out=a.train_step(); assert out is not None and np.isfinite(out['critic_loss'])


def test_dqn_one_train_step():
    cfg={'algorithm':'amaf_dqn','network':{'n_heads':3},'replay':{'capacity':100,'batch_size':8},'learn_start':8,'target_update_interval':4}
    a=DQNAgent(5,3,cfg,torch.device('cpu'))
    out=None
    for _ in range(12):
        s=np.random.randn(5).astype('float32'); ns=np.random.randn(5).astype('float32')
        out=a.observe(s,1,1.0,ns,False)
    assert out is not None and np.isfinite(out['critic_loss'])
