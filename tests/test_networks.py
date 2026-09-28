import torch
from amaf.networks.discrete import AMAFDQN
from amaf.networks.continuous import LegacyAMAFQFusionCritic
from amaf.agents.td3 import TD3Agent


def test_amaf_dqn_shape_weights_and_centering():
    m=AMAFDQN(8,4,n_heads=3); x=torch.randn(5,8); q,d=m(x,return_diagnostics=True)
    assert q.shape==(5,4); assert d.weights.shape==(5,3)
    assert torch.allclose(d.weights.sum(-1),torch.ones(5),atol=1e-6)
    assert torch.allclose((q - m.value(m.shared(x))).mean(1), torch.zeros(5), atol=1e-5)


def test_uniform_gate():
    m=AMAFDQN(4,2,n_heads=4,gate_mode="uniform"); _,d=m(torch.randn(3,4),True)
    assert torch.allclose(d.weights,torch.full((3,4),0.25))


def test_legacy_qfusion_shape():
    m=LegacyAMAFQFusionCritic(17,6,n_heads=4); q,d=m(torch.randn(7,17),torch.randn(7,6),return_diagnostics=True)
    assert q.shape==(7,1); assert d.heads.shape==(7,4,1)


def test_paper_amaf_twin_critics_share_gate():
    cfg={"critic_type":"paper_amaf","n_heads":4,"replay":{"capacity":100,"batch_size":8}}
    a=TD3Agent(17,6,1.0,cfg,torch.device("cpu"))
    assert a.critic_1.shared_gate is a.critic_2.shared_gate
    assert a.critic_1_target.shared_gate is a.critic_2_target.shared_gate
    assert a.critic_1.shared_gate is not a.critic_1_target.shared_gate
