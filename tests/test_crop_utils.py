import numpy as np
from amaf.envs.crop import state_to_text, decode_action, economic_reward

def test_crop_text_has_25_numeric_tokens():
    s=np.arange(25,dtype=np.float32); txt=state_to_text(s); assert len(txt.split())==25

def test_action_mapping():
    a=decode_action(24); assert a=={"anfer":160,"amir":24}

def test_terminal_profit():
    s=np.zeros(25,dtype=np.float32); s[4]=10000
    assert abs(economic_reward(s,0,0,True)-1580.0)<1e-6
