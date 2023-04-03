#!/usr/bin/env python3
from ai_old.util.factory import build_model_from_exp


G, _ = build_model_from_exp('rec/16/5', 'G')

print('rec/16/5, e, fullk_weight', G.e.to_z.fullk_weight.item())
print('rec/16/5, e, down_weight', G.e.to_z.down_weight.item())
print('rec/16/5, e, fc_weight', G.e.to_z.fc_weight.item())


G, _ = build_model_from_exp('rec/17/0', 'G')

print('rec/17/0, e, fullk_weight', G.e.to_z.fullk_weight.item())
print('rec/17/0, e, down_weight', G.e.to_z.down_weight.item())
print('rec/17/0, e, fc_weight', G.e.to_z.fc_weight.item())

print('rec/17/0, g, fullk_weight', G.g.z_to_feat_map.fullk_weight.item())
print('rec/17/0, g, up_weight', G.g.z_to_feat_map.up_weight.item())
