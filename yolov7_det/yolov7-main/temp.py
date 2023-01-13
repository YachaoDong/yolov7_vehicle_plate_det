import torch

# iouv = torch.linspace(0.5, 0.95, 10)
# ious = torch.tensor([0.6,0.4,0.8])
#
# print('iouv:', iouv)
# print('ious:', ious)
#
# print('1:', (ious > iouv[0]).nonzero(as_tuple=False))
#
# for j in (ious > iouv[0]).nonzero(as_tuple=False):
#     res = ious[j] > iouv
#     print('ious[j]:', ious[j])
#     print('res:', res)
#
# # ious[j] > iouv

bbox = torch.tensor([[0.5, 0.3, 0.8], [0.1, 0.2, 0.3], [0.7, 0.8, 0.9], [0.5, 0.1, 0.6154865]])

eq_color = torch.tensor([[False, False, False], [False, True, False], [False, False, True], [False, False, True]])

print('bbox:', bbox)
print('eq_color:', eq_color)


color = eq_color.int()
res = torch.mul(bbox, color)
print('res:', res)

