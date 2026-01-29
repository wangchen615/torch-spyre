import torch

print("---------- allocate tensor a -------------")
a = torch.tensor([0], dtype=torch.float16, device="spyre")

print("\n\n---------- allocate tensor b -------------")
b = torch.tensor([0.0], dtype=torch.float16, device="spyre")

print("\n\n---------- allocate tensor c -------------")
c = torch.tensor([1, 2], dtype=torch.float16, device="spyre")

print("\n\n---------- deallocate tensor b -------------")
b = b.to("cpu")

print("\n\n---------- allocate tensor d -------------")
d = torch.tensor([7, 7], dtype=torch.float16, device="spyre")

print("\n\n---------- allocate new tensor, then deallocate d -------------")
d = torch.tensor([1, 9, 8, 4], dtype=torch.float16, device="spyre")

print("\n\n---------- allocate tensor e -------------")
e = torch.tensor(
    [
        [1.0, 0.0, -1.0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1.0, 0.0, -1.0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1.0, 0.0, -1.0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1.0, 0.0, -1.0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    ],
    dtype=torch.float16,
    device="spyre",
)

print("\n\n---------- deallocate tensor a, c, d -------------")
del a
del c
del d

print("\n\n---------- allocate tensor f -------------")
k = torch.tensor([0.1, 0.2], dtype=torch.float16, device="spyre")

print("\n\n---------- allocate tensor j -------------")
j = torch.tensor([1, 2], dtype=torch.float16, device="spyre")

print("\n\n---------- allocate tensor i -------------")
tensor_i = torch.tensor([7, 0, 4, 9], dtype=torch.float16, device="spyre")

print("\n\n---------- allocate tensor f -------------")
f = torch.tensor([6, 6, 6], dtype=torch.float16, device="spyre")

print("\n\n---------- SUCCESS: All tensors allocated and managed correctly! ----------")
print("Final tensors on device:")
print(f"  e: {e.shape}, device={e.device}")
print(f"  k: {k.shape}, device={k.device}")
print(f"  j: {j.shape}, device={j.device}")
print(f"  tensor_i: {tensor_i.shape}, device={tensor_i.device}")
print(f"  f: {f.shape}, device={f.device}")
