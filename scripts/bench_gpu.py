"""ResNet34@384 학습 처리량 벤치마크 (합성 입력) — 데이터 로딩 없이 GPU 한계치 측정"""
import time, torch, torchvision
torch.backends.cudnn.benchmark = True
dev = "cuda"
m = torchvision.models.resnet34().to(dev)
opt = torch.optim.AdamW(m.parameters(), 1e-4)
scaler = torch.amp.GradScaler()
x = torch.randn(32, 3, 384, 384, device=dev); y = torch.randint(0, 2, (32,), device=dev).float()
def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x)[:, 0], y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
for _ in range(5): step()
torch.cuda.synchronize(); t = time.time(); n = 20
for _ in range(n): step()
torch.cuda.synchronize(); dt = time.time() - t
print(f"resnet34 384 bs32 AMP: {32*n/dt:.0f} img/s  ({dt/n*1000:.0f} ms/step)  arch={torch.cuda.get_device_capability()} cudnn={torch.backends.cudnn.version()}")
m = m.to(memory_format=torch.channels_last); x = x.to(memory_format=torch.channels_last)
for _ in range(5): step()
torch.cuda.synchronize(); t = time.time()
for _ in range(n): step()
torch.cuda.synchronize(); dt = time.time() - t
print(f"channels_last:            {32*n/dt:.0f} img/s")
