import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ.get("WORLD_SIZE", "3"))
dist.init_process_group(
    "nccl",
    rank=rank,
    world_size=world,
    init_method="tcp://192.168.1.6:29511",
)
x = torch.ones(4 * 1024 * 1024, device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
if rank == 0:
    print("allreduce ok, sum =", float(x[0].item()), flush=True)
dist.destroy_process_group()
