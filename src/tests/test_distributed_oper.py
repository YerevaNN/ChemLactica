import torch.distributed as dist
from accelerate import Accelerator

# dist.init_process_group()
# print("Process", dist.get_rank())
# dist.barrier()

def main():
    accelerator = Accelerator()
    print("Process:", accelerator.process_index)
    accelerator.wait_for_everyone()
    print("Reached here:", accelerator.process_index)

if __name__ == "__main__":
    main()