from transformers import AutoTokenizer
import torch
from accelerate import Accelerator

# sys.path.insert(0, "/home/tigranfahradyan/ChemLactica/ChemLactica/src")
from utils import load_model, chemlactica_special_tokens


if __name__ == "__main__":
    checkpoint_dir = "/home/tigranfahradyan/ChemLactica/checkpoints/facebook/galactica-125m/none/checkpoint-5"  # noqa
    accelerator = Accelerator()
    saved_model = load_model(
        "facebook/galactica-125m", flash_att=True, dtype=torch.bfloat16
    )
    saved_model.resize_token_embeddings(50000 + len(chemlactica_special_tokens))
    saved_model = accelerator.prepare(saved_model)
    accelerator.load_state(checkpoint_dir)
    saved_model.to("cuda:0")
    saved_model.eval()
    print(saved_model)

    saved_model = load_model(checkpoint_dir, flash_att=True, dtype=torch.bfloat16)
    saved_model.to("cuda:0")
    # saved_model.eval()

    contexts = [
        "[CLOGP 100][START_SMILES]",
        "[SAS 1][START_SMILES]",
        "[WEIGHT 41.123][START_SMILES]",
        "random input",
    ]
    tokenizer = AutoTokenizer.from_pretrained("chemlactica/tokenizer/galactica-125m")

    with torch.no_grad():
        for cont in contexts:
            max_length = 400
            inputs = tokenizer(cont, return_tensors="pt").to(saved_model.device)
            generated_toks = saved_model.generate(
                inputs["input_ids"], max_length=max_length, do_sample=False
            )
            print(generated_toks)
