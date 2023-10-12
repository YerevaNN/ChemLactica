from optimum.bettertransformer import BetterTransformer
from typing import Union
import torch
import json
import os
import sys
import re
from rdkit.Chem.Draw import rdMolDraw2D
from transformers import AutoTokenizer, OPTForCausalLM
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.QED import qed
from rdkit.Chem import RDConfig
from PIL import Image
import cairosvg
import io
import gradio as gr
from check_prompt import check_prompt
from chem_props import ChemLacticaProperty
import argparse
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa


def main(model_path):
    print(model_path)
    model = OPTForCausalLM.from_pretrained(model_path)
    model = BetterTransformer.transform(model)
    tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m")
    valid_property_options = [str(prop) for prop in ChemLacticaProperty]
    print(valid_property_options)

    class CustomTheme(Base):
        def __init__(
            self,
            *,
            primary_hue: Union[colors.Color, str] = colors.red,
            secondary_hue: Union[colors.Color, str] = colors.cyan,
            neutral_hue: Union[colors.Color, str] = colors.gray,
            spacing_size: Union[sizes.Size, str] = sizes.spacing_md,
            radius_size: Union[sizes.Size, str] = sizes.radius_md,
            text_size: Union[sizes.Size, str] = sizes.text_lg,
            font: Union[fonts.Font, str] = (
                fonts.GoogleFont("Quicksand"),
                "ui-sans-serif",
                "sans-serif",
            ),
            # font_mono: fonts.Font
            # | str
            # | Iterable[fonts.Font | str] = (
            #     fonts.GoogleFont("IBM Plex Mono"),
            #     "ui-monospace",
            #     "monospace",
            # ),
        ):
            super().__init__(
                primary_hue=primary_hue,
                secondary_hue=secondary_hue,
                neutral_hue=neutral_hue,
                spacing_size=spacing_size,
                radius_size=radius_size,
                text_size=text_size,
                font=font,
            )

    def generate_molecule_image(mol):
        drawer = rdMolDraw2D.MolDraw2DSVG(800, 800)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        image = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg)))
        return image

    def smiles_to_mol(smiles_string):
        mol = Chem.MolFromSmiles(smiles_string)
        return mol

    def calculate_properties(mol):
        weight = Descriptors.MolWt(mol)
        clogp = Descriptors.MolLogP(mol)
        qed_score = qed(mol)
        sas_score = sascorer.calculateScore(mol)
        prop_json = json.dumps(
            {"Weight": weight, "CLogP": clogp, "QED": qed_score, "SAS": sas_score},
            indent=4,
        )

        return prop_json

    def calculate_perplexity(text):
        inputs = tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            return perplexity.item()

    def generate_text(
        input_text, greedy, temperature, top_k, top_p, hide_special_tokens
    ):
        try:
            check_prompt(input_text, valid_property_options)
        except Exception as e:
            gr.Warning(str(e))
            pass
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            output = model.generate(
                inputs,
                do_sample=not greedy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_length=200,
            )
        generated_text = tokenizer.decode(  # noqa: E203
            output[:, inputs.shape[-1] :][0],  # noqa: E203
            skip_special_tokens=hide_special_tokens,  # noqa: E203
        )  # noqa: E203
        full_text = input_text + "" + generated_text

        # Extract the SMILES string
        image = None
        properties = None

        try:
            smiles_string = re.search(
                r"\[START_SMILES\](.*?)\[END_SMILES\]", full_text
            ).group(1)
            mol = smiles_to_mol(smiles_string)
            image = generate_molecule_image(mol)
            properties = calculate_properties(mol)
        except Exception as e:
            print(e)
        perplexity = calculate_perplexity(generated_text)

        return image, generated_text, properties, perplexity

    theme = CustomTheme()
    theme.set(
        slider_color="*secondary_300",
        slider_color_dark="*secondary_600",
    )

    iface = gr.Interface(
        fn=generate_text,
        theme=theme,
        interpretation="default",
        examples=[["""[START_SMILES] """]],
        title="ChemLactica",
        analytics_enabled=False,
        inputs=[
            gr.inputs.Textbox(lines=2, label="Input Text"),
            gr.inputs.Checkbox(label="Greedy Decoding"),
            gr.inputs.Slider(
                minimum=0.0, maximum=1.0, step=0.01, default=1.0, label="Temperature"
            ),
            gr.inputs.Slider(minimum=0, maximum=100, step=1, default=50, label="Top-k"),
            gr.inputs.Slider(
                minimum=0.0, maximum=1.0, step=0.01, default=1.0, label="Top-p"
            ),
            gr.inputs.Checkbox(label="Hide Special Tokens"),
        ],
        outputs=[
            gr.Image(label="Molecule Image", type="pil", height=350, width=600),
            gr.outputs.Textbox(label="Generated Text"),
            gr.outputs.JSON(label="Molecular Properties"),
            gr.outputs.Textbox(label="Perplexity"),
        ],
    )
    iface.launch(share=True)


# iface.queue(max_size=20, concurrency_count=5)
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Put in the model path")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/tigranfahradyan/BIG-benchWr/checkpoint-190464",
        required=False,
    )
    args = parser.parse_args()
    main(args.model_path)
