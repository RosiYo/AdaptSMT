"""
This file reveals some insights about the SMTPP model.
author: Adrián Roselló Pedraza (RosiYo)
"""

from types import SimpleNamespace

from adapt.module.data import GrandStaffIterableDataset
from adapt.utils.cfg import parse_dataset_arguments
from analyze.components import extract_pca_image_from_embeddings, extract_tsne_image_from_embeddings
from data import RealDataset, SyntheticOMRDataset
from smt_model.modeling_smt import SMTModelForCausalLM


def parse_args() -> SimpleNamespace:
    """Parse the arguments for the adaptation script."""
    return SimpleNamespace(
        target_dataset={
            "config": parse_dataset_arguments("config/Mozarteum/finetuning.json"),
            "fold": 0,
        },
        checkpoint="synthetic_mozarteum",
    )

def obtain_encoder_embeddings(
    model: SMTModelForCausalLM,
    dataset: RealDataset | SyntheticOMRDataset,
    num_batches: int = 10,
) -> list:
    """Obtain the embeddings from the encoder of the model."""
    embeddings = []
    for i, batch in enumerate(dataset):
        if i == num_batches:
            break
        embeddings.append(
            model.forward_encoder(batch["input_ids"]).detach().cpu().numpy()
        )
    return embeddings


if __name__ == "__main__":
    args = parse_args()
    smt_model = SMTModelForCausalLM.from_pretrained(f"antoniorv6/{args.checkpoint}")
    for param in smt_model.parameters():
        param.requires_grad = False

    src_dataset = GrandStaffIterableDataset(nsamples=100)
    feats = { "source": obtain_encoder_embeddings(smt_model, src_dataset) }
    del src_dataset

    tgt_dataset = RealDataset(**args.target_dataset)
    feats["target"] = obtain_encoder_embeddings(smt_model, tgt_dataset)
    del tgt_dataset

    extract_pca_image_from_embeddings(feats, "pca.png")
    extract_tsne_image_from_embeddings(feats, "tsne.png")
