"""
This file reveals some insights about the SMTPP model.
author: Adrián Roselló Pedraza (RosiYo)
"""

from PIL import Image
from types import SimpleNamespace
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
import numpy as np
import torch
import cv2

from adapt.module.data import GrandStaffIterableDataset
from adapt.utils.cfg import parse_dataset_arguments
from analyze.components import extract_pca_image_from_embeddings, extract_tsne_image_from_embeddings
from data import RealDataset, SyntheticOMRDataset
from smt_model.modeling_smt import SMTModelForCausalLM
from utils.vocab_utils import check_and_retrieveVocabulary

class RealDatasetFacade(Dataset):
    """Facade class for the RealDataset"""

    db: RealDataset

    def __init__(self, data: SimpleNamespace):
        db = RealDataset(
            data_path=data.data_path,
            split="train",
            augment=True,
            tokenization_mode=data.tokenization_mode,
            reduce_ratio=data.reduce_ratio,
        )
        w2i, i2w = check_and_retrieveVocabulary(
            [db.get_gt()],
            "vocab/",
            f"{data.vocab_name}",
        )
        db.set_dictionaries(w2i, i2w)
        self.db = db

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        img, _, _ = self.db[idx]
        cv2.imwrite("test.png", img.numpy())
        img = to_pil_image(img)
        img = img.resize((1485, 1050))
        img = np.array(img)
        cv2.imwrite("test.png", img)
        img = torch.tensor(img)
        return img, None

class SMTModelFacade(SMTModelForCausalLM):
    """Facade class for the SMTModelForCausalLM"""

    def obtain_feats(self, img):
        encoder_output = self.forward_encoder(img)
        return torch.flatten(encoder_output)


def parse_args() -> SimpleNamespace:
    """Parse the arguments for the adaptation script."""
    return SimpleNamespace(
        target_dataset=parse_dataset_arguments(
            "config/Mozarteum/finetuning.json"),
        checkpoint="synthetic_mozarteum",
    )


def obtain_encoder_embeddings(
    model: SMTModelForCausalLM,
    dataset: RealDatasetFacade | SyntheticOMRDataset,
    num_samples: int = 10,
) -> np.ndarray:
    """Obtain the embeddings from the encoder of the model."""
    embeddings = []
    for i, (img, _) in enumerate(dataset):
        if i == num_samples:
            break
        embeddings.append(
            model.obtain_feats(img).detach()
        )

    return torch.stack(embeddings).numpy()


if __name__ == "__main__":
    args = parse_args()
    smt_model = SMTModelFacade.from_pretrained(
        f"antoniorv6/{args.checkpoint}")
    for param in smt_model.parameters():
        param.requires_grad = False
    print("Model loaded successfully.")

    # src_dataset = GrandStaffIterableDataset(nsamples=100)
    # feats = {"source": obtain_encoder_embeddings(smt_model, src_dataset)}
    # print("Source embeddings obtained.")

    tgt_dataset = RealDatasetFacade(args.target_dataset.data)
    feats["target"] = obtain_encoder_embeddings(smt_model, tgt_dataset)
    print("Target embeddings obtained.")

    # extract_pca_image_from_embeddings(feats, "pca.png")
    extract_tsne_image_from_embeddings(feats, "tsne.png")
