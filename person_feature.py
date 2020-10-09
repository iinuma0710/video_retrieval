import sys
sys.path.append("./fast-reid")

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor, default_argument_parser, default_setup
from fastreid.utils.checkpoint import Checkpointer

import os
import cv2
import torch
import numpy as np

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # args.config_file = "./configs/person_reid.yml"
    args.config_file = "./fast-reid/configs/person_reid.yml"
    cfg = setup(args)
    pred = DefaultPredictor(cfg)
    inputs = cv2.imread("input.png")
    inputs = torch.from_numpy(inputs.astype(np.float32)).clone().unsqueeze(0).permute(0, 3, 1, 2)
    print(inputs.size())
    outputs = pred(inputs)
    print(outputs.shape)