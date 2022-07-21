""" main.py """
import numpy as np
from configs.config import CFG
from src.model.unet import Unet


def run_unet():
    """loads data, Builds model, trains and evaluates"""
    FOLD = [0, 1, 2, 3, 4]
    for i in FOLD:
        model = Unet(CFG, i)
        model.load_data()
        model.build()
        model.train()
        # model.evaluate()
        model.test_model()


if __name__ == '__main__':
    # run()
    run_unet()
    np.loadtxt()
