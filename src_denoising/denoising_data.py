'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import os

basedir_bsds = "../datasets/BSDS500/data/rgb/"
bsds500_train_dir = os.path.join(basedir_bsds, "train/")
bsds500_test_dir = os.path.join(basedir_bsds, "test/")
bsds500_val_dir = os.path.join(basedir_bsds, "val/")
bsds500_val68_dir = os.path.join(basedir_bsds, "val68/")

basedir_urban = "../datasets/Urban100/rgb/"
urban_val_dir = os.path.join(basedir_urban)

basedir_set12 = "../datasets/Set12/"
set12_val_dir = os.path.join(basedir_set12)


basedir_div2k = "../datasets/DIV2K/"
div2k_train_dir = os.path.join(basedir_div2k, "DIV2K_train_HR")
div2k_val_dir = os.path.join(basedir_div2k, "DIV2K_valid_HR")

waterloo_train_dir = "../datasets/Waterloo/exploration_database_and_code/pristine_images/train/"