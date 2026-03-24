import logging
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
import argparse
import os

from preprocess import *

DATASETS = (
    'MATR',
    'HUST',
    'CALCE',
    'RWTH',
    'HNEI',
    'SNL',
    'UL_PUR',
    'CALB',
    'ISU_ILCC',
    'MICH',
    'MICH_EXP',
    'NAion',
    'SDU',
    'Stanford',
    'Tongji',
    'XJTU',
    'ZNion'
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, 'all'],
        default='all',
        help="Dataset to preprocess"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="main")
    config = OmegaConf.create(config)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    datasets_to_preprocess = DATASETS if args.dataset == 'all' else [args.dataset]
    for dataset in datasets_to_preprocess:
        raw_dir = os.path.join(config.path.raw_dir, dataset)
        preprocessed_dir = os.path.join(config.path.preprocessed_dir, dataset)
        os.makedirs(preprocessed_dir, exist_ok=True)

        PREPROCESSOR_MAP = {
            'CALCE': 'CALCEPreprocessor',
            'HUST': 'HUSTPreprocessor',
            'MATR': 'MATRPreprocessor',
            'RWTH': 'RWTHPreprocessor',
            'HNEI': 'HNEIPreprocessor',
            'SNL': 'SNLPreprocessor',
            'UL-PUR': 'UL_PURPreprocessor',
            'CALB': 'CALBPreprocessor',
            'ISU-ILCC': 'ISU_ILCCPreprocessor',
            'MICH': 'MICHPreprocessor',
            'MICH_EXP': 'MICH_EXPPreprocessor',
            'SDU': 'SDUPreprocessor',
            'Stanford': 'StanfordPreprocessor',
            'Tongji': 'TongjiPreprocessor',
            'XJTU': 'XJTUPreprocessor',
            'NAion': 'NAionPreprocessor',
            'ZNion': 'ZNionPreprocessor',
        }

        preprocessor_class = globals()[PREPROCESSOR_MAP[dataset]]
        # preprocessor = preprocessor_class(preprocessed_dir, silent=True)
        preprocessor = preprocessor_class(preprocessed_dir)

        processed, skipped = preprocessor.process(raw_dir)
        logging.info(f'Preprocessed {processed} batteries from {dataset}, skipped {skipped} batteries.')


