#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
import numpy as np
import re
import librosa
import seaborn as sns
import os
import json
import IPython.display as ipd
import soundfile as sf
import torch
import h5py
import onnxruntime as ort
import openvino as ov
import torch.quantization.quantize_fx as quantize_fx

from glob import glob
from tqdm import tqdm
from pprint import pprint
from matplotlib import pyplot as plt
from itertools import chain
from os.path import join as pjoin
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from copy import deepcopy
from sklearn.metrics import f1_score

# from code_base.utils import parallel_librosa_load, groupby_np_array, stack_and_max_by_samples, macro_f1_similarity, N_CLASSES_2021_2022, N_CLASSES_2021, comp_metric, N_CLASSES_XC_LIGIT_SHORTEN, N_CLASSES_XC_LIGIT_EVEN_SHORTEN
# from code_base.utils.constants import SAMPLE_RATE
from code_base.utils.onnx_utils import ONNXEnsemble, convert_to_onnx
from code_base.models import WaveCNNAttenClasifier
from code_base.datasets import WaveDataset, WaveAllFileDataset
from code_base.utils.swa import avarage_weights, delete_prefix_from_chkp
from code_base.inefernce import BirdsInference
from code_base.utils import load_json, compose_submission_dataframe, groupby_np_array, stack_and_max_by_samples, write_json
from code_base.utils.metrics import score_numpy
from code_base.utils.main_utils import get_device


POSTFIX = ""
EXP_NAME = "eca_nfnet_l0_Exp_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005"
TRAIN_PERIOD = 5
print("Possible checkpoints:\n\n{}".format("\n".join(set([os.path.basename(el) for el in glob(f"../logdirs/{EXP_NAME}/*/checkpoints/*.ckpt*") if "train" not in os.path.basename(el)]))))

conf_path = glob(f"../logdirs/{EXP_NAME}/code/*train_configs*.py")
assert len(conf_path) == 1, len(conf_path)
conf_path = conf_path[0]

CONFIG = {
    # Inference Class
    "use_sigmoid": False,
    # Data config
    "train_df_path": "../data/train_classifier_xcm.csv",
    "split_path": "../data/cv_split_xcm_group_recordists.npy",
    "n_folds":5,
    "train_data_root":"/workspace/birdsongs",
    "test_data_root":"/workspace/birdsongs",
    "label_map_data_path": "../data/bird2int_xcm.json",
    "scored_birds_path": None, 
    "lookback": None,
    "lookahead": None,
    "segment_len": 5,
    "step": None,
    "late_normalize": True,
    "add_dataset_config": None,
    # Model config
    "exp_name": EXP_NAME,
    "model_class": WaveCNNAttenClasifier,
    "model_config": dict(
            backbone="eca_nfnet_l0",
            mel_spec_paramms={
                "sample_rate": 32000,
                "n_mels": 128,
                "f_min": 20,
                "n_fft": 2048,
                "hop_length": 512,
                "normalized": True,
            },
            head_config={
                "p": 0.5,
                "num_class": 412,
                "train_period": TRAIN_PERIOD,
                "infer_period": TRAIN_PERIOD,
            },
            exportable=True,
            fixed_amplitude_to_db=True,
        ), 
    "chkp_name": "last.ckpt",
    "swa_checkpoint_regex": r'(?P<key>\w+)=(?P<value>[\d.]+)(?=\.ckpt|$)',
    "swa_sort_rule": lambda x: -float(x["valid_roc_auc"]),
    "delete_prefix": "model.",
    "n_swa_models": 1,
    "model_output_key": "clipwise_pred_long",
}

if CONFIG.get("use_sed_mode", False):
    assert CONFIG["step"] is not None
else:
    assert CONFIG["step"] is None
    
if "folds" not in CONFIG:
    CONFIG["folds"] = list(range(CONFIG["n_folds"]))
    

bird2id = load_json(CONFIG["label_map_data_path"])

df = pd.read_csv(CONFIG["train_df_path"])
split = np.load(CONFIG["split_path"], allow_pickle=True)
val_df = [df.iloc[split[i][1]].reset_index(drop=True) for i in CONFIG["folds"]]

val_ds_conig = {
    "root": CONFIG["train_data_root"],
    "label_str2int_mapping_path": CONFIG["label_map_data_path"],
    "use_audio_cache": True,
    "test_mode": True,
    "n_cores": 64,
    "verbose": False,
    "segment_len": CONFIG["segment_len"],
    "lookback":CONFIG["lookback"],
    "lookahead":CONFIG["lookahead"],
    "sample_id": None,
    "late_normalize": CONFIG["late_normalize"],
    "step": CONFIG["step"],
    "duration_col": "duration_s"
    # "validate_sr": 32_000,
}
if CONFIG.get("add_dataset_config") is not None:
    val_ds_conig.update(CONFIG["add_dataset_config"])
loader_config = {
    "batch_size": 512,
    "drop_last": False,
    "shuffle": False,
    "num_workers": 32,
}

ds_val = [WaveAllFileDataset(df=df, **val_ds_conig) for df in val_df]

loader_val = [torch.utils.data.DataLoader(
    ds,
    **loader_config,
)for ds in ds_val]


def create_model_and_upload_chkp(
    model_class,
    model_config,
    model_device,
    model_chkp_root,
    model_chkp_basename=None,
    model_chkp_regex=None,
    delete_prefix=None,
    swa_sort_rule=None,
    n_swa_to_take=3,
    prune_checkpoint_func=None
):
    if model_chkp_basename is None:
        basenames = os.listdir(model_chkp_root)
        checkpoints = []
        for el in basenames:
            matches = re.findall(model_chkp_regex, el)
            if not matches:
                continue
            parsed_dict = {key: value for key, value in matches}
            parsed_dict["name"] = el
            checkpoints.append(parsed_dict)
        print("SWA checkpoints")
        pprint(checkpoints)
        checkpoints = sorted(checkpoints, key=swa_sort_rule)
        checkpoints = checkpoints[:n_swa_to_take]
        print("SWA sorted checkpoints")
        pprint(checkpoints)
        if len(checkpoints) > 1:
            checkpoints = [
                torch.load(os.path.join(model_chkp_root, el["name"]), map_location="cpu")["state_dict"] for el in checkpoints
            ]
            t_chkp = avarage_weights(
                nn_weights=checkpoints,
                delete_prefix=delete_prefix
            )
        else:
            chkp_path = os.path.join(model_chkp_root, checkpoints[0]["name"])
            print("vanilla model")
            print("Loading", chkp_path)
            t_chkp = torch.load(
                chkp_path, 
                map_location="cpu"
            )["state_dict"]
            if delete_prefix is not None:
                t_chkp = delete_prefix_from_chkp(t_chkp, delete_prefix)
    else:
        chkp_path = os.path.join(model_chkp_root, model_chkp_basename)
        print("vanilla model")
        print("Loading", chkp_path)
        t_chkp = torch.load(
            chkp_path, 
            map_location="cpu"
        )["state_dict"]
        if delete_prefix is not None:
            t_chkp = delete_prefix_from_chkp(t_chkp, delete_prefix)

    if prune_checkpoint_func is not None:
        t_chkp = prune_checkpoint_func(t_chkp)
    t_model = model_class(**model_config, device=model_device) 
    print("Missing keys: ", set(t_model.state_dict().keys()) - set(t_chkp))
    print("Extra keys: ",  set(t_chkp) - set(t_model.state_dict().keys()))
    t_model.load_state_dict(t_chkp, strict=False)
    t_model.eval()
    return t_model

model = [create_model_and_upload_chkp(
    model_class=CONFIG["model_class"],
    model_config=CONFIG['model_config'],
    model_device=get_device(),
    model_chkp_root=f"../logdirs/{CONFIG['exp_name']}/fold_{m_i}/checkpoints",
    # model_chkp_root=f"../logdirs/{CONFIG['exp_name']}/checkpoints",
    model_chkp_basename=CONFIG["chkp_name"] if CONFIG["swa_checkpoint_regex"] is None else None,
    model_chkp_regex=CONFIG.get("swa_checkpoint_regex"),
    swa_sort_rule=CONFIG.get("swa_sort_rule"),
    n_swa_to_take=CONFIG.get("n_swa_models", 3),
    delete_prefix=CONFIG.get("delete_prefix"),
    prune_checkpoint_func=CONFIG.get("prune_checkpoint_func")
) for m_i in range(CONFIG["n_folds"])]


inference_class = BirdsInference(
    device="cuda",
    verbose_tqdm=True,
    use_sigmoid=CONFIG["use_sigmoid"],
    model_output_key=CONFIG["model_output_key"],
)

bird2id = load_json(CONFIG["label_map_data_path"])

all_predicted_folds_df = []
for fold_model, fold_loader in tqdm(zip(model, loader_val), total=len(loader_val), desc='Inferencing folds'):
    fold_filenames = fold_loader.dataset.df[fold_loader.dataset.name_col].copy()
    assert len(set(fold_filenames)) == len(fold_filenames)
    fold_preds, fold_dfidx, fold_end = inference_class.predict_test_loader(
        nn_models=[fold_model],
        data_loader=fold_loader
    )
    all_predicted_folds_df.append(compose_submission_dataframe(
        probs=fold_preds,
        dfidxs=fold_dfidx,
        end_seconds=fold_end,
        filenames=fold_filenames,
        bird2id=bird2id
    ))

for i in range(len(all_predicted_folds_df)):
    all_predicted_folds_df[i]["fold"] = i


all_predicted_folds_df = pd.concat(all_predicted_folds_df).reset_index(drop=True)

plt.title("Most 'Probable' class probability distribution")
plt.hist(all_predicted_folds_df.iloc[:,1:-1].values.max(axis=1), bins=30)
plt.savefig("../data/predictions/probability_distribution.png")

print(
    "Max Prob: ", all_predicted_folds_df.iloc[:,1:-1].values.max(), 
    "Min Prob: ", all_predicted_folds_df.iloc[:,1:-1].values.min(),
    "Median Prob: ", np.median(all_predicted_folds_df.iloc[:,1:-1].values)
)


SAVE_PATH = os.path.join(
    "../data/predictions",
    EXP_NAME + POSTFIX + ".csv"
)
print(SAVE_PATH)
all_predicted_folds_df.to_csv(SAVE_PATH, index=False)

