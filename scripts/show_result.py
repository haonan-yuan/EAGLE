import sys

sys.path.append("..")

from EAGLE.utils.mp import *
import os
import pandas as pd
import shutil
from pathlib import Path
import os.path as osp
import json
from collections import Counter

fdir = f"../logs/history"


def show():
    pass
    res = []
    for r, OOD, fs in os.walk(fdir):
        for f in fs:
            if osp.splitext(f)[1] == ".json":
                if "0" not in osp.splitext(f)[0]:
                    info = json.load(open(osp.join(r, f)))
                    line = {}
                    line["dataset"] = dataset = info["dataset"]
                    if "syn" in dataset:
                        line["w/o OOD"] = info["train_auc"]
                        line["w/ OOD"] = info["test_auc"]
                    else:
                        line["w/o OOD"] = info["test_auc"]
                        line["w/ OOD"] = info["test_test_auc"]
                    res.append(line)

    df = pd.DataFrame(res)
    df = (
        df.groupby(by="dataset")
        .agg(
            mean_WOOOD=("w/o OOD", "mean"),
            std_WOOOD=("w/o OOD", "std"),
            mean_WOOD=("w/ OOD", "mean"),
            std_WOOD=("w/ OOD", "std"),
        )
        .reset_index()
    )
    df = df.applymap(lambda x: f"{x * 100:.2f}" if isinstance(x, float) else x)
    df["w/o OOD"] = df["mean_WOOOD"] + "+-" + df["std_WOOOD"]
    df["w/ OOD"] = df["mean_WOOD"] + "+-" + df["std_WOOD"]
    df.drop(columns=["mean_WOOOD", "std_WOOOD", "mean_WOOD", "std_WOOD"], inplace=True)

    dataset_order = ["collab", "yelp", "act"]
    df["dataset"] = pd.Categorical(
        df["dataset"], categories=dataset_order, ordered=True
    )

    res1 = df

    print(df.to_string(), file=open(f"{fdir}/results.txt", "w"))

    res = []
    for r, OOD, fs in os.walk(fdir):
        for f in fs:
            if osp.splitext(f)[1] == ".json":
                if "0" in osp.splitext(f)[0]:
                    info = json.load(open(osp.join(r, f)))
                    line = {}
                    line["dataset"] = dataset = info["dataset"]
                    if "syn" in dataset:
                        line["Train"] = info["train_auc"]
                        line["Test"] = info["test_auc"]
                    else:
                        line["Train"] = info["test_auc"]
                        line["Test"] = info["test_test_auc"]
                    res.append(line)

    df = pd.DataFrame(res)
    df = (
        df.groupby(by="dataset")
        .agg(
            mean_WOOOD=("Train", "mean"),
            std_WOOOD=("Train", "std"),
            mean_WOOD=("Test", "mean"),
            std_WOOD=("Test", "std"),
        )
        .reset_index()
    )
    df = df.applymap(lambda x: f"{x * 100:.2f}" if isinstance(x, float) else x)
    df["Train"] = df["mean_WOOOD"] + "+-" + df["std_WOOOD"]
    df["Test"] = df["mean_WOOD"] + "+-" + df["std_WOOD"]
    df.drop(columns=["mean_WOOOD", "std_WOOOD", "mean_WOOD", "std_WOOD"], inplace=True)

    dataset_order = ["collab_04", "collab_06", "collab_08"]
    df["dataset"] = pd.Categorical(
        df["dataset"], categories=dataset_order, ordered=True
    )

    res2 = df

    print(
        res1.to_string() + "\n\n" + res2.to_string(),
        file=open(f"{fdir}/results.txt", "w"),
    )


if __name__ == "__main__":
    show()
