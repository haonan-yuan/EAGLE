import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR, "../../data")
processed_datafile = f"{dataroot}/collab_06"

dataset = "collab_06"
testlength = 5
vallength = 1
