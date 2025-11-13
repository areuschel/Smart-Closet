#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 09:28:35 2025

@author: areuschel
"""

import os
import pandas as pd
import numpy as np

os.chdir("/Users/areuschel/Desktop/data/")

# read data
raw = pd.read_csv("raw/20251012_raw.csv")

for col in raw:
    print(col)

"""
- make response categories ordinal categorical 
- address missing values
- fill in any 0's'
"""


upper_cols = [col for col in raw.columns if "upper layer success" in col.lower()]
lower_cols = [col for col in raw.columns if "lower layer success" in col.lower()]

def collapse_success(df, cols):
    out = df[cols].apply(lambda x: x.first_valid_index() if x.notna().any() else np.nan, axis=1)
    
    if df[cols].isin([1]).any().any():
        out = df[cols].idxmax(axis=1)
    
    out = out.str.lower()
    
    mapping = {
        "cold": -1,
        "real cold": -1,
        "little cold": -1,
        "just right": 0,
        "hot": 1,
        "little hot": 1,
        "real hot": 1
    }
    
    return out.apply(lambda x: next((v for k,v in mapping.items() if pd.notna(x) and k in x), np.nan))

raw["upper_success"] = collapse_success(raw, upper_cols)
raw["lower_success"] = collapse_success(raw, lower_cols)

raw2 = raw.drop(columns=upper_cols + lower_cols)

# check nas in response
upper_nonmissing = (raw[upper_cols].sum(axis=1) > 0).sum()
lower_nonmissing = (raw[lower_cols].sum(axis=1) > 0).sum()

print("Original non-missing upper success:", upper_nonmissing)
print("Original non-missing lower success:", lower_nonmissing)

print("New non-missing upper_success:", raw2["upper_success"].notna().sum())
print("New non-missing lower_success:", raw2["lower_success"].notna().sum())


# only up to row 54 has valid data
raw3 = raw2[0:55]

# remove non-outside runs
raw3 = raw3[raw3["Location"] != "Treadhell"]
