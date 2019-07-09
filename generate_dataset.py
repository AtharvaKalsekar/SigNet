import os
from natsort import natsorted, ns
from sklearn.model_selection import train_test_split
import random
import pandas as pd
import numpy as np

PATH_ORG = "drive/My Drive/Colab Notebooks/SigNet/signatures/signatures/full_org"
PATH_FORG = "drive/My Drive/Colab Notebooks/SigNet/signatures/signatures/full_forg"

def get_image_list():
  org_signs = os.listdir("drive/My Drive/Colab Notebooks/SigNet/signatures/signatures/full_org")
  forg_signs = os.listdir("drive/My Drive/Colab Notebooks/SigNet/signatures/signatures/full_forg")
  org_signs = [s for s in org_signs if s.endswith(".png")]
  forg_signs = [s for s in forg_signs if s.endswith(".png")]
  org_signs = natsorted(org_signs, alg=ns.IGNORECASE)
  forg_signs = natsorted(forg_signs, alg=ns.IGNORECASE)
  return org_signs, forg_signs

# data cleaning
def check_lists(org_signs,forg_signs):
  flag = False
  for i in range(len(org_signs)):
    org_ext = org_signs[i][8:]
    forg_ext = forg_signs[i][9:]
    if org_ext != forg_ext:
      flag = True
      #print(i,org_ext,forg_ext)
  '''if(flag):
    #print("Mismatches found")
  else:
    #print("No mismatch found")
'''
def refine_lists(org_signs,forg_signs):
  refined_org_signs = []
  for i in range(len(org_signs)):
    if "_41_" in org_signs[i]:
      continue
    refined_org_signs.append(org_signs[i])
  
  refined_forg_signs = []
  for i in range(len(forg_signs)):
    if "_41_" in forg_signs[i]:
      continue
    refined_forg_signs.append(forg_signs[i])
  return refined_org_signs, refined_forg_signs

def get_clean_lists():
  org_signs, forg_signs = get_image_list()
  check_lists(org_signs,forg_signs)
  org_signs, forg_signs = refine_lists(org_signs,forg_signs)
  #print(len(org_signs),len(forg_signs))
  check_lists(org_signs,forg_signs)
  return org_signs, forg_signs

def get_dataframe(org_signs,forg_signs):
  no_of_ppl = len(org_signs)//24

  raw_data = {"image_1":[], "image_2":[], "label":[]}
  for i in range(no_of_ppl):
    i1_batch_1 = []
    i1_batch_2 = []
    i2_batch = []

    start = i*24
    end = (i+1)*24

    for j in range(start,end): 
      i1_batch_1.append(os.path.join(PATH_ORG,org_signs[j]))
      i1_batch_2.append(os.path.join(PATH_ORG,org_signs[j]))
      raw_data["label"].append(1)#0

    temp_rot = (i1_batch_1[-12:]+i1_batch_1[:-12])
    i1_batch_1.extend(i1_batch_2)

    for elem in temp_rot:
      i2_batch.append(elem)

    for j in range(start,end): 
      i2_batch.append(os.path.join(PATH_FORG,forg_signs[j]))
      raw_data["label"].append(0)#1

    raw_data["image_1"].extend(i1_batch_1)
    raw_data["image_2"].extend(i2_batch)
  df = pd.DataFrame(raw_data, columns = ["image_1","image_2","label"])
  df=df.reindex(np.random.permutation(df.index))
  return df


def get_dataset(subset=None):
  org_signs,forg_signs = get_clean_lists()
  df = get_dataframe(org_signs,forg_signs)
  #print(df.shape)
  train_set, val_set = train_test_split(df,test_size=0.3,random_state=0)
  if(subset.lower()=="train"):
    dataset = train_set
  else:
    dataset = val_set
  return dataset