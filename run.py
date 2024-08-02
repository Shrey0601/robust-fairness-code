import data
import optimization
import losses
import model
import utils
import softweights_training
import tensorflow as tf
import numpy as np

df = data.load_dataset_adult()
# df = data.load_dataset_credit()  # Uncomment this line to use the credit dataset

LABEL_COLUMN = "label" #(for Adult); "default" (for Credit)
FEATURE_NAMES = list(df.keys())
FEATURE_NAMES.remove(LABEL_COLUMN)
PROTECTED_COLUMNS = ['race_White', 'race_Black', 'race_Other_combined'] #(for Adult); ['EDUCATION_grad', 'EDUCATION_uni', 'EDUCATION_hs_other'] (for Credit)

noise_parameter = 0.1
PROXY_COLUMNS = data.get_proxy_column_names(PROTECTED_COLUMNS, noise_parameter)

results = softweights_training.get_results_for_learning_rates(df, FEATURE_NAMES, PROTECTED_COLUMNS, PROXY_COLUMNS, LABEL_COLUMN,constraint='tpr')