# 2024-10-22
import os
import json
import logging
import copy
import pickle
import tqdm
import IPython
from collections import defaultdict
import logging

import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import lightgbm

from maen_score import normalize, maen_score


CONFIG = json.load(open('../config.json'))
os.makedirs(CONFIG['RESULTS_DIR'], exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s -- %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def get_last_sunday(date_str):
    date = pd.to_datetime(date_str)
    if date.weekday() == 6:
        return pd.Timestamp(date)
    else:
        last_sunday = date - pd.Timedelta(days=date.weekday() + 1)
        return last_sunday


def recover_timeline(df, date_column, from_: str = None, to_: str = None):
    if from_ is None:
        from_ = df[date_column].min()
    if to_ is None:
        to_ = df[date_column].max()
    date_range = pd.date_range(from_, to_)
    df_date_range = pd.DataFrame(date_range, columns=[date_column])
    return df_date_range.merge(df, how='left')


def get_products_by_revenue(df_sales_site):
    df_sales_site['REVENUE'] = df_sales_site['SALES_QUANTITY'] * df_sales_site['PRICE']
    products = df_sales_site\
        .groupby('PRODUCT')['REVENUE']\
        .sum()\
        .sort_values(ascending=False)\
        .index\
        .to_list()
    df_sales_site.drop(columns=['REVENUE'], inplace=True)
    return products


def get_sites_by_revenue(df_sales_product):
    df_sales_product['REVENUE'] = df_sales_product['SALES_QUANTITY']\
      * df_sales_product['PRICE']
    sites = df_sales_product\
        .groupby('SITE')['REVENUE']\
        .sum()\
        .sort_values(ascending=False)\
        .index\
        .to_list()
    df_sales_product.drop(columns=['REVENUE'], inplace=True)
    return sites


def wape_score(true, pred):
  if not sum(abs(true)):
    return None
  return 100 * sum(abs(true - pred)) / sum(abs(true))


