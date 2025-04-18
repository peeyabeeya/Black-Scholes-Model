#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:12:14 2025

@author: jayakumarpriyadharshini
"""

import math
from scipy.stats import norm
#import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# S = 42 #Underlying Price
# K = 40 #Strike Price
# T = 0.5 #Time to Expiration
# r = 0.1 #Risk-Free Rate
# vol = 0.2 #Volatility(s.d.)

def black_scholes_model(S,K,T,r,vol):
    d1 = (math.log(S/K) + (r + 0.5 * vol **2)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)

    C = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    P = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return(S,vol,C,P)

    # print('The value of d1 is: ', round(d1,4))
    # print('The value of d2 is: ', round(d2,4))
    # print('The price of the call option is: ', round(C,2))
    # print('The price of the put option is: ', round(P,2))

#black_scholes_model(42,40,0.5,0.1,0.2)

model = []

def black_scholes_model_heatmap(S_min, S_max, K, T, r, vol_min, vol_max):
    S_step = (S_max - S_min)/10
    df_S = np.arange(S_min, (S_max + S_step), S_step)
    vol_step = (vol_max - vol_min)/10
    df_vol = np.arange(vol_min, (vol_max + vol_step), vol_step)
    for i in df_S:
        for j in df_vol:
            k = black_scholes_model(i,K,T,r,j)
            model.append(k)
    model_2 = pd.DataFrame(model).round(3)
    C_dataframe = model_2.pivot(index = 0, columns = 1, values = 2)
    plt.figure(0)
    sns.heatmap(C_dataframe, annot = True, fmt=".1f", cmap = 'CMRmap_r')
    plt.xlabel('Spot Price') # x-axis label with fontsize 15
    plt.ylabel('Volatility') # y-axis label with fontsize 15
    plt.title('Black Scholes Model Call Price Based on Spot Price & Volatility')
    # plt.show(0)

    P_dataframe = model_2.pivot(index = 0, columns = 1, values = 3)
    plt.figure(1)
    sns.heatmap(P_dataframe, annot = True, fmt=".1f", cmap = 'CMRmap_r')
    plt.xlabel('Spot Price') # x-axis label with fontsize 15
    plt.ylabel('Volatility') # y-axis label with fontsize 15
    plt.title('Black Scholes Model Put Price Based on Spot Price & Volatility')
