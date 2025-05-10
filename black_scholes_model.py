#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:12:14 2025

@author: jayakumarpriyadharshini
"""

import math
from scipy.stats import norm
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#Defining Black Scholes Model function

def black_scholes_model(S,K,T,r,vol):
    d1 = (math.log(S/K) + (r + 0.5 * vol **2)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)

    C = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    P = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return(S,vol,C,P)


# Creating Streamlit website


st.title('Predicting Call and Put Price Using Black Scholes Model') #Title
K = st.sidebar.number_input('Strike Price') # Strike Price Input
T = st.sidebar.number_input('Time to Expiration in Years') # Time value input
r = st.sidebar.number_input('Risk-Free Rate') # Risk free rate input
S_max = st.sidebar.slider('Max Underlying Price', 0, 100, 1) # Slider for maximum underlying price
S_min = st.sidebar.slider('Min Underlying Price', 0,100, 1) # Slider for minimum underlying price
vol_max = st.sidebar.slider('Max Volatility', 0.0 , 1.0, 0.01) # Slider for maximum volatility
vol_min = st.sidebar.slider('Min Volatility', 0.0 , 1.0, 0.01) # Slider for minimum volatility


#Setting conditions
if S_max == S_min:
    st.error("Max Spot Price must be greater than Min Spot Price")
elif vol_max == vol_min:
    st.error("Max Volatility must be greater than Min Volatility")
#If S_step is not equal to 0, call Black Scholes model function
else:
    S_step = (S_max - S_min)/10
    vol_step = (vol_max - vol_min)/10
    df_S = np.arange(S_min, S_max + S_step, S_step)
    df_vol = np.arange(vol_min, vol_max + vol_step, vol_step)
    model = []
    for i in df_S:
        for j in df_vol:
            model.append(black_scholes_model(i, K, T, r, j))

    model_2 = pd.DataFrame(model).round(3)
    col1, col2 = st.columns(2)
    C_dataframe = model_2.pivot(index = 0, columns = 1, values = 2)
    P_dataframe = model_2.pivot(index = 0, columns = 1, values = 3)

# Plotting heatmap on streamlit
    st.subheader("Call Option Prices Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(C_dataframe, annot=True, fmt=".2f", cmap='YlOrBr')
    plt.xlabel("Volatility")
    plt.ylabel("Spot Price")
    st.pyplot(plt.gcf())

    st.subheader("Put Option Prices Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(P_dataframe, annot=True, fmt=".2f", cmap='YlOrBr')
    plt.xlabel("Volatility")
    plt.ylabel("Spot Price")
    st.pyplot(plt.gcf())
