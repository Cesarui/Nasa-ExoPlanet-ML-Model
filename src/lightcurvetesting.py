import lightkurve as lk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

#This is my attempt to implementing lightcurve data to train the ML algorithm but it took too much time to learn all of it


def check_Dispostion(data):
  if(data == 'CONFIRMED'):
    return 0
  elif(data == 'CANDIDATE'):
    return 1
  else:
    return 2

id = '10797460'

search_result = lk.search_targetpixelfile(f'KIC {id}', author="Kepler", quarter=4, cadence="long").download()
lc = search_result.to_lightcurve(method='pld').remove_nans()

flat_lc = lc.flatten(window_length=101, polyorder=2)

bls = flat_lc.to_periodogram(method='bls', period=np.linspace(0.5, 50, 5000))
period = bls.period_at_max_power
t0 = bls.transit_time_at_max_power
folded = flat_lc.fold(period=period, epoch_time=t0)

