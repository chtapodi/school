#!/usr/bin/env python
# coding: utf-8
import csv
import matplotlib.pyplot as plt
import math
import copy
import random
import statistics as stat
import ast

R1=0.6146523718
X1=1.106
Xm=26.3
R2=0.332
X2=0.464
V0=265.6
Nsync=1800

def torque(Nm) :
	top=(3*Vth()**2)((R2)/S(Nm))
	bottom=Wsync()*((Rth()+R2/S(Nm))**2+(Xth+X2)**2)
	return top/bottom


def Vth() :
	divider=math.sqrt(R1**2+(X1+Xm)**2)
	return (V0*Xm)/divider


def S(Nm) :
	return (Nsync-Nm)/Nsync

def Wsync() :
	return 188.5

for i in range(1800) :
	print(torque(i))
