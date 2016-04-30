#! /usr/bin/env python

"""
File: MCint_class.py

Copyright (c) 2016 Taylor Patti

License: MIT

This module contains various methods for the numeric approximation
of a set of coupled differential equations for computation in a vectorized format.

"""

import numpy as np
import matplotlib.pyplot as mp

def Euler(u0, v0, duration, N):
    """Euler method for numeric approximation."""
    t = np.zeros(N+1)
    u = np.zeros(N+1)
    v = np.zeros(N+1)
    u[0] = u0
    v[0] = v0
    dt = duration / float(N)
    for k in range(N):
        t[k+1] = t[k] + dt
        u[k+1] = u[k] + dt*v[k]
        v[k+1] = v[k] - dt*u[k]
    return t, u, v

def Heun(u0, v0, duration, N):
    """Heun method for numeric approximation."""
    t = np.zeros(N+1)
    u = np.zeros(N+1)
    v = np.zeros(N+1)
    u[0] = u0
    v[0] = v0
    dt = duration / float(N)
    for k in range(N):
        t[k+1] = t[k] + dt
        u_tilde = u[k] + dt*v[k]
        v_tilde = v[k] - dt*u[k]
        u[k+1] = u[k] + (dt/2.0)*(v[k] + v_tilde)
        v[k+1] = v[k] - (dt/2.0)*(u[k] + u_tilde)
    return t, u, v

def SRK(u0, v0, duration, N):
    """Second-order Runga-Kutta method for numeric approximation."""
    t = np.zeros(N+1)
    u = np.zeros(N+1)
    v = np.zeros(N+1)
    u[0] = u0
    v[0] = v0
    dt = duration / float(N)
    for k in range(N):
        t[k+1] = t[k] + dt
        K1u = dt*v[k]
        K1v = -dt*u[k]
        u[k+1] = u[k] + dt*(v[k] + K1u/2.0)
        v[k+1] = v[k] + dt*(-u[k] + K1v/2.0)
    return t, u, v

def FRK(u0, v0, duration, N):
    """Fourth-order Runga-Kutta method for numeric approximation."""
    t = np.zeros(N+1)
    u = np.zeros(N+1)
    v = np.zeros(N+1)
    u[0] = u0
    v[0] = v0
    dt = duration / float(N)
    for k in range(N):
        t[k+1] = t[k] + dt
        K1u = dt*v[k]
        K1v = -dt*u[k]
        K2u = dt*(v[k] + K1u/2.0)
        K2v = dt*(-u[k] + K1v/2.0)
        K3u = dt*(v[k] + K2u/2.0)
        K3v = dt*(-u[k] + K2v/2.0)
        K4u = dt*(v[k] + K3u)
        K4v = dt*(-u[k] + K3v)
        u[k+1] = u[k] + (K1u + 2*K2u + 2*K3u + K4u) / 6.0
        v[k+1] = v[k] + (K1v + 2*K2v + 2*K3v + K4v) / 6.0
    return t, u, v

def plotter(function, N):
    """Plots both the approximate and exact functions with the proper graph
    annotations present."""
    t, u, v = eval(function)(1, 0, 10*np.pi, N)
    realcos, realsin = np.cos(t), -np.sin(t)
    p1 = mp.plot(t, u, label='u')
    p2 = mp.plot(t, realcos, label='Cos(x)')
    p3 = mp.plot(t, v, label='v')
    p4 = mp.plot(t, realsin, label='Sin(x)')
    mp.title(function + " Method with Step Size of " + str(N))
    mp.ylabel('Function Values')
    mp.xlabel('Dependent Variable')
    legendlist = ['u', 'cos', 'v', '-sin']
    mp.legend(legendlist)

def test_SimpleMethodsClose():
    """Ensures accuracy of the Euler and Heun methods."""
    listmethods = ['Euler', 'Heun']
    for method in listmethods:
        t, u, v = eval(method)(1, 0, 5 * np.pi, 1000000)
        apt = (np.allclose(u, np.cos(t), rtol = 1e-1) and np.allclose(v, -np.sin(t), rtol = 1e-1))
        msg = 'Trig function simple method failure.'
        assert apt, msg
    
def test_RungaKuttaClose():
    """Ensures accuracy of the two Runga-Kutta method orders."""
    listmethods = ['SRK', 'FRK']
    for method in listmethods:
        t, u, v = eval(method)(1, 0, 5 * np.pi, 1000000)
        apt = ((np.amax(np.abs(u - np.cos(t))) < 1e-3) and (np.amax(np.abs(v + np.sin(t))) < 1e-3))
        msg = 'Trig function Runga-Kutta Method failure.'
        assert apt, msg
        
plotter_png('Euler', 1000000)