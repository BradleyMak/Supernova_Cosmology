import numpy as np

#function to read data from txt file
def read_data(filename):
    """
    Args:
        filename (_string_): .txt filename

    Returns:
        _array_: returns arrays containing name, redshift, effective peak magnitude, peak magnitude error data.
    """
    alldata = np.loadtxt(filename, dtype='str', comments='#')
    name = alldata[:, 0]
    redshift = np.array(alldata[:, 1], dtype = float)
    eff_peak_mag = np.array(alldata[:, 2], dtype = float)
    mag_err = np.array(alldata[:, 3], dtype = float)
    return name, redshift, eff_peak_mag, mag_err

#function to calculate peak flux from effective peak magnitudes (and the respective errors)
def get_flux(eff_peak_mag, mag_err, m_0):
    peak_flux = 10**(0.4*(m_0-eff_peak_mag))
    peak_flux_err = 10**(0.4*(m_0-(eff_peak_mag+mag_err)))-peak_flux
    return peak_flux, peak_flux_err

#function to calculate the comoving distance for a given redshift
def get_comoving_distance(redshift, H_0):
    comoving_distance = 3*10**8*redshift/H_0 #in Mpc
    return comoving_distance

#function to calculate the peak luminosity, and its respective error, from peak flux and redshift.
"""def get_peak_luminosity(comoving_distance, redshift, peak_flux, peak_flux_err):
    L_peak = peak_flux*4*np.pi*(1+redshift)**2*comoving_distance**2
    L_peak_err = (peak_flux+peak_flux_err)*4*np.pi*(1+redshift)**2*comoving_distance**2 - L_peak
    return L_peak, L_peak_err"""

#function to calculate the luminosity distance and its error from peak luminosity and flux.
def get_luminosity_distance(redshift, comoving_distance):
    d_L = (1+redshift)*comoving_distance
    return d_L
