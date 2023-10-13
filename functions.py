import numpy as np
import matplotlib.pyplot as plt
import scipy

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
    peak_flux = 10**(0.4*(m_0-eff_peak_mag))*10**-7*10**4 # in Wm^-2
    peak_flux_err = (10**(0.4*(m_0-(eff_peak_mag+mag_err)))*10**-7*10**4 - peak_flux) # in Wm^-2
    return peak_flux, peak_flux_err

#function to calculate the comoving distance for a given redshift (approximation for z<0.1)
def get_comoving_distance_low_z(redshift, H_0):
    comoving_distance = (3*10**8*redshift/H_0) #in m
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

def chisq(model_params, model_funct, x_data, y_data, y_err):
        chisqval=[]
        for i in range(len(x_data)):
            chisqval.append(((y_data[i] - model_funct(x_data[i], *model_params))/y_err[i])**2)
            # the asterisk (*) before 'model_params' here unpacks the model parameters
        return chisqval

def automated_curve_fitting(xval, yval, yerr, model_funct, initial):
    #order arrays in ascending order
    order = np.argsort(xval)
    xval = xval[order]; yval = yval[order]; yerr = yerr[order]

    deg_freedom = xval.size - initial.size
    print('DoF = {}'.format(deg_freedom))

    popt, cov = scipy.optimize.curve_fit(model_funct, # function to fit
                                        xval, # x data
                                        yval, # y data
                                        sigma=yerr, # set yerr as the array of error bars for the fit
                                        absolute_sigma=True, # errors bars DO represent 1 std error
                                        p0=initial, # starting point for fit
                                        check_finite=True) # raise ValueError if NaN encountered (don't allow errors to pass)

    print('Optimised parameters = ', popt, '\n')
    print('Covariance matrix = \n', cov)

    plt.figure()
    plt.errorbar(xval, yval, yerr=yerr, marker='o', linestyle='None')

    # Generate best fit line using model function and best fit parameters, and add to plot. 
    plt.plot(xval, 
                model_funct(xval, *popt),  # NOTE that now we need to 'unpack' our optimised parameters with '*'.
                'k', label='optimised')

    # We can also plot a calculation based on our initial conditions to ensure that something has actually happened!
    plt.plot(xval, 
                model_funct(xval, *initial), # We need to 'unpack' our initial parameters with '*'.
                'r', label='initial')
    plt.legend()
    plt.show()

    chisq_min = np.sum(chisq(popt, model_funct, xval, yval, yerr))
    print('chi^2_min = {}'.format(chisq_min))
    chisq_reduced = chisq_min/deg_freedom
    print('reduced chi^2 = {}'.format(chisq_reduced))
    P = scipy.stats.chi2.sf(chisq_min, deg_freedom)
    print('$P(chi^2_min, DoF)$ = {}'.format(P))

    plt.figure()
    plt.errorbar(xval, yval, yerr=yerr, marker='o', linestyle='None')

    smooth_xval = np.linspace(xval[0], xval[-1], 1001)
    plt.plot(smooth_xval, model_funct(smooth_xval, *popt), color = 'black')
    plt.show()

    popt_errs = np.sqrt(np.diag(cov))
    for i in range(len(popt)):
        print('optimised parameter[{}] = {} +/- {}'.format(i, popt[i], popt_errs[i]))
    
    return popt, popt_errs