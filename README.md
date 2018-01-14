# warblePy
Bioacoustic and Electrophysiology Data Analysis Package

Dynamical Systems Lab - University of Buenos Aires (http://www.lsd.df.uba.ar/)

This package was originally conceived as partial port of warbleR (https://github.com/maRce10/warbleR) to Python. It implements the following classes and methods:

Wav - Inherits from numpy.ndarray. Represents wave objects with audio or electrophysiology data (wav files). Implements methods to read, write, plot, normalize, crop, clip, smooth, filter, apply time manipulations, calculate envelope, correlations, spectrograms, correlation-spectrograms, smooth time warping and other manipulations. 

TimeMatrix - Inherits from numpy.ndarray. Represents matrices where the horizontal dimension is time. Implements methods to read, write, plot, apply time manipulations.

Ens - Inherits from TimeMatrix. Represents ensembles of time-locked signals (audio or electrophysiological). Implements methods to stack, calculate mean, std and other statistics, normalize, sample, subset, plot and transform to a pandas.DataFrame.    

Spe - Inherits from TimeMatrix. Represents spectrograms of waves. Implements methods to plot and correlate spectrograms.  

CorSpe - Inherits from TimeMatrix. Represents a correlation-spectrogram. Implements a plot method. 

Rec - collection representing recordings of one or several channels. Implements methods to read, write, crop, plot and get useful iterators. 

Warble - collection representing all records of a dataset. Implements methods to load, subset, sample records and get ensembles. It allows the detection and manipulation of annotations, and to calculate ad-hoc statistics on those annotations.   

Syn - Implements biophysical models of birdsong production (Perl 2012) and returns synthetic songs (Wav). 

R, utils - Utility methods. 


References
----------

Perl YS, Arneodo EM, Amador A, Mindlin GB (2012) Nonlinear dynamics and the synthesis of Zebra finch song. Int J Bifurc Chaos 22:1250235.