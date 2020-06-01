# direct and modest heston calibration

direct_heston_calib.ipynb:
Based on http://gouthamanbalaraman.com/blog/quantlib-python-tutorials-with-examples.html by Goutham Balaraman.

Given some volatilities, maturities and strikes, calibrates heston model parameters using quantlib and 
scipy optimization tools.


modest_heston.ipynb:
Based on https://nbviewer.jupyter.org/url/people.math.ethz.ch/~jteichma/lecture_ml_web/heston_calibration.ipynb 
by Matteo Gambara.

It is able to generate heston models with random parameters and calculate the implied volatilities
of that day, which the heston model is generated for. This feature is used to generate training data for a neural 
network. The network learns the map parameters->implied volatilities. The last step is similar to the direct 
method. We are interested in the inverse problem: Given the volatilities, what are the parameters? We use
the optimization method differential_evolution to calibrate the parameters, where our trained neural network
is subjected to the loss function provided to differential_evolution.
