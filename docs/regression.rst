Regressions
###########
In this example, a simple network is created to inverse regress the function :math:`f(x)=\sqrt{1.0+x^{2}}`, this means we want to predict :math:`x`, given a set of noisy measurements of :math:`f(x)`.

We created 3D mock data with the structure (ncases, nreas, nfeats). Where ncases is the number of targets values, nreas is the number of realizations associated to different draws of the noise, and nfeats is the number of features. In these examples two options can be tested, using as training features the sets: :math:`\left(f(x)+n\right)`, or :math:`\left(f(x)+n, n\right)`.

Point estimate inverse regression
#################################

After running you will

.. image:: MLTF/examples/regression/inverse/animations/out/point_noise_regression_animation_2feats_mse/validation/inverse_regression.gif
  :width: 400
  :alt: Alternative text


.. image:: ../examples/regression/inverse/animations/out/point_noise_regression_animation_2feats_mse/validation/inverse_regression.gif
  :width: 400
  :alt: Alternative text

Point estimate inverse regression with Dropout
##############################################

One way to account for NN model uncertainties is to use Dropout layers which randomly deactivate some neurons during the training and evalution time.
