Regressions
###########
In this example, a simple network is created to inverse regress the function :math:`f(x)=\sqrt{1.0+x^{2}}`, this means we want to predict :math:`x`, given a set of noisy measurements of :math:`f(x)`.

We created 3D mock data with the struture (ncases, nreas, nfeats). Where ncases is the number of targets values, nreas is the number of realizations associated to different draws of the noise, and nfeats is the number of features. In these examples two options can be tested, using as training features the sets: :math:`\left(f(x)+n\right)`, or :math:`\left(f(x)+n, n\right)`.

Point estimate inverse regression
#################################

As an
