Regressions
###########
In this example, a simple network is created to inverse regress the function :math:`f(x)=\sqrt(1.0+x**2)`, where 3D mock data is created adding noise to :math:`f(x)`, the structure of the data is (ncases, nreas, nfeats). Where ncases is the number of targets values, nreas is the number of realizations associated to different draws of the noise, and nfeats is the number of features. In these examples two options can be tested, using as training features only :math:`f(x)+n`, or :math:`f(x)+n` and n.

Point estimate inverse regression
#################################
