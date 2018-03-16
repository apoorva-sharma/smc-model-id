import numpy as np

def normalize(w):
    return w/np.sum(w)

def normpdf_u(y, mu, sig):
    return np.exp(-0.5* ( (y - mu).T @ np.linalg.inv(sig) @ (y - mu) ) )

class BayesianFilter:
    def __init__(self, param_vec, dyn_model, sig=0.01*np.eye(3)):
        self.params = param_vec
        self.M = len(param_vec)
        self.dyn_model = dyn_model
        self.sig = sig

        self.w = np.ones(self.M)/self.M

    def step(self, x, u, xp):
        #a = np.random.choice(self.M, self.M, p=w)
        x_pred = np.squeeze( self.dyn_model.predict([x], [u]) )
        w = [normpdf_u(xp, mu, self.sig) for mu in x_pred]
        self.w = normalize(self.w*w)

        mu_pred = np.sum(np.expand_dims(w,axis=-1)*x_pred, axis=0)
        cov_pred = np.cov(x_pred.T, aweights=w)
        return mu_pred, cov_pred


class SMCFilter:
    def __init__(self, param_vec, dyn_model, sig=0.05*np.eye(3), prop_std=0.1, N=100):
        self.M = len(param_vec)
        self.N = 100
        self.dyn_model = dyn_model
        self.sig = sig
        self.prop_std = 0.01;

        alpha_basis = np.eye(self.M)
        self.a = np.random.choice(self.M, self.N)
        self.alpha = alpha_basis[self.a,:]
        self.w = np.ones(self.N)/self.N

    @property
    def mean_alpha(self):
        return np.sum(np.expand_dims(self.w, axis=-1)*self.alpha, axis=0)

    def step(self, x, u, xp):
        x_pred_basis = np.squeeze( self.dyn_model.predict([x], [u]) )

        # resample
        a = np.random.choice(self.N, self.N, p=self.w)
        alpha_resamp = self.alpha[a,:]

        # propagate
        alpha_prop = alpha_resamp + self.prop_std*np.random.randn(self.N*self.M).reshape([self.N, self.M])

        # reweight
        x_pred = alpha_prop @ x_pred_basis

        w = [normpdf_u(xp, mu, self.sig) for mu in x_pred]
        self.w = normalize(w)
        self.alpha = alpha_prop

        mu_pred = np.mean(x_pred, axis=0)
        cov_pred = np.cov(x_pred.T, aweights=self.w)

        return mu_pred, cov_pred
