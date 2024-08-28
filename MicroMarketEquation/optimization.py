import numpy as np
from scipy.optimize import minimize


def likelihood(params, S, B, N):
    alpha, delta, mu, epsilon = params
    term1 = alpha * (1 - delta) * (1 + mu / ((1-mu)*0.5*epsilon))**B
    term2 = alpha * delta * (1 + mu / ((1-mu)*0.5*epsilon))**S
    term3 = (1 - alpha) * (1 / (1 - mu))**(S + B + N)
    term4 = ((1 - mu) * (1 - epsilon))**N * ((1-mu)*0.5*epsilon)**(S + B)
    return -np.sum(np.log(term1 + term2 + term3) + np.log(term4))


def estmimate_para(S, B, N, params0 = [0.5, 0.5, 0.5, 0.5]): 
    result = minimize(likelihood, params0, args=(S, B, N), bounds=[(0.01, 0.99)]*4)
    estimated_params = {
        'alpha': result.x[0],
        'delta': result.x[1],
        'mu': result.x[2],
        'epsilon': result.x[3],
        'gamma': result.x[2] / (result.x[2] + (1 - result.x[2]) * result.x[3])}
    return estimated_params