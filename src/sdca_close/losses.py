import numpy as np

LOSSES = ['square_loss', 'hinge_loss', 'absolute_loss', 'logistic_loss', 'smoothed_hinge_loss']

def build_increment(loss, lamb, n):
    if loss not in LOSSES:
        raise Exception("Loss {} is not defined".format(loss))

    if loss is "square_loss":
        f = lambda x, y, w, alpha: (y-np.dot(x.T,w)-alpha/2)/(0.5+np.linalg.norm(x)**2/(lamb*n))

    if loss is "hinge_loss":
        f = lambda x, y, w, alpha: y*max(0,min(1,(1-np.dot(x.T,w)*y)/(np.linalg.norm(x)**2/(lamb*n))+alpha*y))-alpha
    
    if loss is "absolute_loss":
        f = lambda x, y, w, alpha: max(-1,min(1,(y-np.dot(x.T,w)*y)/(np.linalg.norm(x)**2/(lamb*n))+alpha))-alpha
    
    if loss is "logistic_loss":
        f = lambda x, y, w, alpha: (y/(1+np.exp(np.dot(x.T,w)*y))-alpha)/max(1,0.25+np.linalg.norm(x)**2/(lamb*n))
    
    if loss is "smoothed_hinge_loss":
        f = lambda x, y, w, alpha: y*max(0,min(1,(1-np.dot(x.T,w)*y-gamma*alpha*y)/(np.linalg.norm(x)**2/(lamb*n)+gamma)+alpha*y))-alpha

    return f
