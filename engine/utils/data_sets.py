from sklearn.datasets import fetch_covtype, fetch_rcv1
from sklearn.preprocessing import normalize


def real_data_set(data_set_name="covtype", n=1000, d=10):
    if data_set_name == "covtype":
        covtype = fetch_covtype()
        X = normalize(covtype.data[:n])
        y = covtype.target[:n]
        return X, y

    if data_set_name == "rcv1":
        rcv1 = fetch_rcv1()
        X = normalize(rcv1.data[:n])
        y = rcv1.target[:n]
        return X, y

