from engine.optimizers.sdca_logistic import LogisticSDCA
from engine.optimizers.sgd_logistic import LogisticSGD
from engine.utils.normalize import Normalizer
from engine.utils.plot_learning import plot_learning
from engine.utils.data_sets import load_sklearn_dataset, load_adults_dataset
import engine.utils.projections as projections


def compare_on_sklearn():
    x, y = load_sklearn_dataset(data_set_name="lfw", n=1000)

    estimator_sgd = LogisticSGD(c=0.1, eps=1e-3)
    estimator_sdca = LogisticSDCA(c=0.1)

    plot_learning(x, y, chosen_sgd=estimator_sgd, chosen_sdca=estimator_sdca, nb_epochs=1, comp_sgd=True,
                  comp_sdca=True, is_malaptool=False)


def compare_on_adults(c=5, eps=1e-3, epochs=1):
    x, y = load_adults_dataset()

    normalizer = Normalizer(x)
    x = normalizer.normalize(x)

    # projection = projections.build_gaussian_projection(x, sampling_rate=0.01)
    projection = projections.identity_projection

    estimator_sgd = LogisticSGD(c=c, eps=eps)
    estimator_sdca = LogisticSDCA(c=c)

    plot_learning(x, y, chosen_sgd=estimator_sgd, chosen_sdca=estimator_sdca, nb_epochs=epochs, comp_sgd=True,
                  comp_sdca=True, is_malaptool=False, projection=projection)




def main():
    compare_on_adults()


if __name__ == '__main__':
    main()
