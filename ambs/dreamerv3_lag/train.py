from ambs.common.helpers import main as train
from ambs.dreamerv3_lag.agent import Agent


def main(argv=None):
    train(Agent, argv)


if __name__ == '__main__':
    main()
