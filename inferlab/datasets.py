import numpy as np

from collections import namedtuple


def load_emergencyroom():
    """
    Inter-arrival times of patients at an emergency room.

    Returns
    -------
    np.ndarray
        Array of 10 observed inter-arrival times in minutes.
    """

    return np.array([11, 36, 50, 6, 3, 71, 41, 42, 51, 18])

def load_rutherford(format: str = "table"):
    """
    Alpha particle decay counts from the Rutherford-Geiger experiment.

    Polonium radiation was measured with a Geiger counter over 2608
    intervals of 7.5 seconds each. The number of decays per interval
    follows a Poisson distribution.

    Parameters
    ----------
    format : str, optional
        'table' returns a dictionary mapping decay counts to observed
        frequencies. 'raw' returns the full sample as a flat array
        with one entry per interval. Default is 'table'.

    Returns
    -------
    dict or np.ndarray
        Frequency table as dict, or raw sample as np.ndarray.

    Raises
    ------
    ValueError
        If format is not 'table' or 'raw'.
    """

    table = {0: 57, 1: 203, 2: 383, 3: 525, 4: 532, 5: 408, 6: 273, 7: 139, 8: 45, 9: 27, 10: 10, 11: 6}

    raw = np.repeat(np.array(list(table.keys())), np.array(list(table.values())))

    if(format == "table"):
        return table
    elif(format == "raw"):
        return raw
    else:
        raise ValueError("Format must be table or raw!")


def load_iq():
    """
    Normal distribution model for IQ scores in the general population.

    IQ tests are standardized such that scores follow a normal
    distribution. No sample data is available - this function returns
    the known population parameters.

    Returns
    -------
    NormalModel
        Named tuple with fields mu (mean) and sigma (standard deviation).
    """

    NormalModel = namedtuple("NormalModel", ["mu","sigma"])
    iq = NormalModel(mu=100, sigma=15) # Zugriff mit iq.mu und iq.sigma statt iq[0] und iq[1]
    return iq

def load_reactiontime():
    """
    Summary statistics for reaction times during a driving simulation.

    Two groups of participants were compared: one group conducted a
    phone conversation while driving, the other listened to the radio.
    Reaction times were measured in milliseconds. No raw data is
    available - only group-level summary statistics are provided.

    Returns
    -------
    tuple of ReactionTimeModel
        Two named tuples (group1, group2), each with fields
        group_size, mu (sample mean) and sigma (sample standard deviation).
    """
    
    ReactionTimeModel = namedtuple("ReactionTimeModel", ["group_size", "mu", "sigma"])
    group1 = ReactionTimeModel(group_size=32, mu=585.2, sigma=89.6)
    group2 = ReactionTimeModel(group_size=32, mu=533.7, sigma=65.3)
    return group1, group2

