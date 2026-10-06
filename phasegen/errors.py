"""
Exceptions of PhaseGen.
"""


class ModelError(ValueError):
    """
    Raised when the model cannot be evaluated at its parameters: a demography that cannot absorb, a population size of
    zero in an epoch the computation reaches, rates too far apart for a reliable evaluation, or a moment that is not a
    number. An optimizer may treat it as an invalid region of the parameter space, unlike an error in the arguments or
    in a loss function.
    """
