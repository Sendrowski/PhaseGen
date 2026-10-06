.. _modules.inference:

Parameter inference
-------------------

Gradient-based parameter inference. An :class:`~phasegen.inference.Inference` object fits a parametrized :class:`~phasegen.distributions.Coalescent` distribution to observed summary statistics by minimizing a loss, typically built from the norms and likelihoods below.

.. rubric:: Classes

.. autosummary::
   :nosignatures:

   ~phasegen.inference.Inference
   ~phasegen.inference.WeightedLoss
   ~phasegen.norms.Norm
   ~phasegen.norms.LNorm
   ~phasegen.norms.L2Norm
   ~phasegen.norms.L1Norm
   ~phasegen.norms.LInfNorm
   ~phasegen.norms.Likelihood
   ~phasegen.norms.PoissonLikelihood
   ~phasegen.norms.MultinomialLikelihood
   ~phasegen.errors.ModelError

.. autoclass:: phasegen.inference.Inference

.. autoclass:: phasegen.inference.WeightedLoss

.. autoclass:: phasegen.norms.Norm

.. autoclass:: phasegen.norms.LNorm

.. autoclass:: phasegen.norms.L2Norm

.. autoclass:: phasegen.norms.L1Norm

.. autoclass:: phasegen.norms.LInfNorm

.. autoclass:: phasegen.norms.Likelihood

.. autoclass:: phasegen.norms.PoissonLikelihood

.. autoclass:: phasegen.norms.MultinomialLikelihood

.. autoclass:: phasegen.errors.ModelError
   :exclude-members: __init__
