.. _model-type:

Versions of ARIO model
=======================


ARIO vs :class:`~boario.model_base.ARIOBaseModel` vs :class:`~boario.extended_models.ARIOPsiModel`
_____________________________________________________________________________________________________________

Currently, two model classes are implemented, :class:`~boario.model_base.ARIOBaseModel` and :class:`~boario.extended_models.ARIOPsiModel`.
:class:`~boario.model_base.ARIOBaseModel` is essentially a theoretical implementation used to test a "simplistic and essential" version of the model, and should not
be used directly other than for developing new variants.

:class:`~boario.extended_models.ARIOPsiModel` mostly implements the version presented in :cite:`2013:hallegatte` for the multi-regional case,
as well as (optionally) the intermediate order mechanism presented in :cite:`2020:guan`.

One mechanism currently not implemented is the `macro effect` on final demand described in :cite:`2008:hallegatte`. This mechanism should be implemented in a future update.
