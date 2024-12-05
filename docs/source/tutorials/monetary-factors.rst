.. _monetary-factors:

#################
Monetary factors
#################

Different MRIOTs are often expressed in different units ($,€) and in different `monetary factors`, `e.g.`, units, thousand of units or millions of units.

BoARIO handles this through the explicit definition of such a monetary factor for both the model (corresponding to the MRIOT data) and the events.
If the (explicit) monetary factor of an event differs from the model's one, impact values are automatically converted to the correct one by multiplying them by the ratio `event monetary factor / model monetary factor`.

Setting up monetary factor for the model
============================================

Via the MRIOT
--------------

When instantiating a model object from an MRIOT table (``IOSystem``), we check for the existence of a ``monetary_factor`` attribute within the ``IOSystem`` and use its value if it exists.

This takes advantage of the possibility in python to define attributes on the fly, thus allowing to define this `custom` attribute for an ``IOSystem``.

Via parameter
--------------

The more traditional way to setup the ``monetary_factor`` is simply via the ``monetary_factor`` argument at initialisation.

As most MRIOTs are expressed in millions, the default value is ``10**6``. We still recommend to state it explicitly for transparency.

Setting up monetary factor for an event
===========================================

Setting up the monetary factor for an event is done at initialisation via the ``event_monetary_factor`` argument.

.. attention::
   The default value of this argument is 1.0, thus it differs from the model's one and adjustments are made!

Converting different currencies
==================================

Although it has not be specifically explored or tested, the combination of the two factor could in principle be used to convert from one currency to another by using floating values instead of powers of ten.
