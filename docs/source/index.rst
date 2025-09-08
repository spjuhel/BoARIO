:html_theme.sidebar_secondary.remove: true
:notoc:

Welcome to BoARIO's documentation!
##################################

**Date**: |today| **Version**: |version|

**Useful links**:
`Binary Installers <https://pypi.org/project/boario>`__ |
`Source Repository <https://github.com/spjuhel/BoARIO>`__ |
`Issues & Ideas <https://github.com/spjuhel/BoARIO/issues>`__

BoARIO, is an open-source, GPL-3.0 licensed, python implementation project of the Adaptative Regional Input Output (ARIO) model (see :cite:t:`2008:hallegatte` and :cite:t:`2013:hallegatte`).

Its objectives are to give an accessible and inter-operable implementation of ARIO, as well as tools to visualize and analyze simulation outputs and to evaluate the effects of many parameters of the model.

BoARIO has been published in the Journal of Open Source Software :cite:`2024a:juhel`.

.. _`Hal13`: https://doi.org/10.1111/j.1539-6924.2008.01046.x

.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Getting Started
        :shadow: md

        Setting it up quickly

        +++

        .. button-ref:: getting-started
            :ref-type: doc
            :click-parent:
            :color: secondary
            :expand:

            To the getting started [WIP]


    .. grid-item-card:: User Guide
        :shadow: md

        New to *BoARIO*? Check out the User guide. It contains an
        introduction to *BoARIO's* main concepts.

        +++

        .. button-ref:: user-guide
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the user guide



    .. grid-item-card::  API reference
        :shadow: md

        The reference guide contains a detailed description of
        the BoARIO API. The reference describes how the methods work and which parameters can
        be used. It assumes that you have an understanding of the key concepts.

        +++

        .. button-ref:: api-ref
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the reference guide

    .. grid-item-card::  Developer guide
        :shadow: md

        Saw a typo in the documentation? Want to improve
        existing functionalities? The contributing guidelines will guide
        you through the process of improving pandas.

        +++

        .. button-ref:: development
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the development guide

.. toctree::
   :maxdepth: 1
   :hidden:

   getting-started
   user-guide
   boario-api-reference
   development
   release-note
   boario-references
