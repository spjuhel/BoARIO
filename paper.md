---
title: 'BoARIO: A Python package implementing the ARIO indirect economic cost model'
tags:
  - Python
  - economy
  - indirect impacts
  - input-output modeling
authors:
  - name: Samuel Juhel
    orcid: 0000-0001-8801-3890
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: CIRED, France
   index: 1
 - name: LMD, France
   index: 2
date: 13 August 2017
bibliography: paper.bib

---

# Summary

The impacts of economic shocks (caused by natural or technological disasters for
instance) often extend far beyond the cost of their local, direct consequences.
Part of these indirect consequences are caused by the propagation of the economic perturbations along supply chains.
Understanding the additional impacts and costs stemming from this propagation is
key to design efficient risk management policies. The interest is rising for the
evaluation of these "indirect risks" in the context of climate change--which
leads to an increase in the average risk of weather extremes
[@lange-2020-projec-expos], and globalized-just-in-time production processes.
Such evaluations rely on dynamic economic models that represent the interactions
between multiple regions and sectors. Recent research in the field argues in
favor of using more Agent-Based oriented model, associated with an increase in
the complexity of the mechanisms represented [@coronese-2022-econom-impac].
However, the assumptions and hypotheses underlying these economic mechanisms
vary a lot, and sometimes lack transparency, making it difficult to properly
interpret and compare results across models, even more so when the code used is
not published or undocumented.

The Adaptive Regional Input-Output model (or ARIO) is an hybrid input-output /
agent-based economic model, designed to compute indirect costs consequent to
economic shocks. Its first version dates back to 2008 and was originally
developed to assess the indirect costs of natural disasters
[@hallegatte-2008-adapt-region]. ARIO is now a well-established and a pivotal
model in its field, has been used in multiple studies, and has seen several
extensions or adaptations [@wu-2011-region-indir; @ranger-2010-asses-poten;
@henriet-2012-firm-networ; @hallegatte-2013-model-role;
@hallegatte-2010-asses-climat; @hallegatte-2008-adapt-region;
@guan-2020-global-suppl; @jenkins-2013-indir-econom; @koks-2015-integ-direc;
@wang-2020-econom-footp; @wang-2018-quant-spatial].

In ARIO, the economy is modelled as a set of economic sectors and regions, and
we call a specific (region, sector) couple an *industry*. Each industry produces
a unique product which is assumed to be the same for all industries of the same
sector. Each industry keeps an inventory of inputs it requires for production.
Each industry answers a total demand consisting of the final demand (from
households, public spendings and private investments) and of the intermediate
demand (from other industries). An initial equilibrium state for the economy is
built based on a multi-regional input-output table. The model can then describe
how the economy, as depicted, responds to a shock (or multiple ones).

`BoARIO` is an open-source Python package implementing the ARIO model. Its core
purpose is to help support better accessibility, transparency, replicability and
comparability in the field of indirect economic impacts modeling.

# Statement of need

Although the ARIO model has been used in multiple studies, and several extensions
exists, only a few implementations of the model or similar ones are openly available.
We found the following existing implementations:

   - A Python implementation of MRIA [@koks-2016-multir-impac].
   - A Python implementation of Disrupt Supply Chain [@colon-2020-critic-analy].
   - A C++ implementation of the Acclimate model [@otto-2017-model-loss].
   - A Matlab implementation of C. Shughrue's model [@shughrue-2020-global-spread].
   - The ARIO models version used in [@wang-2020-econom-footp, @guan-2020-global-suppl].

We found that none of these implementations offer a comprehensive documentation, and are generally
specific to the case study they were used for. The purpose of the `BoARIO` package is to offer
a generic, documented, easy to use, easy to extend, and replicability-oriented model for indirect impact assessment.

The `BoARIO` package allows to easily run simulations with the ARIO model, via
simple steps:

   - Instantiating a model
   - Defining one or multiple events
   - Creating a simulation instance that will wrap the model and events, allow to run the simulation, and explore the results.

The ARIO model relies on Multi-Regional Input-Output Tables (MRIOTs) to define
the initial state of the economy. `BoARIO` was designed to be entirely agnostic
of the MRIOT used, thanks to the `pymrio` package [@stadler2021_Pymrio]. This
aspect notably permits full benefit from the increasing availability of such tables [@stadler18-exiob; @oecd-2021-oecd-inter;
@thissen-2018-eureg; @lenzen-2012-mappin-struc].

The package allows for different shocking events to be defined (currently,
shocks on production or shocks on both production and demand, by including a
demand stemming from the reconstruction effort, the inclusion of shocks on demand only
and other types of shock will be added in future versions).
As such, different types of case studies can be conducted (at different scope, for
multiple or singular events). Users benefit from a precise control on aspects
such as the distribution of the impact towards the different sectors and
regions, the recovery from the impact, and also from the default
modeling choices common in the corresponding literature. The rationale for the detailed
configuration of the model is "allowing for, but not require".

Simulations log the evolution of each variable of interest (production,
production capacity, intermediate demand, reconstruction demand, etc.) at each
step and for each industry, in `pandas DataFrame` objects, allowing in depth
descriptions and understanding of the economic responses. The package can be
used "live", e.g. in a Jupyter Notebook, as well as in large simulation
pipelines, for instance using the `Snakemake` package from @koester-2012-snakem-scalab[^1].

[^1]: Both these uses have already been extensively employed in ongoing studies.

As such, `BoARIO` is designed to be used by researchers in economics and risk
analysis and analysts, and possibly students, either as a theoretical tool to
better understand the dynamics associated with the propagation of economic
impacts, for more applied-oriented case studies in risk management, or simply as
a pedagogical tool to introduce the indirect impact modeling field.

The Python implementation, accompanied by the [online
documentation](https://spjuhel.github.io/BoARIO/) (where a more in depth
description is available), offers an accessible interface for researchers with
limited programming knowledge. It also aims to be modular and extensible to
include additional economic mechanisms in future versions. Finally, its API aims
at making it interoperable with other modeling software: for instance the `CLIMADA`
platform [@gabriela-aznar-siguan-2023-8383171] to which `BoARIO` is in the
process of being integrated.

# Status

`BoARIO` is released under the open-source GPL-3.0 license and is currently
developed by Samuel Juhel. The core of its development was made over the course
of a PhD at CIRED and LMD, under the supervision of Vincent Viguié and Fabio
D'Andrea, and funded by ADEME (the French agency for transition).

`BoARIO` can be installed from PyPi or Conda-Forge using:

    pip install boario

    conda install -c conda-forge boario

# Acknowledgements

I wish to acknowledge Vincent Viguié and Fabio D'Andrea for their support in the
development of `BoARIO` during his PhD, as well as Adrien Delahais for his
feedbacks on the model use. I also want to thank David N. Bresch for indirectly
inspiring me to develop a package for more than just my personal use, and
Alessio Ciullo, for their interest and valuable suggestions as well as
the work done to integrate `BoARIO` to `CLIMADA`.

# References
