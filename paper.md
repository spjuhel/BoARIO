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
    equal-contrib: true
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
instance) often extend far beyond the cost of their local, direct
consequences, as the economic perturbations they cause propagate along supply
chains. Understanding the additional impacts and costs stemming from this
propagation is key to design efficient risk management policies. The interest is rising
for the evaluation of these "indirect risks" in the context of
climate change (which leads to an increase in the average risk of weather extremes
[@lange-2020-projec-expos]) and globalized-just-in-time production processes.
Such evaluations rely on dynamic economic models that represent the interactions
between multiple regions and sectors. Recent research in the field argues in
favor of using more Agent-Based oriented model, associated with an increase in
the complexity of the mechanisms represented [@coronese-2022-econom-impac].
However, the assumptions and hypotheses underlying these economic mechanisms
vary a lot, and sometime lack transparency, making it difficult to properly
interpret and compare results across models, even more so when the code used is
not published or undocumented.

The Adaptive Regional Input-Output model (or ARIO) is an hybrid input-output /
agent-based economic model, designed to compute indirect costs consequent to
economic shocks. Its first version dates back to 2008 and was originally developed to assess the indirect costs of natural disasters
[@hallegatte-2008-adapt-region]. ARIO is now a well-established and pivotal
model in its field, has been used in multiple studies, and has seen several
extensions or adaptations [@wu-2011-region-indir; @ranger-2010-asses-poten;
@henriet-2012-firm-networ; @hallegatte-2013-model-role;
@hallegatte-2010-asses-climat; @hallegatte-2008-adapt-region;
@guan-2020-global-suppl; @jenkins-2013-indir-econom; @koks-2015-integ-direc;
@wang-2020-econom-footp; @wang-2018-quant-spatial].

In ARIO, the economy is modelled as a set of economic sectors and regions, and
we call a specific (region,sector) couple an *industry*. Each industry produces
a unique product which is assumed to be the same for all industries of the same
sector. Each industry keeps an inventory of inputs it requires for production.
Each industry answers a total demand consisting of the final demand (from
households, public spendings and private investments) and of the intermediate
demand (from other industries). An initial equilibrium state for the economy is
built based on a multi-regional input-output table. The model can then describe
how the economic, as depicted, responds to a shock (or multiple ones).

`BoARIO` is an open-source Python package implementing the ARIO model. Its core
purpose is to help support better accessibility, transparency, replicability and
comparability in the field of indirect economic impacts modeling.

# Statement of need

The `BoARIO` package allows to easily run simulations with the ARIO model, via
simple steps:
- Instantiating a model
- Defining one or multiple events
- Creating a simulation instance that will wrap the model and events, allow to
  run the simulation, and explore the results.

The ARIO model relies on Multi-Regional Input-Output Tables (MRIOTs) to define
the initial state of the economy. `BoARIO` was designed to be entirely agnostic
of the MRIOT used, thanks to the `pymrio` package [@stadler2021_Pymrio]. This
aspect notably allows to fully benefit from the increasing number of such tables
are becoming available [@stadler18-exiob; @oecd-2021-oecd-inter;
@thissen-euregio-2018; @lenzen-2012-mappin-struc].

The package allows for different shocking events to be defined (shock on demand,
shock on production, shock on both, shock involving reconstruction or not, etc).
As such, different types of case-study can be conducted (at different scope, for
multiple or singular events). Users benefit from a precise control on aspects
such as the distribution of the impact towards the different sectors and
regions, the recovery of from the impact, etc. but also from the default
modeling choices common in the corresponding literature. The rationale for detailled configuration of the model is "allowing for, but not require".

Simulations log the evolution of each variable of interest (production,
production capacity, intermediate demand, reconstruction demand, etc.) at each
step and for each industry, in `pandas DataFrames` objects, allowing in depth
descriptions and understanding of the economic responses. The package can be
used "live", e.g. in a Jupyter Notebook, as well as in large simulation
pipelines (for instance using `Snakemake` @koester-2012-snakem-scalab)[^1].

[^1]: Both these uses have already been extensively employed in ongoing studies.

As such, `BoARIO` is designed to be used by researchers in economics and risk
analysis and analysts, and possibly students, either as a theoretical tool to
better understand the dynamics associated with the propagation of economic
impacts, for more applied-oriented case studies in risk management, or simply as
a pedagogical tool to introduce the indirect impact modeling field.

The Python implementation, accompanied by the extensive [online
documentation](https://spjuhel.github.io/BoARIO/) (where a more in depth
description is available), offers an accessible interface for researchers with
limited programming knowledge. It also aims to be modular and extensible to
include additional economic mechanisms in future versions. Finally, its API aims
at making it inter-operable with other modeling software (such as the `CLIMADA`
platform [@gabriela-aznar-siguan-2023-8383171] to which `BoARIO` is in the
process of being integrated).

`BoARIO` is at the core of its author's PhD thesis, and was notably used in
[@juhel-2023-robus], recently submitted to Risk Analysis. Other notable ongoing projects,
are:
- an evaluation of the indirect costs of future floods at the global scope and
comparing its to similar studies using the Acclimate and MRIA models
[@willner-2018-global-econom; @koks-2019-macroec-impac]
- a study on the compounding effect of indirect impacts from multiple events.
- a technical paper on the coupling of `BoARIO` with the `CLIMADA` platform.

# Status

`BoARIO` is released under the open-source GPL-3.0 license and is currently
developed by Samuel Juhel. The core of its development was made over the course
of a PhD at CIRED and LMD, under the supervision of Vincent Viguié and Fabio
D'Andrea, and funded by ADEME (the french agency for transition).

`BoARIO` can be installed from pip using:

    pip install boario

Integration tests can be run using `pytest`

Although its current version is fully operational, further improvements, notably
the implementation of additional economic mechanisms or variations of existing
ones are already planned.

# Acknowledgements

I wish to acknowledge Vincent Viguié and Fabio D'Andrea for their support in the
development of `BoARIO` during his PhD, as well as Adrien Delahais for his
feedbacks on the model use. I also want to thank David N. Bresch for indirectly
inspiring me to develop a package for more than just my personal use, and
Alessio Ciullo, for its interest in the package, its valuable suggestions and
the work done to integrate `BoARIO` to `CLIMADA`.


# References
