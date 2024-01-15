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

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The impacts of economic shocks (caused by natural or technological disasters for
instance) often extend far beyond the cost of replacing their local, direct
consequences, as the economic perturbations they cause propagates along supply
chains. Understanding the additional impacts and costs stemming from this
propagation is key to design efficient risk management policies. The interest
for the evaluation of theses "indirect risks" is raising in the context of
climate change (raising in average the risk of weather extremes
[@lange-2020-projec-expos]) and globalized-just-in-time production processes.
Such evaluations rely on dynamic economic models that represent the interactions
between multiples regions and sectors. Recent research in the field argues in
favor of using more Agent-Based oriented model, associated with an increase in
the complexity of the mechanisms represented [@coronese-2022-econom-impac].
However, the assumptions and hypotheses underlying these economic mechanisms can
vary a lot, and sometime lack transparency, making it difficult to properly
interpret and compare results across models, even more so when the code used is
not published.

The Adaptive Regional Input-Output model (or ARIO) is an hybrid input-output /
agent-based economic model, designed to compute such indirect costs from
economic shocks. Its first version dates back to 2008 and has originally been
developed to assess the indirect costs of natural disasters
[@hallegatte-2008-adapt-region]. ARIO is a well-established and pivotal model in
the field of indirect impacts evaluation [@wu-2011-region-indir;
@ranger-2010-asses-poten; @henriet-2012-firm-networ;
@hallegatte-2013-model-role; @hallegatte-2010-asses-climat;
@hallegatte-2008-adapt-region; @guan-2020-global-suppl;
@jenkins-2013-indir-econom; @koks-2015-integ-direc; @wang-2020-econom-footp;
@wang-2018-quant-spatial].

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
purpose it to help support better accessibility, transparency, replicability and
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
of the MRIOT used, thanks to the `pymrio` ([@stadler2021_Pymrio]) package. This
aspect notably allows to fully benefit from the increasing number of such tables
are becoming available [@stadler18-exiob; @oecd-2021-oecd-inter;
@thissen-euregio-2018; @lenzen-2012-mappin-struc].

Various types of shocking events can be defined (shock on demand, shock on
production, shock on both, shock involving reconstruction or not, etc...), to
allow for the study of a wide range of scenarios.

Simulations register the evolution of each variable of interest (production,
production capacity, intermediate demand, reconstruction demand, etc.) at each step
and for each industry, enabling in depth description and understanding of the economic response.

As such, `BoARIO` is designed to be used by economist researchers and analyst,
and possibly students, either as a theoretical tool to better understand the
dynamics associated with the propagation of economic impacts, for more applied
oriented research, or simply to discover the indirect impact modeling field.

The python implementation, accompanied by the extensive [online
documentation](https://spjuhel.github.io/BoARIO/) (where a more in depth
description is available), offers an accessible interface for researchers with
limited programming knowledge. It also aims to be modular and extensible to
include additional economic mechanisms in future versions. Finally, its API aims
at making it inter-operable with other modeling software (such as the `CLIMADA`
platform [@gabriela-aznar-siguan-2023-8383171] to which `BoARIO` is being
integrated).

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
inspiring me to develop a package for more that just my personal use, and
Alessio Ciullo, for its interest in the package, its valuable suggestions and
the work done to integrate `BoARIO` to `CLIMADA`.


# References
