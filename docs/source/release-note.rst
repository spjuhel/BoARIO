Release notes
================

0.6.0 (11/2024)
----------------

Breaking changes:

* Different event types are now instantiated via module level function to simplify entry point.
* Dropped support for python 3.9
* Including support for python 3.12

Changes:

* All demands within the model are now in a single numpy array `_entire_demand` for efficiency (memory notably). The different demands are accessed/set via properties that return the correct part of the `_entire_demand` array. This should greatly alleviate the memory requirements (multiple copies of arrays have been removed) and maybe also decrease computation requirements. Changes should be seamless from user point of view.
* From that the `distribute_production` method was greatly simplify.
* Added the `EventTracker` class to handle the tracking of events. Event objects are now mostly static (they contain only the initial shock of the event), tracking of their evolution during the simulation is now handled by the `EventTracker` class. This should also mostly be seamless from user perspective.
* Warnings are now issued by the warnings modules instead of within the logger.
* (Supposedly) improved efficiency with faster summing function for matrices.
* The MRIOT IOSystem is now an attribute of the model.

Renaming:

- Within `ARIOBaseModel`
    * `gdp_df` renamed to `gva_df` for consistency
    * `k_stock` renamed to `productive_capital`
    * `k_stock_to_VA_ratio` renamed to `capital_to_VA_ratio`
    * `gdp_share_sector` renamed to `regional_production_share`
    * `matrix_share_thresh` renamed to `matrix_share_thresh`
    * `matrix_stock` renamed to `inputs_stock`
    * `matrix_orders` renamed to `intermediate_demand`
    * `total_demand` renamed to `entire_demand_tot`

Fixes:

* Definitely fix #113 #110 #104 #111
* Should fix #95

0.5.10 - JOSS v1 (06/2024)
---------------------------

This release is associated with the JOSS publication of BoARIO.

Changes:

* Small changes to README and documentation for grammar by @potterzot in https://github.com/spjuhel/BoARIO/pull/119
* JOSS Version 1 by @spjuhel in https://github.com/spjuhel/BoARIO/pull/124

**Full Changelog**: https://github.com/spjuhel/BoARIO/compare/v0.5.9...v0.5.10

0.5.9 (04/2024)
----------------

Changes:

* event: added kwargs in instantiate methods by @spjuhel in https://github.com/spjuhel/BoARIO/pull/103

Dependencies updates:

* Bump numpy from 1.23.5 to 1.26.4 by @dependabot in https://github.com/spjuhel/BoARIO/pull/109
* Bump pytest from 7.4.4 to 8.1.1 by @dependabot in https://github.com/spjuhel/BoARIO/pull/108
* Bump pytest-cov from 4.1.0 to 5.0.0 by @dependabot in https://github.com/spjuhel/BoARIO/pull/107
* Bump sphinx-autodoc-typehints from 2.0.1 to 2.1.0 by @dependabot in https://github.com/spjuhel/BoARIO/pull/106
* Bump dask from 2024.4.1 to 2024.4.2 by @dependabot in https://github.com/spjuhel/BoARIO/pull/105

**Full Changelog**: https://github.com/spjuhel/BoARIO/compare/v0.5.8...v0.5.9

0.5.8 (04/2024)
----------------

* Bump actions/configure-pages from 4 to 5 by @dependabot in https://github.com/spjuhel/BoARIO/pull/94
* v0.5.8 by @spjuhel in https://github.com/spjuhel/BoARIO/pull/102

- Fixed badge in README
- Integrated dependabot in the CI/CD
- Documentation retrofit
- Version switch in documentation
- Multiple dependencies version update

**Full Changelog**: https://github.com/spjuhel/BoARIO/compare/v0.5.7...v0.5.8

0.5.7 (03/2024)
----------------

* Trying to fix dependencies for conda forge by @spjuhel in https://github.com/spjuhel/BoARIO/pull/86

**Full Changelog**: https://github.com/spjuhel/BoARIO/compare/v0.5.6...v0.5.7

0.5.6 (03/2024)
----------------

* Removed the requirement to record in memmaps (variables evolution can be recorder directly in arrays)
* Update to V0.5.6 by @spjuhel in https://github.com/spjuhel/BoARIO/pull/85

**Full Changelog**: https://github.com/spjuhel/BoARIO/compare/v0.5.5...v0.5.6

0.5.5 (02/2024)
----------------

* 📦 🚑 Fixed a problem with multi-events + pandas version by @spjuhel in https://github.com/spjuhel/BoARIO/pull/66
* Create draft-pdf.yml by @spjuhel in https://github.com/spjuhel/BoARIO/pull/71
* V0.5.5 and learning correct workflow ;) by @spjuhel in https://github.com/spjuhel/BoARIO/pull/78

**Full Changelog**: https://github.com/spjuhel/BoARIO/compare/v0.5.3...v0.5.5

0.5.4
------

There is no version 0.5.4

0.5.3 (10/2023)
----------------

Fixed a bug with household rebuilding demand

**Full Changelog**: https://github.com/spjuhel/BoARIO/compare/v0.5.2...v0.5.3


0.5.2 (09/2023)
----------------

**Full Changelog**: https://github.com/spjuhel/BoARIO/compare/v0.5.1...v0.5.2

0.5.1 (08/2023)
----------------

* hotfix for the use of pygit2

0.5.0 (06/2023)
----------------

* Putting in master the nice changes we made when coupling with climada by @spjuhel in https://github.com/spjuhel/BoARIO/pull/30
* Proper merge and Black Formatting (actually working) by @spjuhel in https://github.com/spjuhel/BoARIO/pull/34
* Doc testing merge: master testing by @spjuhel in https://github.com/spjuhel/BoARIO/pull/41
* Master testing by @spjuhel in https://github.com/spjuhel/BoARIO/pull/43
* Update issue templates by @spjuhel in https://github.com/spjuhel/BoARIO/pull/50
* v0.5.0 by @spjuhel in https://github.com/spjuhel/BoARIO/pull/58


**Full Changelog**: https://github.com/spjuhel/BoARIO/compare/v0.4.1b...v0.5.0b
