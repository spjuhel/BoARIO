Release notes
================

Note that release notes here are not always up-to-date, but try to be more descriptive.
(For latest changelog see `GitHub releases <https://github.com/spjuhel/BoARIO/releases>`_)

BoARIO is also still considered in `beta` (No real major 1.0.0 version yet),
yet minor 0.+1.0 versions are reserved for important updates.

0.7.0 (09/2025) - Annie Easley
----------------------------------

* Fix on final demand unmet tracking: The way to compute this attribute brought in `v0.6.4`
  was incorrect.
* Fix on value added tracking: Now sums intermediary input matrix over the correct axis.
* Adds EVENT_ARBITRARY_PRECISION constant to replace hardcoded 6 value when rounding the factors of arbitrary production reduction in this type of event.
* Improves error logs when losses are too high, to be more informative about which sectors are *"too impacted"*.
* Minor updates to the documentations notably reflecting changes to final demand not met and value added tracking.


0.6.4 (07/2025)
-----------------

Final demand not met was not properly computed. It was instead outputting the final demand that was
not be distributed from the supplying industries.

We kept this metric but renamed it correctly to `final_demand_undist` and added the right
"Final demand not met" metric as the final demand that was not fulfilled for the point of view
of supplied regions, distinguished by inputs (final products).

**This new attribute was still incorrectly computed in this version (fixed in 0.7.0)**

0.6.3 (06/2025)
-----------------

* Better authors list in autopopulated files
* Puts git info logging in `debug` level
* Makes `EventArbitraryProd` event type available
* Adds value added tracking in simulation (computed as gross production minus intermediary inputs) **Note this feature had a bug in this version, corrected in v0.7.0**
* Updates events creation documentation to reflect latest changes.

0.6.2 (04/2025)
-----------------

Hot fix for rebuilding demand distribution: There was a bug in the way the rebuilding demand is distributed toward the rebuilding industries,
as the distribution matrix was not grouped by sectors when computing shares. This is now fixed.

0.6.1 (04/2025)
-----------------

Hot fix for _divide_array_ignore(): The function did not use `np.nan_to_num()` correctly,
which produced NaNs in resulting arrays when computing the "market shares" with an intermediate demand
having zeroes. This is now fixed.

0.6.0 (11/2024) - Grace Brewster Hopper
-------------------------------------------

This version is a major refactoring of multiple parts of BoARIO, which should makes it easier to use, as well as more efficient.

BoARIO now has a logo !

Breaking changes:

* Different event types are now instantiated via module level functions to simplify the entry point.
* Dropped support for python 3.9
* Including support for python 3.12

Changes:

* All demands within the model are now in a single numpy array ``_entire_demand`` for efficiency (memory notably). The different demands are accessed/set via properties that return the correct part of the ``_entire_demand`` array. This should greatly alleviate the memory requirements (multiple copies of arrays have been removed) and maybe also decrease computation requirements. Changes should be seamless from user point of view.
* From that the ``distribute_production`` method was greatly simplified.
* Added the ``EventTracker`` class to handle the tracking of events. Event objects are now mostly static (they contain only the initial shock of the event), tracking of their evolution during the simulation is now handled by the ``EventTracker`` class. This should also mostly be seamless from user perspective.
* Warnings are now issued by the warnings modules instead of within the logger.
* (Supposedly) improved efficiency with faster summing functions for matrices.
* For convenience the MRIOT ``IOSystem`` is now an attribute of the model.
* Arbitrary production decrease events are back and working!
* The progress bar is now enabled when creating the ``Simulation`` object.
* The input ``IOSystem`` used to instantiate the model is no longer sorted (The copy used within the model still is, but not the external object)
* Adds a lot of integration tests, although more are still required.
* Removed logger handlers altogether. A logger is still defined and used, but how to actually show/register the logs is now entirely up to the user.
* Lots of new things in the documentation and docstrings.

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
    * `recovery_time` renamed to `recovery_tau` (for consistency with `rebuild_tau`)

Fixes:

* Definitely fixes:
  - https://github.com/spjuhel/BoARIO/issues/113
  - https://github.com/spjuhel/BoARIO/issues/110
  - https://github.com/spjuhel/BoARIO/issues/104
  - https://github.com/spjuhel/BoARIO/issues/111
  - https://github.com/spjuhel/BoARIO/issues/95
  - https://github.com/spjuhel/BoARIO/issues/96

* Probably fixes https://github.com/spjuhel/BoARIO/issues/130 (At least with series)

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
