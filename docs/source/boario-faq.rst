.. _boario-faq:

###############
FAQ
###############

This page lists questions that were raised by users so far and answers, do not hesitate to populate new questions by raising an issue on the GitHub repo.

Section: Model Calibration
==========================

**Q: What variables do I need to care about if I want to change the temporal resolution of the IOT (``iotable_year_to_temporal_unit_factor``)?**

**A:** All the following variable are expressed as a ``iotable_year_to_temporal_unit_factor``th of a year
(days if 365, trimesters if 3, months if 12, etc.). Default values often assume ``iotable_year_to_temporal_unit_factor`` is 365,
so you will need to change them if you change ``iotable_year_to_temporal_unit_factor``:

* **For the Model:**
    - ``rebuild_tau``
    - ``alpha_tau``
    - ``inventory_restoration_tau``
    - ``overprod_tau``
    - ``main_inv_dur``
* **For the Event/Impact Set Up:**
    - ``duration``
    - ``recovery_tau``
    - ``impact``
    - ``occurrence``

* **For the Simulation:**
    - ``n_temporal_units_to_sim``.

.. note:: The temporal resolution has not been extensively tested, and you will receive a warning if you are changing the ``iotable_year_to_temporal_unit_factor``.

---

**Q: Is ``inv_duration`` a vector with the duration (like the variable name says & the model description page) or initial stocks (like the description of this variable says: Initial/Objective inventory vector)?**

**A:** Yes ``inv_duration`` is a vector with the duration (initial and goal) of the inventory: it represents for each sector the theoretical amount of temporal unit during which an industry can produce
without receiving new inputs from that sector. For instance {"Aluminium":30} would mean that industries start with enough Aluminium to produce their product for 30 days, and will try to maintain that level
as a goal, with respect to their production (thus if their production decreases/increases, the absolute amount of Aluminium considered as a goal would become lower/greater.).

**Stocks** (expressed as monetary values) are stored in ``"input_stocks"`` within ``sim.model`` (and updated at each step).
You may also track their evolution by specifically asking when instantiating the ``Simulation`` object.
But given it has a size $n_{industries} \times n_{inputs} \times n_{steps}$, it can grow in size very quickly so it is not tracked by default.

---

Section: Results & Interpretation
=================================

**Q: Why are relatively large production shock values (``EventArbitProd``) not producing meaningful results?**

**A:** Please check whether the shock value is considering the **correct temporal resolution**.

For example, a using a **20% shock** for a **10-day event** represents a reduced **daily** production by 20%,
if you are using annual shock data, this would not match:

$$\frac{10}{365} \times 0.2$$

(The IOT temporal resolution is annual unless changed).

Therefore, the annual shock represented here is $\approx$ **0.5%** (0.005) annual production.
