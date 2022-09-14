.. _boario-math:

########################
Mathematical background
########################

This page details the mathematical background of the ARIO model.

.. _boario-math-notations:

Notations
===========

    ..
       ":math:`\Damage_{\textrm{Tot}} = \Damage_{\textrm{H}} + \Damage_{\textrm{G}} +  \Damage_{\textrm{K}}`","Total direct losses caused by the event, and  Households, Government and Private sectors subparts","scalar"
       ":math:`\delta_{\textrm{H}}`","Share of :math:`\Damage_{\textrm{Tot}}` impacting the  Private Households final demand such that :math:`\Damage_{\textrm{H}} =  \delta_{\textrm{H}} \Damage_{\textrm{Tot}}`","scalar"
       ":math:`\delta_{\textrm{G}}`","Share of :math:`\Damage_{\textrm{Tot}}` impacting the  Government final demand such that :math:`\Damage_{\textrm{G}} =  \delta_{\textrm{G}} \Damage_{\textrm{Tot}}`","scalar"
       ":math:`\delta_{\textrm{K}}`","Share of :math:`\Damage_{\textrm{Tot}}` impacting the  Private sector demand such that :math:`\Damage_{\textrm{K}} = \delta_{\textrm{K}}  \Damage_{\textrm{Tot}}`","scalar"
       ":math:`\delta_{\textrm{Sector}}`","Share of :math:`\Damage_{\textrm{H}}`,  :math:`\Damage_{\textrm{G}}` or :math:`\Damage_{\textrm{K}}` creating additional demand  in the denoted sector","scalar"
       ":math:`\Gamma_{\textrm{Sector}}`","Additional demand toward denoted sector due to damages","scalar"

.. csv-table:: Notations table
    :header: Notation,Description,Dimension,Variable/Parameter name
    :widths: 30,30,20,20

    ":math:`\sectorsset{} = \{ S_1, \ldots, S_n\}`","Set of sectors indices",":math:`|\sectorsset| = n`"
    ":math:`\regionsset = \{ R_1, \ldots, R_m\}`","Set of regions indices (countries)",":math:`|\regionsset| = m`"
    ":math:`\rfirmsset = \{f^R_1, \ldots,  f^R_n\}`","Set of industries of country :math:`R` (:math:`f^R_{S}` is the industry of sector :math:`S` of region :math:`R`)",":math:`|\rfirmsset| = n`"
    ":math:`\firmsset = \{\rfirmsset[R_1], \ldots, \mathbf{F}^{R_{m}}\}`","Global set of industries (a sector in a region) indices.",":math:`|\firmsset| = n \times  m = p`"
    ":math:`\ioz^{RR'}`","Inter-regional transaction matrix (flows are from region :math:`R` to  :math:`R'` and industry :math:`f` to :math:`f'`)",":math:`\ioz^{RR'} = (z_{ff'}^{RR'})_{\substack{f      \in \rfirmsset[R]\\f'      \in \rfirmsset[R']\\(R,R') \in \regionsset}}`"
    ":math:`\ioz = \begin{bmatrix}               \ioz^{R_1,R_1} & \hdots & \ioz^{R_1,R_m}\\        \vdots & \ddots & \vdots\\               \ioz^{R_m,R_1} & \hdots & \ioz^{R_m,R_m}      \end{bmatrix}`","Global transaction matrix",":math:`\ioz = (z_{ff'})_{f,f'    \in \firmsset}`","``Z_0``, ``intmd_demand``"
    ":math:`\ioy^{RR'}`","Inter-regional final demand matrix (flow from industry :math:`f` of  region :math:`R` towards final demand of region :math:`R'`)",":math:`\ioy^{RR'} =  (y_{f}^{RR'})_{\substack{f \in \rfirmsset\\(R,R') \in \regionsset}}`"
    ":math:`\ioy= \begin{bmatrix}          \ioy^{R_1,R_1} & \hdots & \ioy^{R_1,R_m}\\          \vdots & \ddots & \vdots\\          \ioy^{R_m,R_1} & \hdots & \ioy^{R_m,R_m}        \end{bmatrix}`","Global final demand matrix (total flow from firm indexed by :math:`f` toward  final demand)",":math:`\ioy  = (y_{f})_{f \in \firmsset}`", "``Y_0``, ``final_demand``"
    ":math:`\iov^R =      \begin{bmatrix}        v^{R}_{1} & \hdots & v^{R}_{n} \end{bmatrix}`","Regional value added vector",":math:`\iov^R =  (v_{f}^R)_{\substack{f \in \rfirmsset[R]\\R \in \regionsset}}`"
    ":math:`\iov =      \begin{bmatrix}        \iov^{R_1} & \hdots & \iov^{R_m}\\      \end{bmatrix}`","Global value added vector",":math:`\iov =  (v_{f})_{f \in \firmsset}`", ``VA_0``
    ":math:`\iox`","Production vector",":math:`\iox = (x_{f})_{f \in \firmsset}`", ``production``
    ":math:`\ioa^{RR'}`","Inter-regional technical coefficients matrix",":math:`\ioa^{RR'} =  (a_{f,f'}^{RR'})_{\substack{f \in \rfirmsset[R]\\f' \in \rfirmsset[R']\\(R,R') \in \regionsset}}`"
    ":math:`\ioa=    \begin{bmatrix}      \ioa^{R_1,R_1} & \hdots & \ioa^{R_1,R_m}\\      \vdots & \ddots & \vdots\\      \ioa^{R_m,R_1} & \hdots & \ioa^{R_m,R_m}    \end{bmatrix}`","Global technical coefficients matrix",":math:`\ioa =  (a_{f,f'})_{f,f' \in \firmsset}`"
    ":math:`\ioava = (a^{\textrm{va}}_{f})_{f \in \firmsset}`","Global value added technical coefficients matrix"
    ":math:`\ioinv`","Inventories/Stocks matrix",":math:`\ioinv = (\omega^f_{i})_{\substack{f      \in      \firmsset\\ i \in \sectorsset}}`",``matrix_stock``
    ":math:`\Damage_{\textrm{Tot}} = \begin{bmatrix} \gamma_{1} \hdots \gamma_{n \times m} \end{bmatrix}`","Total direct losses caused by the event","scalar",``rebuild_demand``
    ":math:`\cdot(t)`","Value of <:math:`\cdot`> at step :math:`t`, where <:math:`\cdot`> can be any  scalar,  vector or matrix",""
    ":math:`\psi`","Inventories heterogeneity parameter","scalar in :math:`[0,1]`",``psi`` (``"psi_param"`` in dict/json)
    ":math:`\alpha^b`","Base overproduction capacity (same for all sectors and regions)","scalar",``overprod_base`` (``"alpha_base"`` in dict/json)
    ":math:`\alpha^{\textrm{max}}`","Maximum overproduction capacity","scalar",``overprod_max`` (``"alpha_max"`` in dict/json)
    ":math:`\tau_{\alpha}`","Overproduction increase/decrease characteristic time","scalar",``overprod_tau`` (``"alpha_tau"`` in dict/json)
    ":math:`\tau_{\textrm{INV}}`","Characteristic time of inventory restoration","scalar", ``restoration_tau`` (``"inventory_restoration_time"`` in dict/json)
    ":math:`\tau_{\textrm{REBUILD}}`","Characteristic time of rebuilding", "scalar", ``rebuild_tau``
    ":math:`\mathbf{s}`","Initial/Objective inventory vector",":math:`\mathbf{s} = (s_{i})_{i \in \sectorsset}`", ``inv_duration`` (``"inventory_dict"`` in dict/json)
    ":math:`\mathbf{\kappa}`","Capital stock to value added ratio",":math:`\mathbf{\kappa} = (\kappa_{i})_{i \in \sectorsset}`", ``kstock_ratio_to_VA`` (``"capital_ratio_dict"`` in dict/json)
    ":math:`\mathbf{d}`","Damage distribution vector",":math:`\mathbf{d} = (d_{i})_{i \in \firmsset}`", ``kstock_ratio_to_VA`` (``"capital_ratio_dict"`` in dict/json)

.. note::

  We use sets of indices for industries, sectors and regions to simplify the notation of matrix elements.
  As such we use both :math:`\sum_{f' \in \firmsset} (z_{ff'})` and :math:`\sum_{1 \leq f' \leq p} (z_{ff'})`
  to designate the sum of all flows from industry :math:`f` (resp. indexed by :math:`f`) to other industries.

  Additionally, we use :math:`A = B \odot C`, the Hadamard product (element wise product) (i.e. :math:`a_{ij} = b_{ij} \cdot c_{ij}`)

Model(s) details
===================

.. _boario-math-init:

Initial state
--------------

We build :math:`\ioz`, :math:`\ioy` and :math:`\iov` using input output tables.
We compute initial production :math:`\iox_0`, technical matrix :math:`\ioa` and
value added technical matrix :math:`\ioava` as follows:

.. math::
   :nowrap:

    \begin{gather*}
    \iox_0 = \ioz \cdot \mathbf{i} + \ioy \cdot \mathbf{i}\\
    \ioa = \ioz \cdot \mathbf{\hat{x}}_0^{-1}\\
    \ioava = \ioy \cdot \mathbf{\hat{x}}_0^{-1}
    \end{gather*}


Where:

* :math:`\mathbf{i}` is a summation column vector of size :math:`s \times r` (number of sectors times regions)
* :math:`\mathbf{\hat{x}_0}` is the diagonal matrix with the elements of :math:`\iox_0`

.. note::

   Note that we divide these yearly values by the ``timestep_dividing_factor`` :ref:`parameter <boario-sim-params-time>` in order to obtain an approximation of the productions and demands per time unit (most often days).

We also compute the following :

.. _boario-math-z-agg:

.. math::
   :nowrap:

   \begin{gather*}
    \ioa^{\sectorsset} = \mathbf{I_{\textrm{sum}}} \cdot  \ioa\\
    \ioz^{\sectorsset} = \mathbf{I_{\textrm{sum}}} \cdot  \ioz\\
    \ioz^{\textrm{Share}} =  \ioz \oslash \left ( \ioz^{\sectorsset} \otimes \underbrace{\begin{bmatrix} 1 \\ \vdots\\ 1 \end{bmatrix}}_{\text{m-sized}} \right )
   \end{gather*}

Where :

* :math:`\mathbf{I_{\textrm{sum}}}` is a row summation matrix which aggregates by sector.


.. math::

    \mathbf{I_{\textrm{sum}}} =
    \underbrace{
        \begin{bmatrix}
          1 & \cdots & 0 & & 1 & \cdots & 0 \\
          \vdots & \ddots & \vdots & \cdots & \vdots & \ddots & \vdots \\
          0 & \cdots & 1 & & 0 & \cdots & 1
        \end{bmatrix}
      }_{r \times s} s

* :math:`\otimes` is the Kronecker product which repeat :math:`m` times each row of :math:`\ioz^{\sectorsset}`.

* :math:`\oslash` is the matrix element-wise division, defined such that dividing by zero gives zero. (If initial order to an industry was null, the share ordered is also null)


Hence, :math:`\ioa^{\sectorsset}` and :math:`\ioz^{\sectorsset}` are respectfully the technical matrix and the intermediate demand matrix, aggregated by sector. :math:`\ioz^{\textrm{Share}}` represents for each firm and for each of their input, the share of the total ordered to a supplier, initially (i.e. :math:`\ioz` normalized for each input).

.. _boario-math-initial-inv:

The initial inventory matrix :math:`\ioinv` is initialized as follows :

.. math::
   :nowrap:

    \begin{equation*}
      \ioinv(t=0) = \mdefentry{\omega}[0][i][n][f][p][][] =
      \begin{bmatrix}
      \mathbf{s} \hdots \mathbf{s}
      \end{bmatrix}
      %% \colvec{s_1 \hdots s_1}{s_n \hdots s_n}
      \odot \underbrace{
      \begin{bmatrix} \iox(0)\\
      \vdots\\
      \iox(0) \end{bmatrix}}_{\substack{\iox(0)\\
      n\text{ times}}} \odot \ioa^{\sectorsset} = \colvec{s_1 x_1(0) a_{11} \hdots s_1 x_{p}(0) a_{1p}}{s_n x_{1}(0) a_{n1} \hdots s_n x_{p}(0) a_{np}}
    \end{equation*}

Such that :math:`\omega_{if}(0) = s_i \cdot x_{f}(0) \cdot a_{if}` is the
exact amount of product :math:`i` required by industry :math:`f` to produce
:math:`x_{f}(0)` (i.e. the initial equilibrium production of :math:`f`) during :math:`s_i` days.
Hence, all industries start with a stock of each of their intermediate inputs equal to the
amount required for :math:`s_i` days of production at initial production capacity.

.. note::

  :math:`s_i` does not differ on a per-industry basis, only on a per-product basis, (although this could be implemented easily).

The order matrix :math:`\ioorders` is initialized to be equal to :math:`\ioz` :

.. math::
   :nowrap:

    \begin{equation*}
        \ioorders(t=0) = \left ( o_{ff'}(t=0) \right )_{\substack{f
        \in \rfirmsset[R]\\f'
        \in \rfirmsset[R']\\(R,R') \in \regionsset}} = \ioz
    \end{equation*}

And where :math:`o_{ff'}` is the order made by firm :math:`f'` to firm :math:`f`.

.. _boario-math-dyn:

Model dynamics
-----------------

.. _boario-math-prod:

Production module
^^^^^^^^^^^^^^^^^^^^

At each time step :math:`t`, we compute :math:`\iox^a(t)` the vector of actual production for each industry :math:`f \in \firmsset` during this step.

Let :math:`\mathbf{\alpha} = (\alpha_{f})_{f \in \firmsset}` be the vector of overproduction such that :math:`\alpha_{f}` is the overproduction factor of industry :math:`f` and let :math:`\Delta_{f}(t)` be the initial loss of production capacity of industry :math:`f_S^R` :
Production capacity of industry :math:`f` at step :math:`t` before constraints is:

.. math::
   :nowrap:

    \begin{equation*}
      x^{Cap}_{f}(t) = \alpha_{f}(t) (1 - \Delta_{f}(t)) x_{f}(t)
    \end{equation*}

Once we have production capacity, we can compute actual production:

.. math::
   :nowrap:

    \begin{alignat*}{4}
          \mathbf{D}^{\textrm{Tot}}(t) &= (d_{f}^{\textrm{Tot}}(t))_{f \in \firmsset} &&= \ioorders(t) \cdot \irowsum + \ioy \cdot \irowsum + \Damage_{\firmsset} \cdot \tau_{\textrm{REBUILD}} && \text{Total demand matrix} \\
          & && &&\\
          \iox^{\textrm{Opt}}(t) &= (x^{\textrm{Opt}}_{f}(t))_{f \in \firmsset} &&= \left ( \min \left ( d^{\textrm{Tot}}_{f}(t), x^{\textrm{Cap}}_{f}(t) \right ) \right )_{f \in \firmsset} && \text{Optimal production}\\
          & && &&\\
          \ioinv^{\textrm{Cons}}(t) &= (\omega^{\textrm{Cons},f}_p(t))_{\substack{p \in \sectorsset\\f \in \firmsset}} &&=
             \begin{bmatrix}
               s^{1}_1 & \hdots & s^{p}_1 \\
               \vdots & \ddots & \vdots\\
               s^1_n & \hdots & s^{p}_n
             \end{bmatrix}
    \odot \begin{bmatrix} \iox^{\textrm{Opt}}(t)\\
    \vdots\\
    \iox^{\textrm{Opt}}(t) \end{bmatrix} \odot \ioa^{\sectorsset} && \\
    &&&= \begin{bmatrix}
    s^{1}_1 x^{\textrm{Opt}}_{1}(t) a_{11} & \hdots & s^{p}_1 x^{\textrm{Opt}}_{p}(t) a_{1p}\\
    \vdots & \ddots & \vdots\\
    s^1_n x^{\textrm{Opt}}_{1}(t) a_{n1} & \hdots & s^{p}_n x^{\textrm{Opt}}_{p}(t) a_{np}
    \end{bmatrix}
    \cdot \psi && \text{Inventory constraints}  \\
    & && &&\\
          \iox^{a}(t) &= (x^{a}_{f}(t))_{f \in \firmsset} &&= \left \{ \begin{aligned}
                                                                          & x^{\textrm{Opt}}_{f}(t) & \text{if $\omega_{p}^f(t) \geq \omega^{\textrm{Cons},f}_p(t)$} \forall p\\
                                                                          & x^{\textrm{Opt}}_{f}(t) \cdot \min_{p \in \sectorsset} \left ( \frac{\omega^s_{p}(t)}{\omega^{\textrm{Cons,f}}_p(t)} \right ) & \text{if $\omega_{p}^f(t) < \omega^{\textrm{Cons},f}_p(t)$}
                                                                       \end{aligned} \right. \quad && \text{Actual production at $t$}
    \end{alignat*}


First we compute the total demand directed towards each industry with eq. :math:`\text{Total demand matrix}`. Then we compute optimal production without inventory constraints for each industry as the minimum between production capacity (possibly reduced by damages) and total demand, assuming an industry will not produce more than its clients demand (eq. :math:`\text{Optimal production}`).

We define inventory constraints :math:`\ioinv^{\textrm{Cons}}` for each input, as a share :math:`\psi` of the amount of stocks required to produce :math:`s_p^f` days of production at the level of production of the previous step (eq. :math:`\text{Inventory constraints}`).

.. note::

  :class:`ARIOBaseModel` offers a simplified version of the model where :math:`\psi = 1` (among other simplifications).

If the inventory of product :math:`p \in \sectorsset` of an industry :math:`f` is lower than its required level, then :math:`f`'s production is reduced. An inventory shortage of :math:`x` % (w.r.t. its constraint) leads to a :math:`x` % reduction of production.

Distribution and inventory module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _boario-math-distrib:

Once we have realized production, we can compute how it is distributed among clients (industries, final and rebuilding demands). If :math:`d_f^{\textrm{Tot}}(t) = x_f(t)`, each client receive their order. If :math:`d_f^{\textrm{Tot}}(t) > x_f(t)`, each client receive a share of their order as per a proportional rationing scheme:

.. math::
   :nowrap:

    \begin{alignat*}{4}
      &\ioorders^{\textrm{Received}}(t) &&= \left (\frac{o_{ff'}(t)}{d^{\textrm{Tot}}_f(t)} \cdot x^a_f(t) \right )_{f,f'\in \firmsset}\\
      &\ioy^{\textrm{Received}}(t) &&= \left ( \frac{y_{f}}{d^{\textrm{Tot}}_f(t)}\cdot x^a_f(t) \right )_{f\in \firmsset}\\
      &\Damage^{\textrm{Repaired}}(t) &&= \left ( \frac{\gamma_{f} \cdot \tau_{\textrm{REBUILD}}}{d^{\textrm{Tot}}_f(t)} \cdot x^a_f(t) \right )_{f\in \firmsset}\\
    \end{alignat*}

Where :math:`\damage_f` is the total rebuilding demand towards industry :math:`f` and :math:`\tau_{\textrm{REBUILD}}` is the rebuilding characteristic time.

Once distribution is done, we can compute the new inventories:

.. math::
   :nowrap:

    \begin{alignat*}{4}
      &\ioinv(t+1) &&= \ioinv(t) + \underbrace{\left ( \mathbf{I}_{\textrm{sum}} \cdot \ioorders^{\textrm{Received}}(t) \right )}_{\text{orders received aggregated by inputs}} - \underbrace{\left ( \colvec{\iox^{\textrm{a}}(t)}{\iox^{\textrm{a}}(t)} \odot \ioa^{\sectorsset} \right )}_{\text{inputs used during production}}\\
    \end{alignat*}

Order module
^^^^^^^^^^^^^^

.. _boario-math-orders:

We compute the orders made by each industries towards their suppliers.
Industries seek to restore the inventory of each of their input to their goal level.

.. note::

  Note that this goal can vary during simulation as it depends on :math:`\iox^{\textrm{Opt}}_t` (and not :math:`\iox_0`)).

.. note::

  In :class:`ARIOBaseModel`, the 'gap' matrix is simply the difference between :math:`\ioinv^{*}(t)` and :math:`\ioinv(t)` and orders to suppliers are then proportional to the initial transaction matrix (See definition of :math:`\ioz^{\textrm{Share}}`).
  In :class:`ARIOModelPsi`, only a fraction of missing inventories are ordered, but in addition, the totality of inputs used for production during this step is also ordered.
  The differences are shown in red.

.. math::
   :nowrap:

    \begin{alignat*}{4}
       &\ioinv^{*}(t) &&= (\omega_p^{*,f}(t))_{\substack{p \in \sectorsset\\f \in \firmsset}} \quad = \quad s^{f}_p \cdot \begin{bmatrix} \iox^{\textrm{Opt}}(t)\\ \vdots\\ \iox^{\textrm{Opt}}(t) \end{bmatrix} \odot  \ioa^\sectorsset && \quad && \text{Inventory goals} \\
       &\ioinv^{\textrm{Gap}}(t) &&= (\omega_p^{\textrm{Gap},f}(t))_{\substack{p \in \sectorsset\\f \in \firmsset}} \quad = \quad \left ( \ioinv^{*} - \ioinv(t) \right )_{\geq 0} && \quad && \text{Inventory gaps}\\
       &\ioorders^{\sectorsset}(t) &&= \mathcolor{red}{\frac{1}{\tau_{\textrm{Inv}}}} \cdot \ioinv^{\textrm{Gap}}(t) \mathcolor{red}{ + \begin{bmatrix} \iox^a(t)\\ \vdots\\ \iox^a(t) \end{bmatrix} \odot  \ioa^{\sectorsset}} &&\quad && \text{Intermediate demand total orders}\\
       &\ioorders(t) &&= \left ( \ioorders^{\sectorsset}(t) \otimes \underbrace{\begin{bmatrix} 1 \\ \vdots\\ 1 \end{bmatrix}}_{\text{m-sized}} \right ) \odot  \ioz^{\textrm{Share}} &&\quad && \text{Intermediate demand orders}
    \end{alignat*}

* In eq. :math:`\text{Inventory goals}` we compute inventory goals based on optimal production. (Note that, in the version with ``psi``, :math:`\Omega^* = \frac{\Omega^{\textrm{Cons}}}{\psi}`)
* In eq. :math:`\text{Inventory gaps}` we compute the inventory gaps. :math:`(\mathbf{A} - \mathbf{B})_{\geq 0}` denotes the resulting matrix of :math:`\mathbf{A} - \mathbf{B}` where negative values are replaced by 0.
* In eq. :math:`\text{Intermediate total demand orders}` we compute aggregate orders for intermediate demand.
* In eq. :math:`\text{Intermediate demand orders}` we compute the actual order matrix, by distributing total orders along the different suppliers.
  We then have two variants:

  1. We multiply by the initial transaction share :math:`\ioz^{\textrm{Share}}`

  2. We multiply by the initial transaction share, weighted by suppliers current production. To do so, we replace both occurrence of :math:`\ioz` in :math:`\ioz^{\textrm{Share}}` by the following:

     .. math::

       \ioz^{*}(0) = \ioz(0) \odot \underbrace{\begin{bmatrix} \iox(t) \oslash \iox(0) \\ \vdots\\ \iox(t) \oslash \iox(0) \end{bmatrix}}_{\text{m-sized}}


The first variant corresponds to the order module of [`Hallegatte 2013`_] while the other one corresponds to the one defined in [`Guan 2020`_] (See :ref:`order module parameter <order module>` for how to choose implementation).


.. _boario-math-overprod:

Overproduction module
^^^^^^^^^^^^^^^^^^^^^^^^^

Lastly, we compute overproduction. If demand is higher than production, industries start overproducing to adapt. We use the same definition as in [`Hallegatte 2013`_], which uses a scarcity index :math:`\zeta(t)`:

.. math::
   :nowrap:

    \begin{alignat*}{3}
      & \zeta(t) &&= \frac{d_{f}^{\textrm{Tot}}(t) - x^{a}_f(t)}{d_{f}^{\textrm{Tot}}(t)}\\
      & \alpha_f(t+1) &&= \begin{cases}
             \alpha_f(t) + (\alpha^{\textrm{max}} - \alpha_f(t)) \cdot \zeta(t) \cdot \frac{1}{\tau_{\alpha}} & \text{if } \zeta(t) > 0\\
             \alpha_f(t) +  (\alpha^{\textrm{b}}  - \alpha_f(t)) \cdot \frac{1}{\tau_{\alpha}}                & \text{if } \zeta(t) \leq 0\\
                      \end{cases}
    \end{alignat*}


.. _boario-math-events:

Event impact
--------------

We represent the impact of the event via two effects :

1. A decrease of the production capacity of the impacted sectors.
2. An additional final demand corresponding to the capital destroyed and addressed towards the rebuilding sectors.

See :ref:`aff-sectors-params` and :ref:`reb-sectors-params` for the corresponding parameters.

.. _boario-math-prodcapdec:

Production capacity decrease
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


We define :math:`\Delta_{f}(0)` the initial loss of production capacity of industry :math:`f_S^R` as the fraction of capital stock destroyed:

.. math::
   :nowrap:

    \begin{equation*}
     \Delta_f(0) = \frac{
                         \Damage_{\textrm{f}}(0) \cdot d_f
                         }{
                         \mathbf{\kappa}_f \cdot v_{f}
                        }
    \end{equation*}


We update :math:`\Delta_f` during every step according to how much damages remain :

.. math::
   :nowrap:

    \begin{equation*}
            \Delta_{f}(t) = \Delta_{f}(0)
    \frac{\Damage_{\textrm{Tot}}(t)}{\Damage_{\textrm{Tot}}(0)}
       \end{equation*}

.. _boario-math-rebuilding-demand:


Recovery module
^^^^^^^^^^^^^^^^^^

.. _boario-math-recovery:

.. math::
   :nowrap:

    \begin{equation*}
      \Damage_{\textrm{Tot}}(t+1) = \Damage_{\textrm{Tot}}(t) - \Damage^{\textrm{Repaired}}(t)
    \end{equation*}


.. _`Hallegatte 2013`: https://doi.org/10.1111/j.1539-6924.2008.01046.x

.. _`Guan 2020`: https://www.nature.com/articles/s41562-020-0896-8
