###########
Terminology
###########

So far, there is no consistent terminology for MRIO systems and parameters in the scientific community.
For BoARIO, the following variable names are used (the alias columns are other names and abbreviations often found in the literature):

.. list-table::
   :widths: 30 70 30
   :header-rows: 1

   * - BoARIO terminology
     - Description
     - Aliases
   * - Region
     - An economic region represented in the MRIO considered
     - /
   * - Sector
     - An economic sector represented in the MRIO considered (which most often both designate the sector itself and what it produces)
     - Branch, product
   * - Industry
     - A sector in a region, ie a (region,sector) couple.
     - Firm
   * - Input
     - An intermediate input used by an industry to produce (which corresponds to a sector)
     - /
   * - Inventories / (Inputs) stocks
     - A 'buffer' of intermediate input available to an industry
     - /
   * - Direct losses
     - The productive capital destroyed by an event, leading to a reduction of production capacity for the affected industries (Stock)
     - Asset damages, direct damages
   * - Direct production losses
     - The amount of production not realised (ie lost) due to the production capacity being reduced by direct losses. (Flow)
     - /
   * - Indirect (production) losses
     - The amount of production indirectly lost consequent to the direct production losses.
     - /
   * - Production
     - The gross output of industries
     - /
   * - Overproduction
     - The capacity of an industry to temporarily increase its production to meet an increased demand.
     - /
   * - Rebuilding demand
     - The possible additional demand resulting from the destruction caused by an event requiring rebuilding.
     - /
   * - Recovering
     - The recovery of destroyed capital and hence production capacity (without an associated rebuilding demand)
     - /
   * - Rebuilding
     - The rebuilding of destroyed capital and hence production capacity (ie recovery via a rebuilding demand)
     - /
