0.18.0 (2025-07-14)
-------------------
- Updated the wavelength solution to make the 2d solution more stable. We now fit features row by row in a
  flattened order image

0.17.1 (2025-04-29)
-------------------
- We now build arm64 docker images (for apple silicon) in the automated github actions

0.17.0 (2025-04-26)
-------------------
- Minor fixes to deployment
- Added readthedocs config

0.16.0 (2025-04-10)
-------------------
- Bugfix to not override L1PUBDAT for all frames
- We now anonymize the fringe frames better.

0.15.1 (2025-04-09)
-------------------
- Added documenetation about data products

0.15.0 (2025-03-05)
-------------------
- Migrated setup infrastructure to poetry

0.14.0 (2025-02-06)
-------------------
- Initial order x-positions are now dynamic and stored in the db rather
  than being hard coded.

0.13.0 (2024-12-13)
-------------------
- Updated how we fit the profile center/width to better fit faint traces

0.12.0 (2024-12-11)
-------------------
- We now have prefer calibrations in the following order: same block, same proposal, any public calibration.
- If a block is still going, we delay the processing in case there is a calibration taken at
  at the end of the block that we can use for processing
- We now only use arcs and flats taken with the same slit width as the science data

0.11.2 (2024-11-18)
-------------------
- Simplified the fitting for refining the peak centers. We no longer try to fit them all simultaneously
- Updated the used the line list to remove a less isolated arc line

0.11.1 (2024-11-12)
-------------------
- Fixes to the quality of the reductions
- We now trim the edges of orders better to remove artifacts

0.11.0 (2024-11-05)
-------------------
- Added the ability to combine the extraction from both orders into a single spectrum

0.10.0 (2024-11-05)
-------------------
- Numerous fixes based on commissioning experience
- Added/refactored the necessary logic to re-extract
  the data via the UI

0.9.0 (2024-04-02)
------------------
- Fixes based on Joey's comments
- Deployment fixes
- We now don't keep the filepath of the standards in the db. We assume they are in the archive
  or are in the install director

0.8.0 (2024-03-18)
------------------
- Increased the memory limit on the containers to accomodate stacking
- Simplified the triggering flat stacking to make it more testable

0.7.0
-----
- Deployment fixes
0.6.0
-----
- Deployment fixes

0.5.0 (2023-11-03)
------------------
- Helm value fixes to get the pipeline scheduled on nodes.

0.4.0 (2023-11-02)
------------------
- Helm chart fixes.

0.3.0 (2023-11-01)
------------------
- Full alpha release

0.2.1 (2022-03-22)
------------------
- Added in test data from Siding Springs

0.2.0 (2022-02-23)
------------------
- Added order detection functionality

0.1.0 (2022-02-09)
------------------
- Initial Release
