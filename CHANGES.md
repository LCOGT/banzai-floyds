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
