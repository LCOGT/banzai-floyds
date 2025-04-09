BANZAI-FLOYDS: Data Reduction for FLOYDS spectra
------------------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

BANZAI-FLOYDS is the Las Cumbres Observatory pipeline to process `FLOYDS <https://lco.global/observatory/instruments/floyds/>`_ data. We currently have two FLOYDS instruments in operation: one on Haleakala in Hawaii and one at Siding Springs in Australia. The FLOYDS spectrographs are low-resolution, workhorse spectrographs for time-domain astronomy (e.g. explosive transients, AGN, near-Earth objects, microlensing events). FLOYDS has a resolution of R~500 and covers from 320nm - 1000nm. FLOYDS is double dispersed so the images of the slit are curved on the detector (similar to eschelle spectrographs). The sky lines are also tilted to better sample the line spread function to enable better sky subtraction.
 
The dataflow and infrastructure of this pipeline relies heavily on `BANZAI
<https://github.com/lcogt/banzai>`_, enabling this repo to focus on analysis that is specific to slit spectroscopy.

Installation
------------
.. code-block:: sh

    poetry install


Building Documentation
----------------------
.. code-block:: sh

    poetry install -E docs
    # Install pandoc via apt-get or equivalent
    sphinx-build -W -b html docs docs/_build/html

License
-------

.. |copy| unicode:: 0xA9 .. copyright sign

This project is Copyright |copy| cmccully and licensed under
the terms of the GNU GPL v3+ license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.


Contributing
------------

We love contributions! banzai-floyds is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
banzai-floyds based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
