Tree building
=============

.. automodule:: boxtree
.. moduleauthor:: Andreas Kloeckner <inform@tiker.net>

Output
------

.. autoclass:: box_flags_enum
    :members:
    :undoc-members:

.. autoclass:: Tree()

    .. automethod:: get

Entrypoint
----------

.. autoclass:: TreeBuilder

    .. automethod:: __call__

Traversal building
==================

.. automodule:: boxtree.traversal

Output
------

.. autoclass:: FMMTraversalInfo()

    .. automethod:: get

Entrypoint
----------

.. autoclass:: FMMTraversalBuilder

    .. automethod:: __call__

Abstract FMM driver
===================

.. module:: boxtree.fmm

.. autofunction:: drive_fmm

Tree-based geometric lookup
===========================

.. module:: boxtree.geo_lookup

.. autoclass:: LeavesToBallsLookupBuilder

    .. automethod:: __call__

.. autoclass:: LeavesToBallsLookup

    .. automethod:: get

.. vim: sw=4
