Tree building
=============

.. automodule:: boxtree
.. moduleauthor:: Andreas Kloeckner <inform@tiker.net>

Tree data structure
-------------------

.. autoclass:: box_flags_enum
    :members:
    :undoc-members:

.. autoclass:: Tree()

    **Methods**

    .. automethod:: get

    .. automethod:: link_point_sources

Tree with linked point sources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TreeWithLinkedPointSources()

    **Methods**

    .. automethod:: get

Entrypoint
----------

.. autoclass:: TreeBuilder

    .. automethod:: __call__

Traversal building
==================

.. automodule:: boxtree.traversal

Traversal data structure
------------------------

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
