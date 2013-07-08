Tree building
=============

.. currentmodule:: boxtree

Tree data structure
-------------------

.. autoclass:: box_flags_enum
    :members:
    :undoc-members:

.. autoclass:: Tree()

    .. rubric:: Methods

    .. automethod:: get

Tree with linked point sources
------------------------------

.. currentmodule:: boxtree.tree

.. autoclass:: TreeWithLinkedPointSources

    .. rubric:: Methods

    .. automethod:: get

.. autofunction:: link_point_sources

Filtering the lists of targets
------------------------------

.. currentmodule:: boxtree.tree

.. autoclass:: FilteredTargetListsInUserOrder

    .. rubric:: Methods

    .. automethod:: get

.. autofunction:: filter_target_lists_in_user_order

.. autoclass:: FilteredTargetListsInTreeOrder

    .. rubric:: Methods

    .. automethod:: get

.. autofunction:: filter_target_lists_in_tree_order

Build Entrypoint
----------------

.. currentmodule:: boxtree

.. autoclass:: TreeBuilder

    .. automethod:: __call__


.. vim: sw=4
