Tree building
=============

.. currentmodule:: boxtree

.. _tree-kinds:

Supported tree kinds
--------------------

The following tree kinds are supported:

- *Nonadaptive* trees have all leaves on the same (last) level.

- *Adaptive* trees differ from nonadaptive trees in that they may have leaves on
  more than one level. Adaptive trees have the option of being
  *level-restricted*: in a level-restricted tree, neighboring leaves differ by
  at most one level.

All trees returned by the tree builder are pruned so that empty leaves have been
removed. If a level-restricted tree is requested, the tree gets constructed in
such a way that the version of the tree before pruning is also level-restricted.

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

Data structures
^^^^^^^^^^^^^^^

.. autoclass:: FilteredTargetListsInUserOrder()

    .. rubric:: Methods

    .. automethod:: get

.. autoclass:: FilteredTargetListsInTreeOrder()

    .. rubric:: Methods

    .. automethod:: get

Tools
^^^^^

.. autoclass:: ParticleListFilter

.. autofunction:: filter_target_lists_in_user_order

.. autofunction:: filter_target_lists_in_tree_order

Build Entrypoint
----------------

.. currentmodule:: boxtree

.. autoclass:: TreeBuilder

    .. automethod:: __call__


.. vim: sw=4
