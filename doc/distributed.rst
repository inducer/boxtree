Distributed Computation
=======================

High-level interface
--------------------

To perform stardard point-FMM, first construct a
:class:`boxtree.distributed.DistributedFMMRunner` object. The constructor will
distribute the necessary information from the root rank to all worker ranks. Then,
the :meth:`boxtree.distributed.DistributedFMMRunner.drive_dfmm` can be used for
launching FMM.

.. autoclass:: boxtree.distributed.DistributedFMMRunner

    .. automethod:: drive_dfmm

FMM Computation
---------------

.. autoclass:: boxtree.distributed.calculation.DistributedExpansionWrangler
    :members:
