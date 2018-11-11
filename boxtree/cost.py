import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: F401
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.tools import dtype_to_ctype
from mako.template import Template
from functools import partial
import sys

if sys.version_info >= (3, 0):
    Template = partial(Template, strict_undefined=True)
else:
    Template = partial(Template, strict_undefined=True, disable_unicode=True)

if sys.version_info >= (3, 4):
    from abc import ABC, abstractmethod
else:
    from abc import ABCMeta, abstractmethod
    ABC = ABCMeta('ABC', (), {})


class CostCounter(ABC):
    @abstractmethod
    def collect_direct_interaction_data(self, traversal, tree):
        pass


class CLCostCounter(CostCounter):
    def __init__(self, queue):
        self.queue = queue

    def collect_direct_interaction_data(self, traversal, tree):
        ntarget_boxes = len(traversal.target_boxes)
        particle_id_dtype = tree.particle_id_dtype
        box_id_dtype = tree.box_id_dtype

        count_direct_interaction_knl = ElementwiseKernel(
            self.queue.context,
            Template("""
                ${particle_id_t} *srcs_by_itgt_box,
                ${box_id_t} *source_boxes_starts,
                ${box_id_t} *source_boxes_lists,
                ${particle_id_t} *box_source_counts_nonchild
            """).render(
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_id_t=dtype_to_ctype(box_id_dtype)
            ),
            Template("""
                ${particle_id_t} nsources = 0;
                ${box_id_t} source_boxes_start_idx = source_boxes_starts[i];
                ${box_id_t} source_boxes_end_idx = source_boxes_starts[i + 1];

                for(${box_id_t} cur_source_boxes_idx = source_boxes_start_idx;
                    cur_source_boxes_idx < source_boxes_end_idx;
                    cur_source_boxes_idx++)
                {
                    ${box_id_t} cur_source_box = source_boxes_lists[
                        cur_source_boxes_idx
                    ];
                    nsources += box_source_counts_nonchild[cur_source_box];
                }

                srcs_by_itgt_box[i] = nsources
            """).render(
                particle_id_t=dtype_to_ctype(particle_id_dtype),
                box_id_t=dtype_to_ctype(box_id_dtype)
            ),
            name="count_direct_interaction"
        )

        box_source_counts_nonchild_dev = cl.array.to_device(
            self.queue, tree.box_source_counts_nonchild
        )

        # List 1
        nlist1_srcs_by_itgt_box_dev = cl.array.zeros(
            self.queue, (ntarget_boxes,), dtype=particle_id_dtype
        )
        neighbor_source_boxes_starts_dev = cl.array.to_device(
            self.queue, traversal.neighbor_source_boxes_starts
        )
        neighbor_source_boxes_lists_dev = cl.array.to_device(
            self.queue, traversal.neighbor_source_boxes_lists
        )
        count_direct_interaction_knl(
            nlist1_srcs_by_itgt_box_dev,
            neighbor_source_boxes_starts_dev,
            neighbor_source_boxes_lists_dev,
            box_source_counts_nonchild_dev
        )

        result = dict()
        result["nlist1_srcs_by_itgt_box"] = nlist1_srcs_by_itgt_box_dev.get()

        return result


class PythonCostCounter(CostCounter):
    def collect_direct_interaction_data(self, traversal, tree):
        ntarget_boxes = len(traversal.target_boxes)

        # target box index -> nsources
        nlist1_srcs_by_itgt_box = np.zeros(ntarget_boxes, dtype=np.intp)
        nlist3close_srcs_by_itgt_box = np.zeros(ntarget_boxes, dtype=np.intp)
        nlist4close_srcs_by_itgt_box = np.zeros(ntarget_boxes, dtype=np.intp)

        for itgt_box in range(ntarget_boxes):
            nlist1_srcs = 0
            start, end = traversal.neighbor_source_boxes_starts[itgt_box:itgt_box+2]
            for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
                nlist1_srcs += tree.box_source_counts_nonchild[src_ibox]

            nlist1_srcs_by_itgt_box[itgt_box] = nlist1_srcs

            nlist3close_srcs = 0
            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_smaller_starts is not None:
                start, end = (
                        traversal.from_sep_close_smaller_starts[itgt_box:itgt_box+2])
                for src_ibox in traversal.from_sep_close_smaller_lists[start:end]:
                    nlist3close_srcs += tree.box_source_counts_nonchild[src_ibox]

            nlist3close_srcs_by_itgt_box[itgt_box] = nlist3close_srcs

            nlist4close_srcs = 0
            # Could be None, if not using targets with extent.
            if traversal.from_sep_close_bigger_starts is not None:
                start, end = (
                        traversal.from_sep_close_bigger_starts[itgt_box:itgt_box+2])
                for src_ibox in traversal.from_sep_close_bigger_lists[start:end]:
                    nlist4close_srcs += tree.box_source_counts_nonchild[src_ibox]

            nlist4close_srcs_by_itgt_box[itgt_box] = nlist4close_srcs

        result = dict()
        result["nlist1_srcs_by_itgt_box"] = nlist1_srcs_by_itgt_box
        result["nlist3close_srcs_by_itgt_box"] = nlist3close_srcs_by_itgt_box
        result["nlist4close_srcs_by_itgt_box"] = nlist4close_srcs_by_itgt_box

        return result
