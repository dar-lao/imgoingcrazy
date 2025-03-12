# imgoingcrazy

import gi
import ctypes
import numpy as np
import pyds
from gi.repository import Gst, GObject

gi.require_version("Gst", "1.0")

class CustomPreprocess(Gst.Element):
    def __init__(self):
        super(CustomPreprocess, self).__init__()
        self.srcpad = Gst.Pad.new_from_template(
            Gst.PadTemplate.new("src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, Gst.Caps.new_any())
        )
        self.sinkpad = Gst.Pad.new_from_template(
            Gst.PadTemplate.new("sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, Gst.Caps.new_any())
        )
        self.sinkpad.set_chain_function(self.chainfunc)
        self.add_pad(self.srcpad)
        self.add_pad(self.sinkpad)

    def chainfunc(self, pad, parent, buffer):
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list

            while l_obj is not None:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

                if obj_meta.unique_component_id == 1 and obj_meta.class_id == 0:
                    l_obj_user_meta = obj_meta.obj_user_meta_list
                    while l_obj_user_meta is not None:
                        try:
                            user_meta = pyds.NvDsUserMeta.cast(l_obj_user_meta.data)
                        except StopIteration:
                            break

                        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                            np_array = np.ctypeslib.as_array(ptr, shape=(3, 640, 640)).astype(np.float32)

                            # Custom Preprocessing: Normalize input tensor
                            np_array = (np_array - np.mean(np_array)) / (np.std(np_array) + 1e-6)
                            np_array = np.clip(np_array, -1, 1)

                            # Copy back processed data
                            np.copyto(ptr, np_array.astype(np.float32))
                        
                        try:
                            l_obj_user_meta = l_obj_user_meta.next
                        except StopIteration:
                            break

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.FlowReturn.OK

GObject.type_register(CustomPreprocess)
