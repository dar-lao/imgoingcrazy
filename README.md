# imgoingcrazy

import gi
import numpy as np
import pyds
import cv2
import ctypes
import os
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
        os.makedirs("preprocess_input", exist_ok=True)
        os.makedirs("preprocess_output", exist_ok=True)

    def chainfunc(self, pad, parent, buffer):
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            frame_number = frame_meta.frame_num
            l_obj = frame_meta.obj_meta_list

            # Retrieve the original frame
            n_frame = pyds.get_nvds_buf_surface(hash(buffer), frame_meta.batch_id)
            frame_copy = np.array(n_frame, copy=True, order='C')

            while l_obj is not None:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

                # Extract bounding box
                x, y, w, h = (
                    int(obj_meta.rect_params.left),
                    int(obj_meta.rect_params.top),
                    int(obj_meta.rect_params.width),
                    int(obj_meta.rect_params.height)
                )

                # Crop the detected object
                cropped_image = frame_copy[y:y + h, x:x + w]

                # Save input image
                input_filename = f"preprocess_input/frame_{frame_number}_obj_{obj_meta.object_id}.jpg"
                cv2.imwrite(input_filename, cropped_image)

                # Preprocess (e.g., resizing to match SGIE input shape)
                processed_image = cv2.resize(cropped_image, (128, 384))
                processed_image = processed_image.astype(np.float32) / 255.0  # Normalize to [0,1]

                # Save output image
                output_filename = f"preprocess_output/frame_{frame_number}_obj_{obj_meta.object_id}.jpg"
                cv2.imwrite(output_filename, (processed_image * 255).astype(np.uint8))

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
