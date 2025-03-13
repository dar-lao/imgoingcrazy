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
    GST_PLUGIN_NAME = "custompreprocess"

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


# Register the custom GStreamer element
GObject.type_register(CustomPreprocess)
__gstelementfactory__ = (CustomPreprocess.GST_PLUGIN_NAME, Gst.Rank.NONE, CustomPreprocess)


# Helper function to create GStreamer elements safely
def create_element(element_type: str, element_name: str):
    element = Gst.ElementFactory.make(element_type, element_name)
    if not element:
        print(f"❌ Failed to create {element_type} ({element_name})")
    return element


class DeepStreamPipeline:
    def __init__(self):
        self.pipeline = None
        self.loop = None
        self.attach_ts = False
        self.is_live = 0
        self.initialize_pipeline()

    def initialize_pipeline(self):
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            print("❌ Unable to create Pipeline")
            return

        # Create StreamMux
        streammux = create_element("nvstreammux", "stream-muxer")
        self.pipeline.add(streammux)

        # Source Bin
        url = "file:///videos/action/250.mp4"
        source_bin = self.create_source_bin(1, url)
        if not source_bin:
            print("❌ Unable to create source bin")
            return

        self.pipeline.add(source_bin)
        sinkpad = streammux.request_pad_simple("sink_0")  # Use request_pad_simple()
        srcpad = source_bin.get_static_pad("src")
        srcpad.link(sinkpad)

        self.set_streammux_properties(streammux)

        # Primary Inference (Object Detection)
        pgie_human_detection = create_element("nvinfer", "human-pose")
        pgie_human_detection.set_property("config-file-path", "config_human_detection.txt")
        self.pipeline.add(pgie_human_detection)

        # Custom Preprocessing Element (Insert Before SGIE)
        custom_preprocess = create_element("custompreprocess", "custom-preprocess")
        if not custom_preprocess:
            print("❌ CustomPreprocess element creation failed")
            return
        self.pipeline.add(custom_preprocess)

        # Secondary Inference (Re-ID or Embeddings)
        sgie_human_reid = create_element("nvinfer", "human-reid")
        sgie_human_reid.set_property("config-file-path", "config_human_reid.txt")
        self.pipeline.add(sgie_human_reid)

        # Sink
        sink = create_element("fakesink", "fakesink")
        sink.set_property("sync", False)
        self.pipeline.add(sink)

        # Link Elements
        streammux.link(pgie_human_detection)
        pgie_human_detection.link(custom_preprocess)  # New Preprocessing Step
        custom_preprocess.link(sgie_human_reid)
        sgie_human_reid.link(sink)

        # Add Probe to Extract Features
        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.probe_func, None)

    def create_source_bin(self, index, uri):
        bin_name = f"source-bin-{index}"
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            print("❌ Unable to create source bin")
            return None

        uri_decode_bin = create_element("uridecodebin", f"uri-decode-bin-{index}")
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", self.handle_new_pad, nbin)

        nbin.add(uri_decode_bin)
        bin_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        nbin.add_pad(bin_pad)
        return nbin

    def set_streammux_properties(self, streammux):
        streammux.set_property("width", 3840)
        streammux.set_property("height", 2160)
        streammux.set_property("live-source", self.is_live)
        streammux.set_property("batch-size", 1)
        streammux.set_property("attach-sys-ts", self.attach_ts)
        streammux.set_property("sync-inputs", True)
        streammux.set_property("batched-push-timeout", 33333)

    def probe_func(self, pad, info, _data):
        return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    pipe = DeepStreamPipeline()
Gst.Element.register(None, "custompreprocess", Gst.Rank.NONE, CustomPreprocess)


# Set valid metadata for the custom element.
Gst.Element.set_static_metadata(
    CustomPreprocess,
    "Custom Preprocess Element",
    "Preprocessing",
    "A custom element for preprocessing frames before secondary inference",
    "Your Name <your.email@example.com>"
)



    class MyCustomElement(GstBase.BaseTransform):
        __gstmetadata__ = ("MyCustomElement",      # Name
                           "Transform",            # Category
                           "My Custom Element",    # Description
                           "Your Name <your.email@example.com>") # Author

        def __init__(self):
            GstBase.BaseTransform.__init__(self)



