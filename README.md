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


converter = create_element("nvvideoconvert", "video-converter")
# Depending on your setup, you might need to set properties on the converter to force RGBA/RGB.
self.pipeline.add(converter)

# Link the elements appropriately:
streammux.link(pgie_human_detection)
pgie_human_detection.link(custom_preprocess)
# Alternatively, if conversion is needed between elements:
pgie_human_detection.link(converter)
converter.link(custom_preprocess)



# Number of images to display
num_files = len(npy_files)

# Set up subplot grid
cols = 4  # Number of columns in the grid
rows = (num_files + cols - 1) // cols  # Calculate the required rows

fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

# Flatten axes array if necessary
axes = axes.flatten()

# Loop through files and display them
for i, file in enumerate(npy_files):
    data = np.load(os.path.join(directory, file))  # Load .npy file
    axes[i].imshow(data, cmap='gray')  # Display the image (adjust colormap as needed)
    axes[i].set_title(file)
    axes[i].axis("off")

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()]







import redis
import time

# Base manager class: holds the redis connection, thresholds, and current state.
class GlobalTrackStatusManager:
    """
    Base Global Track Status Manager.
    
    Maintains common functions for updating Redis and holds the current state.
    All state-specific logics are implemented in child classes (TentativeState, ActiveState,
    InactiveState, TerminatedState). The update() method delegates processing to the current state.
    """
    def __init__(self, global_id: str, redis_client: redis.Redis,
                 tentative_threshold: float, inactive_threshold: float):
        self.global_id = global_id
        self.redis = redis_client
        self.tentative_threshold = tentative_threshold  # seconds
        self.inactive_threshold = inactive_threshold      # seconds
        # Initialize state as tentative
        self.state = TentativeState(self)
        self.state.on_enter()

    # --- Redis update helper functions ---
    def update_state_in_redis(self, state: str):
        key = f"global:{self.global_id}"
        self.redis.hset(key, "state", state)

    def add_to_set(self, set_name: str):
        self.redis.sadd(set_name, self.global_id)

    def remove_from_set(self, set_name: str):
        self.redis.srem(set_name, self.global_id)

    def add_active_local_track(self, track_id: str):
        key = f"global_is:active_local_track_id_set:{self.global_id}"
        self.redis.sadd(key, track_id)

    def remove_active_local_track(self, track_id: str):
        key = f"global_is:active_local_track_id_set:{self.global_id}"
        self.redis.srem(key, track_id)

    def get_active_local_track_count(self) -> int:
        key = f"global_is:active_local_track_id_set:{self.global_id}"
        return self.redis.scard(key)

    def set_last_detection_time(self, ntp_time: float):
        key = f"global:{self.global_id}"
        self.redis.hset(key, "last_detection_time", ntp_time)

    def get_last_detection_time(self) -> float:
        key = f"global:{self.global_id}"
        value = self.redis.hget(key, "last_detection_time")
        if value is not None:
            return float(value.decode("utf-8"))
        return None

    # --- Update handling ---
    def update(self, message: dict):
        """
        Receives an update message and delegates processing to the current state.
        """
        self.state.update(message)

    def transition_to(self, new_state_class):
        """
        Transition to a new state and run its on_enter actions.
        """
        self.state = new_state_class(self)
        self.state.on_enter()


# Base state class: all state implementations inherit from this.
class GlobalTrackState:
    def __init__(self, manager: GlobalTrackStatusManager):
        self.manager = manager

    def update(self, message: dict):
        raise NotImplementedError("State update() must be implemented by subclasses.")

    def on_enter(self):
        # Optional hook for entering a state.
        pass


# ----- State Implementations -----

class TentativeState(GlobalTrackState):
    """
    When a global track is first created it is in tentative state.
    Depending on the detection time provided in the message, the track either transitions
    to active (if detection_time > tentative_threshold) or to terminated.
    """
    def update(self, message: dict):
        # Expecting a detection_time value to decide on state transition.
        detection_time = message.get("detection_time", 0)
        if detection_time > self.manager.tentative_threshold:
            # Transition to Active state.
            self.manager.update_state_in_redis("active")
            self.manager.remove_from_set("tentative_global_ids")
            self.manager.add_to_set("active_global_ids")
            self.manager.transition_to(ActiveState)
        else:
            # Not enough detection; terminate.
            self.manager.update_state_in_redis("terminated")
            self.manager.remove_from_set("tentative_global_ids")
            self.manager.add_to_set("terminated_global_ids")
            self.manager.transition_to(TerminatedState)

    def on_enter(self):
        print(f"[TentativeState] Global track {self.manager.global_id} entered Tentative state.")
        self.manager.update_state_in_redis("tentative")
        self.manager.add_to_set("tentative_global_ids")


class ActiveState(GlobalTrackState):
    """
    Active state: track is actively associated with one or more local tracks.
    Processes removal events, updates last detection times, and checks inactivity.
    """
    def update(self, message: dict):
        event_type = message.get("event_type")
        # For removal events from local tracks:
        if event_type == "remove":
            track_id = message.get("track_id")
            if track_id:
                self.manager.remove_active_local_track(track_id)
                print(f"[ActiveState] Removed local track {track_id} from global {self.manager.global_id}.")
                # Transition to inactive if no local tracks remain.
                if self.manager.get_active_local_track_count() == 0:
                    self.manager.update_state_in_redis("inactive")
                    self.manager.remove_from_set("active_global_ids")
                    self.manager.add_to_set("inactive_global_ids")
                    self.manager.transition_to(InactiveState)
        # Check inactive threshold: if too much time has passed, terminate the track.
        current_time = time.time()
        last_detection = self.manager.get_last_detection_time()
        if last_detection is not None and (current_time - last_detection > self.manager.inactive_threshold):
            self.manager.update_state_in_redis("terminated")
            self.manager.remove_from_set("active_global_ids")
            self.manager.add_to_set("terminated_global_ids")
            self.manager.transition_to(TerminatedState)
        # Update the last detection time if provided.
        if "ntp_time" in message:
            self.manager.set_last_detection_time(message["ntp_time"])

    def on_enter(self):
        print(f"[ActiveState] Global track {self.manager.global_id} entered Active state.")
        self.manager.update_state_in_redis("active")
        self.manager.remove_from_set("tentative_global_ids")
        self.manager.add_to_set("active_global_ids")


class InactiveState(GlobalTrackState):
    """
    Inactive state: No active local tracks are currently associated.
    A rematch event can transition the state back to active.
    """
    def update(self, message: dict):
        event_type = message.get("event_type")
        # When a rematch occurs, return to active state.
        if event_type == "match":
            self.manager.update_state_in_redis("active")
            self.manager.remove_from_set("inactive_global_ids")
            self.manager.add_to_set("active_global_ids")
            self.manager.transition_to(ActiveState)

    def on_enter(self):
        print(f"[InactiveState] Global track {self.manager.global_id} entered Inactive state.")
        self.manager.update_state_in_redis("inactive")
        self.manager.remove_from_set("active_global_ids")
        self.manager.add_to_set("inactive_global_ids")


class TerminatedState(GlobalTrackState):
    """
    Terminated state: no further updates are processed.
    """
    def update(self, message: dict):
        print(f"[TerminatedState] Global track {self.manager.global_id} is terminated; update ignored.")

    def on_enter(self):
        print(f"[TerminatedState] Global track {self.manager.global_id} entered Terminated state.")
        self.manager.update_state_in_redis("terminated")
        # Clean up: remove from other state sets and add to terminated set.
        self.manager.remove_from_set("active_global_ids")
        self.manager.remove_from_set("tentative_global_ids")
        self.manager.remove_from_set("inactive_global_ids")
        self.manager.add_to_set("terminated_global_ids")


# ----- Example Usage -----
if __name__ == "__main__":
    # Connect to Redis (adjust host/port as needed)
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    # Create a manager for a specific global track.
    global_id = "global_1234"
    tentative_threshold = 5.0    # seconds: detection_time > 5 means active
    inactive_threshold = 10.0    # seconds: inactive too long triggers termination

    manager = GlobalTrackStatusManager(global_id, redis_client, tentative_threshold, inactive_threshold)
    
    # Example: Process a tentative update with detection_time.
    # For instance, a message that comes with detection_time = 6 should trigger a transition to ActiveState.
    message_detection = {"detection_time": 6.0, "ntp_time": time.time()}
    manager.update(message_detection)
    
    # Simulate an active state removal event:
    # Here we assume a local track removal event provides "remove" and the track_id.
    message_remove = {"event_type": "remove", "track_id": "27-1", "ntp_time": time.time()}
    manager.update(message_remove)
    
    # Simulate a rematch event while in Inactive state:
    message_rematch = {"event_type": "match"}
    manager.update(message_rematch)
    
    # Simulate inactivity causing termination:
    # Provide an ntp_time value in the past to trigger the inactivity threshold.
    message_inactive = {"ntp_time": time.time() - 20}  # 20 seconds ago
    manager.update(message_inactive)

