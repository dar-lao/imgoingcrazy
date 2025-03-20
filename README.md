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




from enum import Enum

class GlobalTrackStateType(Enum):
    TENTATIVE = "tentative"
    ACTIVE = "active"
    INACTIVE = "inactive"
    TERMINATED = "terminated"



import redis
import time
from global_track_state_enum import GlobalTrackStateType

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
        # state_type must be set by the child class from GlobalTrackStateType Enum.
        self.state_type: GlobalTrackStateType = None

    def update(self, message: dict):
        raise NotImplementedError("State update() must be implemented by subclasses.")

    def on_enter(self):
        # Update Redis with the current state's value.
        self.manager.update_state_in_redis(self.state_type.value)
        print(f"[{self.state_type.value.capitalize()}State] Global track {self.manager.global_id} entered {self.state_type.value.capitalize()} state.")


# ----- State Implementations -----

class TentativeState(GlobalTrackState):
    """
    When a global track is first created it is in tentative state.
    Depending on the detection time provided in the message, the track either transitions
    to active (if detection_time > tentative_threshold) or to terminated.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.state_type = GlobalTrackStateType.TENTATIVE

    def update(self, message: dict):
        # Expecting a detection_time value to decide on state transition.
        detection_time = message.get("detection_time", 0)
        if detection_time > self.manager.tentative_threshold:
            # Transition to Active state.
            self.manager.remove_from_set("tentative_global_ids")
            self.manager.add_to_set("active_global_ids")
            self.manager.transition_to(ActiveState)
        else:
            # Not enough detection; terminate.
            self.manager.remove_from_set("tentative_global_ids")
            self.manager.add_to_set("terminated_global_ids")
            self.manager.transition_to(TerminatedState)

    def on_enter(self):
        super().on_enter()
        self.manager.add_to_set("tentative_global_ids")


class ActiveState(GlobalTrackState):
    """
    Active state: track is actively associated with one or more local tracks.
    Processes removal events, updates last detection times, and checks inactivity.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.state_type = GlobalTrackStateType.ACTIVE

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
                    self.manager.remove_from_set("active_global_ids")
                    self.manager.add_to_set("inactive_global_ids")
                    self.manager.transition_to(InactiveState)
                    return
        # Check inactive threshold: if too much time has passed, terminate the track.
        current_time = time.time()
        last_detection = self.manager.get_last_detection_time()
        if last_detection is not None and (current_time - last_detection > self.manager.inactive_threshold):
            self.manager.remove_from_set("active_global_ids")
            self.manager.add_to_set("terminated_global_ids")
            self.manager.transition_to(TerminatedState)
            return
        # Update the last detection time if provided.
        if "ntp_time" in message:
            self.manager.set_last_detection_time(message["ntp_time"])

    def on_enter(self):
        super().on_enter()
        self.manager.remove_from_set("tentative_global_ids")
        self.manager.add_to_set("active_global_ids")


class InactiveState(GlobalTrackState):
    """
    Inactive state: No active local tracks are currently associated.
    A rematch event can transition the state back to active.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.state_type = GlobalTrackStateType.INACTIVE

    def update(self, message: dict):
        event_type = message.get("event_type")
        # When a rematch occurs, return to active state.
        if event_type == "match":
            self.manager.remove_from_set("inactive_global_ids")
            self.manager.add_to_set("active_global_ids")
            self.manager.transition_to(ActiveState)

    def on_enter(self):
        super().on_enter()
        self.manager.remove_from_set("active_global_ids")
        self.manager.add_to_set("inactive_global_ids")


class TerminatedState(GlobalTrackState):
    """
    Terminated state: no further updates are processed.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.state_type = GlobalTrackStateType.TERMINATED

    def update(self, message: dict):
        print(f"[TerminatedState] Global track {self.manager.global_id} is terminated; update ignored.")

    def on_enter(self):
        super().on_enter()
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







import time
from cachetools import TTLCache
from global_track_state_enum import GlobalTrackStateType
from redis_client import RedisClient  # Uses our redis client abstraction

# Base manager class: holds the redis client instance, thresholds, TTL caches, and current state.
class GlobalTrackStatusManager:
    """
    Base Global Track Status Manager.

    Maintains common functions for updating the data store and holds the current state.
    All state-specific logic is implemented in child classes (TentativeState, ActiveState,
    InactiveState, TerminatedState). The update() method delegates processing to the current state.
    """
    def __init__(self, global_id: str, redis_client: RedisClient,
                 tentative_threshold: float, inactive_threshold: float):
        self.global_id = global_id
        self.redis = redis_client  # An instance of RedisClient

        self.tentative_threshold = tentative_threshold  # seconds for tentative TTL
        self.inactive_threshold = inactive_threshold      # seconds for inactive TTL

        # TTLCache for tentative and inactive states. The caches store a marker value (e.g., True).
        # If the key expires, it means the time-based event should trigger.
        self.tentative_cache = TTLCache(maxsize=1000, ttl=self.tentative_threshold)
        self.inactive_cache = TTLCache(maxsize=1000, ttl=self.inactive_threshold)

        # Initialize state as Tentative.
        self.state = TentativeState(self)
        self.state.on_enter()

    # --- Data store update helper functions (using redis_client methods) ---
    def update_state_in_store(self, state: str):
        key = f"global:{self.global_id}"
        self.redis.set_global_id(key, "state", state)

    def add_to_set(self, set_name: str):
        self.redis.add_to_set(set_name, self.global_id)

    def remove_from_set(self, set_name: str):
        self.redis.remove_from_set(set_name, self.global_id)

    def add_active_local_track(self, track_id: str):
        key = f"global_is:active_local_track_id_set:{self.global_id}"
        self.redis.add_to_set(key, track_id)

    def remove_active_local_track(self, track_id: str):
        key = f"global_is:active_local_track_id_set:{self.global_id}"
        self.redis.remove_from_set(key, track_id)

    def get_active_local_track_count(self) -> int:
        key = f"global_is:active_local_track_id_set:{self.global_id}"
        return self.redis.get_set_cardinality(key)

    def set_last_detection_time(self, ntp_time: float):
        key = f"global:{self.global_id}"
        self.redis.set_global_id(key, "last_detection_time", ntp_time)

    def get_last_detection_time(self) -> float:
        key = f"global:{self.global_id}"
        value = self.redis.get_global_id(key, "last_detection_time")
        if value is not None:
            return float(value)
        return None

    # --- TTL Event Processing ---
    def process_ttl_events(self):
        """
        Checks if the TTL-based events (from the TTL caches) have expired.
        If in TentativeState and the tentative_cache entry is gone, it indicates a timeout.
        Likewise for InactiveState.
        """
        # Check for tentative state expiration.
        if isinstance(self.state, TentativeState):
            if self.global_id not in self.tentative_cache:
                print(f"[TTL] Tentative TTL expired for global track {self.global_id}. Transitioning to terminated state.")
                self.remove_from_set("tentative_global_ids")
                self.add_to_set("terminated_global_ids")
                self.transition_to(TerminatedState)
                return

        # Check for inactive state expiration.
        if isinstance(self.state, InactiveState):
            if self.global_id not in self.inactive_cache:
                print(f"[TTL] Inactive TTL expired for global track {self.global_id}. Transitioning to terminated state.")
                self.remove_from_set("inactive_global_ids")
                self.add_to_set("terminated_global_ids")
                self.transition_to(TerminatedState)
                return

    # --- Update handling ---
    def update(self, message: dict):
        """
        Receives an update message, processes any TTL events first, then delegates processing to the current state.
        """
        self.process_ttl_events()
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
        # Child classes must set state_type using GlobalTrackStateType Enum.
        self.state_type: GlobalTrackStateType = None

    def update(self, message: dict):
        raise NotImplementedError("State update() must be implemented by subclasses.")

    def on_enter(self):
        # Update the data store with the current state's value.
        self.manager.update_state_in_store(self.state_type.value)
        print(f"[{self.state_type.value.capitalize()}State] Global track {self.manager.global_id} entered {self.state_type.value.capitalize()} state.")


# ----- State Implementations -----

class TentativeState(GlobalTrackState):
    """
    Tentative state: When a global track is first created, it enters tentative state.
    A TTL is added to the tentative_cache. If a detection event (with sufficient detection_time)
    occurs before TTL expires, the track transitions to ActiveState; otherwise, TTL expiration
    triggers a transition to TerminatedState.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.state_type = GlobalTrackStateType.TENTATIVE

    def update(self, message: dict):
        # Process a detection event.
        detection_time = message.get("detection_time", 0)
        # If detection event comes in before TTL expires:
        if detection_time > self.manager.tentative_threshold:
            # Cancel the TTL by removing from tentative_cache.
            if self.manager.global_id in self.manager.tentative_cache:
                del self.manager.tentative_cache[self.manager.global_id]
            self.manager.remove_from_set("tentative_global_ids")
            self.manager.add_to_set("active_global_ids")
            self.manager.transition_to(ActiveState)
        else:
            # If detection_time insufficient, we can choose to transition immediately.
            if self.manager.global_id in self.manager.tentative_cache:
                del self.manager.tentative_cache[self.manager.global_id]
            self.manager.remove_from_set("tentative_global_ids")
            self.manager.add_to_set("terminated_global_ids")
            self.manager.transition_to(TerminatedState)

    def on_enter(self):
        super().on_enter()
        # Insert an entry into the tentative_cache. Its TTL is tentative_threshold seconds.
        self.manager.tentative_cache[self.manager.global_id] = True
        self.manager.add_to_set("tentative_global_ids")


class ActiveState(GlobalTrackState):
    """
    Active state: Track is actively associated with one or more local tracks.
    Handles add/remove events and updates the last detection time.
    Note: TTL-based events are not used in ActiveState; instead, inactivity is checked via timestamps.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.state_type = GlobalTrackStateType.ACTIVE

    def update(self, message: dict):
        event_type = message.get("event_type")
        if event_type == "remove":
            track_id = message.get("track_id")
            if track_id:
                self.manager.remove_active_local_track(track_id)
                print(f"[ActiveState] Removed local track {track_id} from global {self.manager.global_id}.")
                if self.manager.get_active_local_track_count() == 0:
                    self.manager.remove_from_set("active_global_ids")
                    self.manager.add_to_set("inactive_global_ids")
                    # When transitioning to inactive state, add a TTL entry.
                    self.manager.transition_to(InactiveState)
                    return
        elif event_type == "add":
            track_id = message.get("track_id")
            if track_id:
                self.manager.add_active_local_track(track_id)
                print(f"[ActiveState] Added local track {track_id} to global {self.manager.global_id}.")
        # Update the last detection time if provided.
        if "ntp_time" in message:
            self.manager.set_last_detection_time(message["ntp_time"])

        # In addition, check inactivity via timestamps.
        current_time = time.time()
        last_detection = self.manager.get_last_detection_time()
        if last_detection is not None and (current_time - last_detection > self.manager.inactive_threshold):
            self.manager.remove_from_set("active_global_ids")
            self.manager.add_to_set("terminated_global_ids")
            self.manager.transition_to(TerminatedState)
            return

    def on_enter(self):
        super().on_enter()
        self.manager.remove_from_set("tentative_global_ids")
        self.manager.add_to_set("active_global_ids")


class InactiveState(GlobalTrackState):
    """
    Inactive state: No active local tracks are associated.
    A TTL is added to the inactive_cache. If no rematch event occurs before TTL expires,
    the state automatically transitions to TerminatedState.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.state_type = GlobalTrackStateType.INACTIVE

    def update(self, message: dict):
        event_type = message.get("event_type")
        # Rematch event triggers transition back to ActiveState.
        if event_type == "match":
            # Cancel the TTL by removing from inactive_cache.
            if self.manager.global_id in self.manager.inactive_cache:
                del self.manager.inactive_cache[self.manager.global_id]
            self.manager.remove_from_set("inactive_global_ids")
            self.manager.add_to_set("active_global_ids")
            self.manager.transition_to(ActiveState)

    def on_enter(self):
        super().on_enter()
        self.manager.remove_from_set("active_global_ids")
        self.manager.add_to_set("inactive_global_ids")
        # Insert an entry into the inactive_cache. Its TTL is inactive_threshold seconds.
        self.manager.inactive_cache[self.manager.global_id] = True


class TerminatedState(GlobalTrackState):
    """
    Terminated state: No further updates are processed.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.state_type = GlobalTrackStateType.TERMINATED

    def update(self, message: dict):
        print(f"[TerminatedState] Global track {self.manager.global_id} is terminated; update ignored.")

    def on_enter(self):
        super().on_enter()
        self.manager.remove_from_set("active_global_ids")
        self.manager.remove_from_set("tentative_global_ids")
        self.manager.remove_from_set("inactive_global_ids")
        self.manager.add_to_set("terminated_global_ids")
        # Clean up from TTL caches if present.
        self.manager.tentative_cache.pop(self.manager.global_id, None)
        self.manager.inactive_cache.pop(self.manager.global_id, None)


# ----- Example Usage -----
if __name__ == "__main__":
    # Create an instance of RedisClient (from redis_client.py)
    redis_client = RedisClient()  # Assumes its __init__ handles connection details.
    
    # Create a manager for a specific global track.
    global_id = "global_1234"
    tentative_threshold = 5.0    # seconds: TTL for tentative state
    inactive_threshold = 10.0    # seconds: TTL for inactive state

    manager = GlobalTrackStatusManager(global_id, redis_client, tentative_threshold, inactive_threshold)
    
    # Example: Process a tentative update with detection_time.
    # A message with detection_time = 6 should trigger a transition to ActiveState (if received before TTL expires).
    message_detection = {"detection_time": 6.0, "ntp_time": time.time()}
    manager.update(message_detection)
    
    # Simulate an active state event: add a local track.
    message_add = {"event_type": "add", "track_id": "27-1", "ntp_time": time.time()}
    manager.update(message_add)
    
    # Simulate an active state removal event.
    message_remove = {"event_type": "remove", "track_id": "27-1", "ntp_time": time.time()}
    manager.update(message_remove)
    
    # When no local tracks remain, the state transitions to InactiveState.
    # Then, if no rematch occurs before the TTL expires in the inactive_cache,
    # process_ttl_events (called on the next update) will transition the state to TerminatedState.
    time.sleep(inactive_threshold + 1)  # wait for TTL to expire
    manager.update({})  # a call to update() will trigger process_ttl_events() and cause the termination




import time
from cachetools import TTLCache
from apscheduler.schedulers.background import BackgroundScheduler
from global_track_state_enum import GlobalTrackStateType
from redis_client import RedisClient  # This module encapsulates the data-store operations

# Base manager class: holds the redis client instance, thresholds, TTL caches, scheduler, and current state.
class GlobalTrackStatusManager:
    """
    Global Track Status Manager

    Manages the state of a global track by delegating logic to state-specific classes.
    Uses a scheduler (APScheduler) to automatically trigger time-based transitions:
      - In TentativeState, if no match event occurs before TTL expires, the tentative global track becomes active.
      - In InactiveState, if no rematch occurs before TTL expires, the global track is terminated.
    """
    def __init__(self, global_id: str, redis_client: RedisClient,
                 tentative_threshold: float, inactive_threshold: float):
        self.global_id = global_id
        self.redis = redis_client  # Instance of RedisClient

        self.tentative_threshold = tentative_threshold  # TTL (seconds) for tentative state
        self.inactive_threshold = inactive_threshold      # TTL (seconds) for inactive state

        # TTLCache for tentative and inactive states.
        self.tentative_cache = TTLCache(maxsize=1000, ttl=self.tentative_threshold)
        self.inactive_cache = TTLCache(maxsize=1000, ttl=self.inactive_threshold)

        # Set up the scheduler to automatically check for TTL events every second.
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.process_ttl_events, 'interval', seconds=1)
        self.scheduler.start()

        # Initially, a new global track is created as tentative.
        self.state = TentativeState(self)
        self.state.on_enter()

    # --- Data store update helper functions (via redis_client methods) ---
    def update_state_in_store(self, state: str):
        key = f"global:{self.global_id}"
        self.redis.set_global_id(key, "state", state)

    def add_to_set(self, set_name: str):
        self.redis.add_to_set(set_name, self.global_id)

    def remove_from_set(self, set_name: str):
        self.redis.remove_from_set(set_name, self.global_id)

    def add_active_local_track(self, track_id: str):
        key = f"global_is:active_local_track_id_set:{self.global_id}"
        self.redis.add_to_set(key, track_id)

    def remove_active_local_track(self, track_id: str):
        key = f"global_is:active_local_track_id_set:{self.global_id}"
        self.redis.remove_from_set(key, track_id)

    def get_active_local_track_count(self) -> int:
        key = f"global_is:active_local_track_id_set:{self.global_id}"
        return self.redis.get_set_cardinality(key)

    def set_last_detection_time(self, ntp_time: float):
        key = f"global:{self.global_id}"
        self.redis.set_global_id(key, "last_detection_time", ntp_time)

    def get_last_detection_time(self) -> float:
        key = f"global:{self.global_id}"
        value = self.redis.get_global_id(key, "last_detection_time")
        if value is not None:
            return float(value)
        return None

    # --- TTL Event Processing (automatically triggered by APScheduler) ---
    def process_ttl_events(self):
        """
        Checks TTL-based events and triggers time-based transitions.
        For a tentative state, if the TTL expires (i.e. no match event occurred), 
        the tentative global track transitions to active.
        For an inactive state, TTL expiration transitions to terminated.
        """
        if isinstance(self.state, TentativeState):
            if self.global_id not in self.tentative_cache:
                print(f"[TTL] Tentative TTL expired for global track {self.global_id}. Transitioning to active state.")
                self.remove_from_set("tentative_global_ids")
                self.add_to_set("active_global_ids")
                self.transition_to(ActiveState)
                return

        if isinstance(self.state, InactiveState):
            if self.global_id not in self.inactive_cache:
                print(f"[TTL] Inactive TTL expired for global track {self.global_id}. Transitioning to terminated state.")
                self.remove_from_set("inactive_global_ids")
                self.add_to_set("terminated_global_ids")
                self.transition_to(TerminatedState)
                return

    # --- Update handling ---
    def update(self, message: dict):
        """
        Process the update message by delegating to the current state's logic.
        TTL events are handled automatically by the scheduler.
        """
        self.state.update(message)

    def transition_to(self, new_state_class):
        """
        Transition to a new state and perform its on_enter() actions.
        """
        self.state = new_state_class(self)
        self.state.on_enter()

    def shutdown_scheduler(self):
        """
        Shut down the background scheduler. Call this when terminating the service.
        """
        self.scheduler.shutdown()


# Base state class: all state implementations inherit from this.
class GlobalTrackState:
    def __init__(self, manager: GlobalTrackStatusManager):
        self.manager = manager
        # Child classes must assign a state_type from GlobalTrackStateType.
        self.state_type: GlobalTrackStateType = None

    def update(self, message: dict):
        raise NotImplementedError("Subclasses must implement update().")

    def on_enter(self):
        self.manager.update_state_in_store(self.state_type.value)
        print(f"[{self.state_type.value.capitalize()}State] Global track {self.manager.global_id} entered {self.state_type.value.capitalize()} state.")


# ----- State Implementations -----

class TentativeState(GlobalTrackState):
    """
    Tentative state:
      - A new tentative global track is created for a new local track.
      - If a match event occurs (i.e. the tentative global track is matched with an existing global track),
        the two global tracks merge and this tentative ID is no longer used—it is marked as terminated.
      - If no match occurs and the TTL expires, the tentative global track transitions to active.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.state_type = GlobalTrackStateType.TENTATIVE

    def update(self, message: dict):
        event_type = message.get("event_type")
        if event_type == "match":
            print(f"[TentativeState] Match event received for tentative global track {self.manager.global_id}. Merging and terminating tentative ID.")
            self.manager.remove_from_set("tentative_global_ids")
            self.manager.add_to_set("terminated_global_ids")
            self.manager.transition_to(TerminatedState)
        else:
            # Optionally, if a detection event comes in with strong evidence, transition to active immediately.
            detection_time = message.get("detection_time")
            if detection_time and detection_time > self.manager.tentative_threshold:
                print(f"[TentativeState] Sufficient detection for {self.manager.global_id}. Transitioning to active state.")
                if self.manager.global_id in self.manager.tentative_cache:
                    del self.manager.tentative_cache[self.manager.global_id]
                self.manager.remove_from_set("tentative_global_ids")
                self.manager.add_to_set("active_global_ids")
                self.manager.transition_to(ActiveState)

    def on_enter(self):
        super().on_enter()
        # Insert an entry into the tentative cache so that if no match event occurs within TTL, 
        # the tentative track becomes active.
        self.manager.tentative_cache[self.manager.global_id] = True
        self.manager.add_to_set("tentative_global_ids")


class ActiveState(GlobalTrackState):
    """
    Active state:
      - The global track is actively associated with one or more local tracks.
      - Handles add and remove events.
      - Inactivity (measured by last detection time) is used to transition to inactive.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.state_type = GlobalTrackStateType.ACTIVE

    def update(self, message: dict):
        event_type = message.get("event_type")
        if event_type == "remove":
            track_id = message.get("track_id")
            if track_id:
                self.manager.remove_active_local_track(track_id)
                print(f"[ActiveState] Removed local track {track_id} from global {self.manager.global_id}.")
                if self.manager.get_active_local_track_count() == 0:
                    self.manager.remove_from_set("active_global_ids")
                    self.manager.add_to_set("inactive_global_ids")
                    self.manager.transition_to(InactiveState)
                    return
        elif event_type == "add":
            track_id = message.get("track_id")
            if track_id:
                self.manager.add_active_local_track(track_id)
                print(f"[ActiveState] Added local track {track_id} to global {self.manager.global_id}.")
        # Update last detection time.
        if "ntp_time" in message:
            self.manager.set_last_detection_time(message["ntp_time"])

        # Check inactivity based on last detection timestamp.
        current_time = time.time()
        last_detection = self.manager.get_last_detection_time()
        if last_detection and (current_time - last_detection > self.manager.inactive_threshold):
            self.manager.remove_from_set("active_global_ids")
            self.manager.add_to_set("terminated_global_ids")
            self.manager.transition_to(TerminatedState)
            return

    def on_enter(self):
        super().on_enter()
        self.manager.remove_from_set("tentative_global_ids")
        self.manager.add_to_set("active_global_ids")


class InactiveState(GlobalTrackState):
    """
    Inactive state:
      - No active local tracks are associated.
      - A TTL is used: if a rematch event (i.e. a new local track association) does not occur before TTL expires,
        the global track transitions to terminated.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.state_type = GlobalTrackStateType.INACTIVE

    def update(self, message: dict):
        event_type = message.get("event_type")
        if event_type == "match":
            print(f"[InactiveState] Rematch event received for global track {self.manager.global_id}. Transitioning back to active.")
            if self.manager.global_id in self.manager.inactive_cache:
                del self.manager.inactive_cache[self.manager.global_id]
            self.manager.remove_from_set("inactive_global_ids")
            self.manager.add_to_set("active_global_ids")
            self.manager.transition_to(ActiveState)

    def on_enter(self):
        super().on_enter()
        self.manager.remove_from_set("active_global_ids")
        self.manager.add_to_set("inactive_global_ids")
        self.manager.inactive_cache[self.manager.global_id] = True


class TerminatedState(GlobalTrackState):
    """
    Terminated state:
      - No further updates are processed.
      - The global track is considered defunct.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.state_type = GlobalTrackStateType.TERMINATED

    def update(self, message: dict):
        print(f"[TerminatedState] Global track {self.manager.global_id} is terminated; update ignored.")

    def on_enter(self):
        super().on_enter()
        self.manager.remove_from_set("active_global_ids")
        self.manager.remove_from_set("tentative_global_ids")
        self.manager.remove_from_set("inactive_global_ids")
        self.manager.add_to_set("terminated_global_ids")
        # Clean up any TTL cache entries.
        self.manager.tentative_cache.pop(self.manager.global_id, None)
        self.manager.inactive_cache.pop(self.manager.global_id, None)


# ----- Example Usage -----
if __name__ == "__main__":
    # Create an instance of RedisClient (from redis_client.py)
    redis_client = RedisClient()  # Assumes __init__ handles connection details.
    
    # Create a manager for a specific global track.
    global_id = "global_1234"
    tentative_threshold = 5.0    # seconds: TTL for tentative state
    inactive_threshold = 10.0    # seconds: TTL for inactive state

    manager = GlobalTrackStatusManager(global_id, redis_client, tentative_threshold, inactive_threshold)
    
    # Example: A new local track triggers creation of a tentative global track.
    # (No detection event is needed to create the tentative state since it is the default.)
    
    # Later, if a match event is received indicating the tentative ID merges with an existing global track:
    message_match = {"event_type": "match"}
    manager.update(message_match)
    
    # Alternatively, if no match event occurs and the TTL expires, the scheduled job will trigger the transition to active.
    
    # When done (e.g., on application shutdown), shut down the scheduler:
    # manager.shutdown_scheduler()



    class GlobalTrackRegistry:
    """
    Manages multiple GlobalTrackStatusManager instances, one per global ID.
    When a new event arrives, if the global ID doesn't exist in the registry,
    a new GlobalTrackStatusManager is created.
    """
    def __init__(self, redis_client, tentative_threshold, inactive_threshold):
        self.managers = {}  # key: global_id, value: GlobalTrackStatusManager instance
        self.redis_client = redis_client
        self.tentative_threshold = tentative_threshold
        self.inactive_threshold = inactive_threshold

    def handle_event(self, event: dict):
        # Assume the event contains a global_id field.
        global_id = event.get("global_id")
        if not global_id:
            print("Event does not contain a global_id, ignoring event.")
            return

        if global_id not in self.managers:
            # Create a new manager for this global ID.
            print(f"Creating a new GlobalTrackStatusManager for global ID: {global_id}")
            from global_track_status_manager import GlobalTrackStatusManager
            manager = GlobalTrackStatusManager(
                global_id,
                self.redis_client,
                self.tentative_threshold,
                self.inactive_threshold
            )
            self.managers[global_id] = manager

        # Dispatch the event to the corresponding manager.
        self.managers[global_id].update(event)

    def remove_manager(self, global_id: str):
        """
        Optionally remove a manager if it is terminated or no longer needed.
        """
        if global_id in self.managers:
            del self.managers[global_id]























import time
from redis_client import RedisClient
from global_track_status_manager import GlobalTrackStatusManager
from global_track_state_enum import GlobalTrackStateType

# A simple registry to manage multiple global track managers.
class GlobalTrackRegistry:
    def __init__(self, redis_client, tentative_threshold, inactive_threshold):
        self.managers = {}  # key: global_id, value: GlobalTrackStatusManager instance
        self.redis_client = redis_client
        self.tentative_threshold = tentative_threshold
        self.inactive_threshold = inactive_threshold

    def handle_event(self, event: dict):
        """
        If the event contains a 'global_id', use or create a manager for it.
        Otherwise, ignore or route appropriately.
        """
        global_id = event.get("global_id")
        if not global_id:
            print("Local event received without a global_id; skipping.")
            return

        if global_id not in self.managers:
            print(f"Creating GlobalTrackStatusManager for global_id: {global_id}")
            manager = GlobalTrackStatusManager(
                global_id,
                self.redis_client,
                self.tentative_threshold,
                self.inactive_threshold
            )
            self.managers[global_id] = manager

        print(f"Dispatching event to {global_id}: {event}")
        self.managers[global_id].update(event)

    def shutdown_all(self):
        for manager in self.managers.values():
            manager.shutdown_scheduler()

if __name__ == "__main__":
    # Create an instance of RedisClient (your implementation in redis_client.py)
    redis_client = RedisClient()

    # Set low thresholds (in seconds) for testing.
    tentative_threshold = 3.0   # seconds for tentative TTL
    inactive_threshold = 3.0    # seconds for inactive TTL

    # Create a registry instance.
    registry = GlobalTrackRegistry(redis_client, tentative_threshold, inactive_threshold)

    # Define a complete pipeline (queue) of test messages:
    messages = [
        # 1. New global track "global_new_1": creation (tentative by default) via an add event.
        {"global_id": "global_new_1", "event_type": "add", "track_id": "new_1", "ntp_time": time.time(), "detection_time": 0.5},

        # 2. A detection event for "global_new_1" (detection_time > tentative_threshold) to force transition to active.
        {"global_id": "global_new_1", "detection_time": 4.0, "ntp_time": time.time()},

        # 3. In active state, add another local track.
        {"global_id": "global_new_1", "event_type": "add", "track_id": "new_2", "ntp_time": time.time()},

        # 4. Remove one track (still active since another remains).
        {"global_id": "global_new_1", "event_type": "remove", "track_id": "new_1", "ntp_time": time.time()},

        # 5. Remove the last active track to trigger transition to inactive.
        {"global_id": "global_new_1", "event_type": "remove", "track_id": "new_2", "ntp_time": time.time()},

        # 6. (No message here) Wait for inactive TTL to expire to trigger automatic transition to terminated.

        # 7. New global track "global_match_1": creation as tentative.
        {"global_id": "global_match_1", "event_type": "add", "track_id": "match_1", "ntp_time": time.time()},

        # 8. A match event for "global_match_1" that indicates merging with an existing global track.
        {"global_id": "global_match_1", "event_type": "match"},

        # 9. A split/assoc event from global for an existing global track.
        {"global_id": "afjaoieur-arjeoarj-uouh-4314n",
         "event": {"track_id": "27-1", "event_type": "remove"}},

        # 10. A local track terminated event (TimeSeriesEvent) from a camera.
        {"global_id": "global_local_1",
         "event_type": "track",
         "cam_id": "27",
         "ntp_time": 200,
         "events": [
             {"event_type": "remove", "track id": "27-1", "class_id": 1},
             {"event_type": "remove", "track_id": "27-2", "class_id": 1}
         ]}
    ]

    # Process each message with a brief pause between them.
    for msg in messages:
        print("\n--- Processing message ---")
        print(msg)
        registry.handle_event(msg)
        time.sleep(1)

    # Wait for TTL-based transitions (inactive and tentative TTL expirations).
    print("\nWaiting for TTL-based transitions...")
    time.sleep(tentative_threshold + inactive_threshold + 1)

    # Print final state for each managed global ID.
    print("\nFinal global track states:")
    for global_id, manager in registry.managers.items():
        state = redis_client.get_global_id(f"global:{global_id}", "state")
        print(f"Global ID: {global_id}, final state: {state}")

    # Shut down the scheduler for all managers.
    registry.shutdown_all()

