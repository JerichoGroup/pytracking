## How to Use the Object Tracker:
```zsh
from object_tracker.object_tracker import ObjectTracker
tracker = ObjectTracker(run_optical_flow=False, tracker_run_iter=3)
frame = ... 
tracker.init_bounding_box(frame, [x,y,w,h])
image, tracker_output = self.tracker.run_frame(frame)
```
### About
run_optical_flow: do we want to use optical flow and run the DiMP network every several iterations instead of for each and every frame. 

tracker_run_iter: if we use optical flow, this param tells the tracker how often to use the DiMP network (e.g., if set to 3, it will run OF for 2 frames, then DiMP, then optical flow...).

### Output
image: a raw image (should be uint8 image, color format: bgr) 

tracker_output: [top_left_x, top_left_y, box_width, box_height, was_frame_algorithmic_situation_normal, output_score]

### Run Tests + Coverage
```zsh
coverage run -m pytest ./tests/test.py
```

### patch PreciseRoIPooling for faster loading
```zsh
patch ltr/external/PreciseRoIPooling/pytorch/prroi_pool/functional.py faster_import_patch
```
