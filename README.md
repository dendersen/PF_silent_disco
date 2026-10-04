# Silent disco estimator - borrow processes and models

## intro
how the current process runs (in parts)
* find heads - "head"
* determin if a headphone is in the picture - "presence"
* determin the color on those headsets - "color"

to use anything from this in your project, add this git repository as a submodule using the git command:
``git submodule add https://github.com/dendersen/PF_silent_disco.git ${submodule_path}``

replace `${submodule_path}` with the path you want to save the submodule to

## head
the head detector is a YOLO person model with extra training to make it find a head.
when running it uses any device (rocm, xpu, cuda, x86) in theory it supports arm though it is untested

to use the head detector in your own project, simply import this:

```py
from ${submodule_path}/src/silent_disco import resolve_device, load_person_detector, detect_people
```

* resolve_device: get a device to performe the calculations
* load_person_detector: load the model
* detect_people: run the model on an image

## presence
the presence detector os a SmallConvNet that insures that a given image indeed has a person wearing a headset in it.
The main purpose is to add an extra filter for entry, this step can be skipped for performance, but has been found to decrease the stability of the model

to use the presence detector in your own project, simply import this:

```py
from ${submodule_path}/src/silent_disco import resolve_device, load_model, predict_probabilities
```

* resolve_device: get a device to performe the calculations
* load_model: load the model
* predict_probabilities: run the model on a cropped image

## color
the color detector is a SmallConvNet that categorises the color of a headset worn by a person in a cropped image.
The model does not support no headset and that should be handled by the presence model

to use the color detector in your own project, simply import this:

```py
from ${submodule_path}/src/silent_disco import resolve_device, load_model, predict_probabilities
```

* resolve_device: get a device to performe the calculations
* load_model: load the model
* predict_probabilities: run the model on a cropped image


## all
to use the full program, but managed from outside, you can use the analyze_frame function

```py
from ${submodule_path}/src/silent_disco import resolve_device, load_model, load_person_detector, analyze_frame
```

* analyze_frame: run a full headset count suite and get values out