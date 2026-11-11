"""Export selected cortex regions and layers using the existing atlas groups."""

import os


def genSingleRegion(*args, **kwargs):
    # Import imaging dependencies only when an export is actually requested.
    if __package__:
        from .freesia_export_to_BrainRegion import genSingleRegion as export
    else:
        from Analysis.freesia_export_to_BrainRegion import genSingleRegion as export
    return export(*args, **kwargs)


def genPointsSingleRegion(*args, **kwargs):
    if __package__:
        from .freesia_export_to_BrainRegion import genPointsSingleRegion as export
    else:
        from Analysis.freesia_export_to_BrainRegion import genPointsSingleRegion as export
    return export(*args, **kwargs)


_LAYER_NAMES_5 = ("_layer_1", "_layer_2_3", "_layer_4", "_layer_5", "_layer_6")
_LAYER_NAMES_4 = ("_layer_1", "_layer_2_3", "_layer_5", "_layer_6")
# These spellings and groups match the historical atlas configuration.
_NO_LAYER_4 = frozenset((
    "FRP", "MOp", "MOs", "RSPagl", "RSPd", "RSPv", "ORBI", "ORBm",
    "ORBvl", "ACAd", "ACAv", "PL", "ILA", "Ald", "Alp", "Alv",
    "PERI", "TEa", "ECT",
))


def get_multi_region(region_color, image_root, atlas_root, Region_root,
                     Points_SingleRegion_root, get_point=False):
    for region_name, labels in region_color.items():
        if get_point:
            genPointsSingleRegion(image_root, Region_root,
                                  Points_SingleRegion_root, region_name)
        else:
            genSingleRegion(atlas_root, Region_root, region_name, labels)
        print("{} region".format(region_name))


def get_multi_layer(region_color, image_root, atlas_root, Region_root,
                    Points_SingleRegion_root, get_point=False):
    for region_name, labels in region_color.items():
        layers = _LAYER_NAMES_4 if region_name in _NO_LAYER_4 else _LAYER_NAMES_5
        for index, suffix in enumerate(layers):
            layer_name = region_name + suffix
            if get_point:
                genPointsSingleRegion(image_root, Region_root,
                                      Points_SingleRegion_root, layer_name)
            else:
                start = 2 * index
                # Layer 6 retains every remaining atlas label, as before.
                layer_labels = (labels[start:] if index == len(layers) - 1
                                else labels[start:start + 2])
                genSingleRegion(atlas_root, Region_root, layer_name, layer_labels)
            print("{} region".format(layer_name))


if __name__ == "__main__":
    if __package__ and "." in __package__:
        from ..Parameter import Data_root, Isocortex, Stats
    else:
        from Parameter import Data_root, Isocortex, Stats

    root = Data_root
    for data_name in Stats["Pred_data_list"]:
        image_root = os.path.join(root, data_name, "whole_brain_cell_counts/images")
        atlas_root = os.path.join(image_root, "..", "atlas")
        save_region_root = os.path.join(image_root, "..", "cortex_region")
        save_points_region_root = os.path.join(image_root, "..", "points_cortex_region")
        get_multi_region(Isocortex, image_root, atlas_root, save_region_root,
                         save_points_region_root, get_point=True)

        save_layer_root = os.path.join(image_root, "..", "cortex_layer")
        save_points_layer_root = os.path.join(image_root, "..", "points_cortex_layer")
        get_multi_layer(Isocortex, image_root, atlas_root, save_layer_root,
                        save_points_layer_root, get_point=True)
        print("finished {}".format(data_name))
