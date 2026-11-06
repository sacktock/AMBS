
from ambs.model_checking.labelling.Seaquest import Surface
from ambs.model_checking.labelling.Seaquest import Diver
from ambs.model_checking.labelling.Seaquest import Early_Surface
from ambs.model_checking.labelling.Seaquest import Out_Of_Oxygen
from ambs.model_checking.labelling.Seaquest import Death

prop_map = {
    "seaquest": {
        "surface": (Surface, {}),
        "diver": (Diver, {}),
        "early-surface": (Early_Surface, {}),
        "out-of-oxygen": (Out_Of_Oxygen, {}),
        "death" : (Death, {}),
    },
}

def get_property(env, name, prop_string):
    to_call = prop_map[name][prop_string]
    return to_call[0](env, **to_call[1])