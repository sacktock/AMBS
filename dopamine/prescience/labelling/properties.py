from prescience.labelling.Freeway import Hit
from prescience.labelling.Death import Death
from prescience.labelling.Assault import Overheat
from prescience.labelling.Below_Reward import Below_Reward
from prescience.labelling.Bowling import No_Hit
from prescience.labelling.Bowling import No_Strike
from prescience.labelling.DoubleDunk import Out_Of_Bounds
from prescience.labelling.DoubleDunk import Shoot_Bf_Clear
from prescience.labelling.Seaquest import Early_Surface
from prescience.labelling.Seaquest import Out_Of_Oxygen
from prescience.labelling.InstantNegativeReward import Instant_Negative_Reward
from prescience.labelling.Frostbite import Freezing
from prescience.labelling.Gravitar import Fuel
from prescience.labelling.Hero import Dynamite
from prescience.labelling.KungFuMaster import Energy_Loss

prop_map = {
    "freeway": {
        "hit": (Hit, {})
    },
    "assault": {
        "overheat": (Overheat, {}),
        "death": (Death, {})
    },
    "boxing": {
        "knock-out": (Below_Reward, {'threshold': -99, 'count_pos': False}),
        "lose": (Below_Reward, {'threshold': 0}),
        "no-enemy-ko": (Below_Reward, {'threshold': 99, 'count_neg': False})
    },
    "fishingderby": {
        "lose": (Below_Reward, {'threshold': 99, 'count_neg': False})
    },
    "beamrider": {
        "death": (Death, {})
    },
    "bowling": {
        "no-hit": (No_Hit, {}),
        "no-strike": (No_Strike, {})
    },
    "frostbite": {
        "death": (Death, {}),
        "freezing": (Freezing, {})
    },
    "berzerk": {
        "death": (Death, {})
    },
    "doubledunk": {
        "out-of-bounds": (Out_Of_Bounds, {}),
        "shoot-bf-clear": (Shoot_Bf_Clear, {})
    },
    "seaquest": {
        "death": (Death, {}),
        "early-surface": (Early_Surface, {}),
        "out-of-oxygen": (Out_Of_Oxygen, {})
    },
    "enduro": {
        "crash-car": (Instant_Negative_Reward, {})
    },
    "alien": {
        "death": (Death, {})
    },
    "amidar": {
        "death": (Death, {})    
    },
    "asterix": {
        "death": (Death, {})
    },
    "asteroids": {
        "death": (Death, {})
    },
    "atlantis": {
        "death": (Death, {})
    },
    "battlezone": {
        "death": (Death, {})
    },
    "breakout": {
        "death": (Death, {})
    },
    "centipede": {
        "death": (Death, {})
    },
    "crazyclimber": {
        "death": (Death, {})
    }, 
    "demonattack": {
        "death": (Death, {})
    },
    "gopher": {
        "lose-carrot": (Death, {})
    },
    "gravitar": {
        "death": (Death, {}),
        "fuel": (Fuel, {})
    },
    "jamesbond": {
        "death": (Death, {})
    },
    "icehockey": {
        "enemy-score": (Instant_Negative_Reward, {})
    },
    "kangaroo": {
        "death": (Death, {})
    },
    "kungfumaster": {
        "death": (Death, {}),
        "energy-loss": (Energy_Loss, {})
    },
    "hero": {
        "death": (Death, {}),
        "dynamite": (Dynamite, {})
    },
    "mspacman": {
        "death": (Death, {})
    },
    "namethisgame": {
        "death": (Death, {})
    },
    "bankheist": {
        "death": (Death, {})
    }
}

def get_property(env, name, prop_string):
    to_call = prop_map[name][prop_string]
    return to_call[0](env, **to_call[1])