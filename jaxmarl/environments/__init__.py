"""
This file was adapted from the original in the process of creating the imp-act adaption of JaxMARL under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
"""
from .multi_agent_env import MultiAgentEnv, State
from .mpe import (
    SimpleMPE,
    SimpleTagMPE,
    SimpleWorldCommMPE,
    SimpleSpreadMPE,
    SimpleCryptoMPE,
    SimpleSpeakerListenerMPE,
    SimplePushMPE,
    SimpleAdversaryMPE,
    SimpleReferenceMPE,
    SimpleFacmacMPE,
    SimpleFacmacMPE3a,
    SimpleFacmacMPE6a,
    SimpleFacmacMPE9a,
)
from .smax import SMAX, HeuristicEnemySMAX, LearnedPolicyEnemySMAX
from .switch_riddle import SwitchRiddle
from .overcooked import Overcooked, overcooked_layouts
from .overcooked_v2 import OvercookedV2, overcooked_v2_layouts
from .mabrax import Ant, Humanoid, Hopper, Walker2d, HalfCheetah
from .hanabi import Hanabi
from .storm import InTheGrid, InTheGrid_2p, InTheMatrix
from .coin_game import CoinGame
from .jaxnav import JaxNav

from .roadenv_wrapper import RoadEnvironment_Wrapper