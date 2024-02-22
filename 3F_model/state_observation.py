from base_type import *

@dataclass
class State(BaseTypeClass):
    days: int = 0 # [0, 364] / 01-01 == 0
    minutes: int = 0 # = hour * 60 + minute
    T_ra_1F: float = 0. # Temperature of room air
    T_ra_2F: float = 0.
    T_ra_3F: float = 0.
    T_ra_4F: float = 0.
    T_ra_5F: float = 0.
    T_oa: float = 0. # T of outside air
    T_oa_min: float = 0. # min T_oa from this timestep to end
    T_oa_max: float = 0. # max T_oa from this timestep to end
    CA: float = 0. # clouds
    n_HC_instant: float = 0. # head count of that time
    n_HC: float = 0. # cumulative head count
    observation_space = [16]