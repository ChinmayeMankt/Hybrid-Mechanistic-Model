from hybrid_model import HybridChromatographyModel
from tdm_solver import ColumnParameters
from sma_isotherm import SMAParameters
import numpy as np

# 1. Define column parameters
column = ColumnParameters(
    length=20.0,
    diameter=2.6,
    void_fraction=0.35,
    flow_rate=5.0,
    n_grid_points=50
)

# 2. Define SMA isotherm parameters
sma = SMAParameters(
    Lambda=1200.,
    K=[0.05, 0.02],
    nu=[5., 4.],
    sigma=[50., 30.],
    n_components=2
)

# 3. Create the model (without GPR)
model = HybridChromatographyModel(column, sma)

# 4. Define simple inputs
operating_conditions = {
    "flow_rate": 5.0,
    "load_volume": 10.0,
    "gradient_slope": 0.02
}

def c_inlet_fn(t):
    # Simple inlet profile (you can improve this later)
    c = np.zeros(3)  # salt + 2 proteins
    if t < 5:
        c[1] = 5.0   # Protein 1
        c[2] = 3.0   # Protein 2
    return c

c_initial = np.zeros(3)

# 5. Run prediction (mechanistic only)
result = model.predict(
    operating_conditions=operating_conditions,
    c_inlet_fn=c_inlet_fn,
    c_initial=c_initial,
    t_end=60,
    apply_gpr=False,        # ← Start with this
    verbose=True
)

print("Mechanistic run successful!")
print("Peak summaries:", result.mechanistic_summaries)
