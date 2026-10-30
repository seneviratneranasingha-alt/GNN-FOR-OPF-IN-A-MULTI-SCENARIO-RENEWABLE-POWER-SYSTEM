# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 16:04:50 2026

@author: Owner
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ---- Design variable: solar farm size ----
FARM_SIZES_MW = np.arange(0, 151, 10)          # 0, 10, 20, ..., 150 MW
ROOFTOP_MW_EACH = 2.0                           # fixed
N_ROOFTOP = 29

# ---- Operating scenarios ----
LOAD_PROFILES = {
    "night":  0.55,
    "morning":0.80,
    "noon":   0.95,
    "peak":   1.10,
    "low":    0.65,
}
SOLAR_PROFILES = {
    "night":   0.00,
    "morning": 0.40,
    "noon":    0.95,
    "peak":    0.60,    # evening — sun lower
    "low":     0.20,
}

""" Data Generation Code"""

def build_base_net():
    net = pn.case30()
    pp.runpp(net)
    return net

def add_rooftop_generators(net):
    """Add one rooftop PV per non-slack bus."""
    slack_bus = net.ext_grid.bus.iloc[0]
    for bus_idx in net.bus.index:
        if bus_idx == slack_bus:
            continue
        pp.create_sgen(
            net, bus=bus_idx,
            p_mw=ROOFTOP_MW_EACH, q_mvar=0.0,
            name=f"Rooftop @ {bus_idx}", type="solar"
        )
    return net

def add_solar_farm(net, bus, p_mw, pf=0.95):
    """Add the solar farm at a chosen bus."""
    if p_mw <= 0:
        return net
    q_mvar = p_mw * np.tan(np.arccos(pf))
    pp.create_sgen(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar,
                   name=f"Solar Farm {p_mw:.0f} MW", type="solar")
    return net

def generate_multiscenario_dataset(farm_bus=12, n_seeds=20, seed=0):
    """
    Build a multi-scenario dataset by sweeping:
        - solar farm size (design)
        - load profile (operating)
        - solar profile (operating)
        - random perturbation seed
    """
    np.random.seed(seed)
    base = build_base_net()
    base_loads = base.load[["p_mw", "q_mvar"]].copy()
    base_gens  = base.gen[["p_mw", "vm_pu"]].copy()

    data_list = []
    V_MIN, V_MAX = 0.95, 1.05   # ANSI limits

    for farm_mw in FARM_SIZES_MW:
        for load_name, load_scale in LOAD_PROFILES.items():
            for solar_name, solar_scale in SOLAR_PROFILES.items():
                for s in range(n_seeds):
                    # ---- Fresh network per sample ----
                    net = pn.case30()

                    # 1. Scale loads
                    jitter_l = np.random.uniform(0.95, 1.05, size=len(net.load))
                    net.load.p_mw  = base_loads.p_mw  * load_scale * jitter_l
                    net.load.q_mvar= base_loads.q_mvar* load_scale * jitter_l

                    # 2. Scale conventional generation
                    jitter_g = np.random.uniform(0.95, 1.05, size=len(net.gen))
                    net.gen.p_mw = base_gens.p_mw * jitter_g
                    net.gen.vm_pu = base_gens.vm_pu

                    # 3. Add rooftop PV with solar-profile scaling
                    add_rooftop_generators(net)
                    net.sgen.p_mw *= solar_scale
                    net.sgen.q_mvar *= solar_scale

                    # 4. Add solar farm (design variable)
                    add_solar_farm(net, bus=farm_bus,
                                   p_mw=farm_mw * solar_scale)

                    # 5. Solve
                    try:
                        pp.runpp(net, max_iteration=30)
                    except Exception:
                        continue

                    # 6. Build graph tensors
                    data = net_to_pyg(
                        net,
                        design={"farm_mw": farm_mw},
                        operating={"load": load_scale, "solar": solar_scale}
                    )
                    data_list.append(data)

    return data_list

def net_to_pyg(net, design, operating):
    """
    Convert a solved pandapower network to a PyG Data object.

    Node features (per bus):
        [Pd, Qd, Pg_conv, Pg_ren, Qg_ren, V_set, type_slack, type_pv, type_pq,
         farm_mw, load_scale, solar_scale]
    Edge index from line + transformer topology.
    Edge features: |Y_series|.
    Labels: [V, theta, violation_flag]
    """
    n_bus = len(net.bus)
    n_gen = len(net.gen)

    # ---------- Node features ----------
    Pd = net.load.p_mw.reindex(net.bus.index, fill_value=0).values / net.sn_mva
    Qd = net.load.q_mvar.reindex(net.bus.index, fill_value=0).values / net.sn_mva
    Pg_conv = net.gen.p_mw.reindex(net.bus.index, fill_value=0).values / net.sn_mva
    V_set  = net.gen.vm_pu.reindex(net.bus.index, fill_value=1.0).values

    # Renewable totals per bus
    Pg_ren = np.zeros(n_bus)
    Qg_ren = np.zeros(n_bus)
    for _, sgen in net.sgen.iterrows():
        Pg_ren[int(sgen.bus)] += sgen.p_mw / net.sn_mva
        Qg_ren[int(sgen.bus)] += sgen.q_mvar / net.sn_mva

    # Bus-type one-hot
    type_oh = np.zeros((n_bus, 3))
    slack_buses = set(net.ext_grid.bus.values)
    pv_buses    = set(net.gen.bus.values) - slack_buses
    for i in net.bus.index:
        if i in slack_buses:
            type_oh[i, 0] = 1
        elif i in pv_buses:
            type_oh[i, 1] = 1
        else:
            type_oh[i, 2] = 1

    # Global design/operating scalars broadcast to every node
    farm_n   = design["farm_mw"] / 150.0
    load_n   = operating["load"]
    solar_n  = operating["solar"]
    globals_ = np.tile([farm_n, load_n, solar_n], (n_bus, 1))

    X = np.hstack([
        Pd[:, None], Qd[:, None], Pg_conv[:, None],
        Pg_ren[:, None], Qg_ren[:, None], V_set[:, None],
        type_oh, globals_
    ])
    X = torch.tensor(X, dtype=torch.float)

    # ---------- Edge index + edge attr ----------
    edges, edge_attr = [], []
    for _, line in net.line.iterrows():
        i, j = int(line.from_bus), int(line.to_bus)
        y = 1.0 / complex(line.r_ohm_per_km * line.length_km / net.sn_mva,
                          line.x_ohm_per_km * line.length_km / net.sn_mva) \
            if False else 1.0 / complex(0.02, 0.06)   # placeholder
        edges.append([i, j])
        edge_attr.append([abs(y)])
    for _, tr in net.trafo.iterrows():
        i, j = int(tr.hv_bus), int(tr.lv_bus)
        y = 1.0 / complex(tr.vk_percent / 100, 0.1)
        edges.append([i, j])
        edge_attr.append([abs(y)])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

    # ---------- Labels ----------
    V     = net.res_bus.vm_pu.values
    theta = np.radians(net.res_bus.va_degree.values)
    viol  = ((V < 0.95) | (V > 1.05)).astype(np.float32)

    y_V     = torch.tensor(V, dtype=torch.float)
    y_th    = torch.tensor(theta, dtype=torch.float)
    y_viol  = torch.tensor(viol, dtype=torch.float)

    return Data(
        x=X,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_V=y_V,
        y_th=y_th,
        y_viol=y_viol,
        farm_mw=torch.tensor([design["farm_mw"]], dtype=torch.float)
    )



    """
    Multi Task GNN-GCN model

    The GNN predicts three outputs per bus:

    (1) Voltage magnitude 
​
    (2) Voltage angle 

    (3) Voltage-violation probability 
    """


    
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

class MultiScenarioGNN(nn.Module):
    def __init__(self, in_dim, hidden=64, n_layers=6):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            GCNConv(hidden, hidden, improved=True, cached=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.head_V    = nn.Linear(hidden, 1)
        self.head_th   = nn.Linear(hidden, 1)
        self.head_viol = nn.Linear(hidden, 1)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        h = F.relu(self.lin_in(x))
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, ei)
            h = norm(F.relu(h)) + h_res
        return {
            "V":    self.head_V(h).squeeze(-1),
            "th":   self.head_th(h).squeeze(-1),
            "viol": self.head_viol(h).squeeze(-1),
        }
    
    
    
    
    
    """The violation head is a binary classifier, 
  so use BCE with logits. The V and θ heads use MSE."""

def multi_task_loss(out, batch, w_V=1.0, w_th=1.0, w_viol=0.3):
           loss_V   = F.mse_loss(out["V"],    batch.y_V)
           loss_th  = F.mse_loss(out["th"],   batch.y_th)
           loss_viol= F.binary_cross_entropy_with_logits(out["viol"], batch.y_viol)
           return w_V * loss_V + w_th * loss_th + w_viol * loss_viol
       
        
from sklearn.model_selection import train_test_split

print("Generating multi-scenario dataset...")
all_data = generate_multiscenario_dataset(farm_bus=12, n_seeds=20)
print(f"Total samples: {len(all_data)}")

# Stratify by farm size to ensure each split spans the design space
farm_labels = [int(d.farm_mw.item()) for d in all_data]
train, test = train_test_split(all_data, test_size=0.30,
                               random_state=42, stratify=farm_labels)
val, test   = train_test_split(test, test_size=0.50,
                               random_state=42,
                               stratify=[int(d.farm_mw.item()) for d in test])

train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader   = DataLoader(val,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test,  batch_size=32, shuffle=False)



"""Use the Trained data"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiScenarioGNN(in_dim=12, hidden=64, n_layers=6).to(device)

opt   = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)

best_val, best_state = float("inf"), None

for epoch in range(1, 20):
    # ---- train ----
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        opt.zero_grad()
        out  = model(batch)
        loss = multi_task_loss(out, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tr_loss += loss.item() * batch.num_graphs
    tr_loss /= len(train_loader.dataset)

    # ---- validate ----
    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            va_loss += multi_task_loss(model(batch), batch).item() * batch.num_graphs
    va_loss /= len(val_loader.dataset)
    sched.step(va_loss)

    if va_loss < best_val:
        best_val   = va_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}  train={tr_loss:.5e}  val={va_loss:.5e}  "
              f"lr={opt.param_groups[0]['lr']:.2e}")

model.load_state_dict(best_state)
torch.save(model.state_dict(), "multiscenario_gnn.pt")


"""Using the Trained GNN as a design tool"""

def predict_violation_curve(model, base_feat, edge_index, edge_attr,
                            load_scale, solar_scale, farm_sizes):
    """
    Return, for each farm size, the number of buses with predicted violation.
    """
    model.eval()
    curves = []
    with torch.no_grad():
        for f in farm_sizes:
            # Clone base features and update global channels (last 3)
            feat = base_feat.clone()
            feat[:, -3] = f / 150.0
            feat[:, -2] = load_scale
            feat[:, -1] = solar_scale
            data = Data(x=feat, edge_index=edge_index, edge_attr=edge_attr)
            out  = model(data)
            prob = torch.sigmoid(out["viol"])
            n_viol = (prob > 0.5).sum().item()
            curves.append({"farm_mw": f, "n_violations": n_viol,
                           "min_V": out["V"].min().item()})
    return curves

# Example: worst case — peak load, night (no solar)
base = train[0]
feat = base.x.clone()
feat[:, -3] = 0.0
feat[:, -2] = 1.10   # peak load
feat[:, -1] = 0.0    # night

curve = predict_violation_curve(
    model, feat, base.edge_index, base.edge_attr,
    load_scale=1.10, solar_scale=0.0,
    farm_sizes=np.arange(0, 151, 10)
)
for c in curve:
    print(c)
    
    
"""Evaluation on the Test set"""    
    
model.eval()
mae_V, mae_th, acc_viol = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)

        mae_V.append((out["V"]  - batch.y_V ).abs().mean().item())
        mae_th.append((out["th"] - batch.y_th).abs().mean().item())

        pred = (torch.sigmoid(out["viol"]) > 0.5).float()
        acc_viol.append((pred == batch.y_viol).float().mean().item())

print(f"MAE V       : {np.mean(mae_V):.5f} pu")
print(f"MAE θ       : {np.mean(mae_th):.5f} rad")
print(f"Violation acc: {np.mean(acc_viol)*100:.2f} %")

