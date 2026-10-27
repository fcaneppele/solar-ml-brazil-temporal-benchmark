from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # all paths below are relative to the repository root
import urllib.request
NE = "data/external/naturalearth"
os.makedirs(NE, exist_ok=True)
for base in ["ne_50m_admin_0_countries", "ne_50m_admin_1_states_provinces"]:
    for ext in ["shp", "shx", "dbf", "prj"]:
        fn = f"{NE}/{base}.{ext}"
        if not os.path.exists(fn):  # Natural Earth (public domain), mirrored on GitHub
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/50m_cultural/{base}.{ext}", fn)
import geopandas as gpd, matplotlib.pyplot as plt, os
from matplotlib.lines import Line2D
os.makedirs("results/figures", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "savefig.dpi": 600})
CITIES = {  # name: (lat, lon, region, Koppen, label offset)
    "Manaus": (-3.119, -60.022, "North", "Am", (0.6, 0.6)),
    "Fortaleza": (-3.732, -38.527, "Northeast", "As", (-0.8, 0.9)),
    "Recife": (-8.054, -34.881, "Northeast", "Am", (-0.8, -1.6)),
    "Brasília": (-15.794, -47.882, "Center-West", "Aw", (0.7, 0.4)),
    "Cuiabá": (-15.601, -56.097, "Center-West", "Aw", (-0.8, 0.7)),
    "Belo Horizonte": (-19.917, -43.935, "Southeast", "Cwa", (0.6, 0.4)),
    "Rio de Janeiro": (-22.907, -43.173, "Southeast", "Aw", (0.7, -1.2)),
    "São Paulo": (-23.550, -46.633, "Southeast", "Cfa", (-0.8, 0.5)),
    "Curitiba": (-25.428, -49.273, "South", "Cfb", (-0.8, -0.4)),
    "Porto Alegre": (-30.035, -51.218, "South", "Cfa", (0.8, -0.8)),
}
REGION_OF_STATE = {
    "North": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
    "Northeast": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Center-West": ["DF", "GO", "MS", "MT"],
    "Southeast": ["ES", "MG", "RJ", "SP"],
    "South": ["PR", "RS", "SC"],
}
COL = {"North": "#d9ead3", "Northeast": "#fde9c9", "Center-West": "#f4d6d6",
       "Southeast": "#d6e4f4", "South": "#e5dcf0"}
MK = {"North": "o", "Northeast": "s", "Center-West": "^", "Southeast": "D", "South": "v"}

world = gpd.read_file(f"{NE}/ne_50m_admin_0_countries.shp")
states = gpd.read_file(f"{NE}/ne_50m_admin_1_states_provinces.shp")
br = states[states.admin == "Brazil"].copy()
code = br["postal"].fillna(br["iso_3166_2"].str[-2:])
br["region"] = code.map({s: r for r, ss in REGION_OF_STATE.items() for s in ss})
assert br.region.notna().all(), br[br.region.isna()][["name", "postal"]]

fig, ax = plt.subplots(figsize=(3.45, 3.6))
world.plot(ax=ax, color="#f2f2f2", edgecolor="#bdbdbd", lw=0.3)
for r, g in br.groupby("region"):
    g.plot(ax=ax, color=COL[r], edgecolor="white", lw=0.35)
br.dissolve("region").boundary.plot(ax=ax, color="#555555", lw=0.6)
for name, (lat, lon, reg, kg, off) in CITIES.items():
    ax.plot(lon, lat, MK[reg], ms=4.2, mfc="k", mec="white", mew=0.5, zorder=5)
    ha = "right" if off[0] < 0 else "left"
    ax.annotate(f"{name} ({kg})", (lon, lat), (lon + off[0], lat + off[1]), fontsize=5.8, zorder=6, ha=ha,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
ax.set_xlim(-75, -28); ax.set_ylim(-34.5, 6)
ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
ax.tick_params(labelsize=6)
ax.axhline(-23.44, color="#888888", lw=0.5, ls="--"); ax.text(-74.5, -23.0, "Tropic of Capricorn", fontsize=5, color="#666666")
ax.axhline(0, color="#888888", lw=0.5, ls=":"); ax.text(-74.5, 0.4, "Equator", fontsize=5, color="#666666")
h = [Line2D([], [], marker=MK[r], ls="", mfc="k", mec="white", ms=4, label=r) for r in COL]
patches = [plt.Rectangle((0, 0), 1, 1, fc=COL[r], ec="#555555", lw=0.4) for r in COL]
ax.legend([(p, m) for p, m in zip(patches, h)], list(COL), fontsize=5, loc="lower left",
          title="Region", title_fontsize=5.5, frameon=True, framealpha=0.9, handlelength=1.6, borderpad=0.4, labelspacing=0.3)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("results/figures/Fig1_study_area.png", bbox_inches="tight")
fig.savefig("results/figures/Fig1_study_area.pdf", bbox_inches="tight")
print("ok")
