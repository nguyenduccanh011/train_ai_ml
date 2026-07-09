"""SMAC v48: PyTorch GRU SEQUENCE entry model on the v47 high-trade base.

The genuinely-new architecture lever: a recurrent model sees a CAUSAL WINDOW of the
last `window` bars per symbol (vs the LightGBM snapshot). Same high-trade config as
v47_et50 (action_oracle pct0.06/mfl0.08/confirm0.07 + breadth gate + cooldown40 +
P(ENTER)>=0.5) — ONLY the entry model architecture changes. Tests whether temporal
sequence carries entry-timing signal the snapshot model can't see.

Usage:  python stock_ml/scripts/build_smac_v48_seq.py [smoke|full]
  smoke = 1 seed, 4 epochs (pipeline validation);  full = seeds 42,7,99, 20 epochs.
"""
import asyncio, copy, sys, statistics as st
sys.path.insert(0, "c:/Users/DUC CANH PC/Desktop/train_ai_ml")
import psycopg2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository, ModelComponentRepository
from stock_ml.scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE = 2459
# 2nd arg selects the entry feature channel set for the GRU: "eng" = engineered
# (entry_dyn_sec, v48) vs "raw" = raw per-bar primitives (seq_raw, v49 — the
# architecturally-correct sequence input that lets the net learn its own dynamics).
FSARG = sys.argv[2] if len(sys.argv) > 2 else "eng"
FS = {"raw": "seq_raw", "rich": "seq_rich"}.get(FSARG, "entry_dyn_sec")
BG = {"metric": "pct_above_ma50", "threshold": 0.35, "ma_win": 50, "mode": "level", "z_lookback": 60}
# forensic v52: losses cluster in the 2022 bear (-11.5 = whole net profit, wr.48, hold64).
# STRICTER breadth gate to cut bear-regime dip-buys (the dip-buy edge inverts in bears).
_REG = sys.argv[5] if len(sys.argv) > 5 else None
if _REG:
    BG = {**BG, "threshold": float(_REG)}
# 3rd arg = swing preset. "std" = v49 champion swing (pct0.06/mfl0.08, ~650tr/hold51).
# "fast" = smaller/faster zigzag (pct0.045/mfl0.05) + lower P(ENTER) cutoff -> push toward
# the user's ~1700-trade / 20-40d-hold objective on the proven GRU architecture.
SWARG = sys.argv[3] if len(sys.argv) > 3 else "std"
# 4th arg = loss-cut preset (forensic v52: exit_priority was ['signal'] only, so max_hold/
# hard_stop never fired -> losers held ~48d to -18% MAE = the entire net loss is in >30d
# holds). "cut" enforces a time-cap + hard stop to kill the slow-bleed loser tail.
CUTARG = sys.argv[4] if len(sys.argv) > 4 else "none"
COOLDOWN = 40
if SWARG == "fast":
    TGT = {"type": "action_oracle", "pct": 0.045, "min_fwd_leg": 0.05, "min_leg_bars": 2,
           "entry_min_ret_120": -0.02, "entry_confirm_pct": 0.05}
    ET_OVERRIDE = 0.40
elif SWARG == "vfast":
    # push hardest toward the ~1700-trade goal: smaller/faster swings + lower P(ENTER)
    # cutoff + SHORTER re-entry cooldown (40->15, the main per-symbol trade-count limiter).
    TGT = {"type": "action_oracle", "pct": 0.035, "min_fwd_leg": 0.04, "min_leg_bars": 1,
           "entry_min_ret_120": -0.02, "entry_confirm_pct": 0.04}
    ET_OVERRIDE = 0.30
    COOLDOWN = 15
else:
    TGT = {"type": "action_oracle", "pct": 0.06, "min_fwd_leg": 0.08, "min_leg_bars": 3,
           "entry_min_ret_120": -0.02, "entry_confirm_pct": 0.07}
    ET_OVERRIDE = None

MODE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
if MODE == "smoke":
    SEEDS = [42]
    GRU_PARAMS = {"window": 24, "hidden": 32, "layers": 1, "dropout": 0.10, "epochs": 4, "batch": 512, "lr": 1e-3}
elif MODE == "mid":
    SEEDS = [42]
    GRU_PARAMS = {"window": 24, "hidden": 32, "layers": 1, "dropout": 0.10, "epochs": 20, "batch": 512, "lr": 1e-3}
elif MODE == "deep":
    # convergence check: more epochs on the best config to rule out undertraining
    SEEDS = [42]
    GRU_PARAMS = {"window": 24, "hidden": 32, "layers": 1, "dropout": 0.10, "epochs": 40, "batch": 512, "lr": 1e-3}
elif MODE == "vdeep":
    # raw-GRU kept improving 20->40 epochs; push to find the plateau
    SEEDS = [42]
    GRU_PARAMS = {"window": 24, "hidden": 32, "layers": 1, "dropout": 0.10, "epochs": 80, "batch": 512, "lr": 1e-3}
elif MODE == "xdeep":
    # 80ep beat the champion and was still climbing; probe 120ep for the peak (seed 42 trajectory)
    SEEDS = [42]
    GRU_PARAMS = {"window": 24, "hidden": 32, "layers": 1, "dropout": 0.10, "epochs": 120, "batch": 512, "lr": 1e-3}
elif MODE == "wide":
    # bigger entry representation: more hidden capacity + longer window (see more history) ->
    # potentially better entry timing/quality (tested in the rideplus exit regime).
    SEEDS = [42]
    GRU_PARAMS = {"window": 40, "hidden": 48, "layers": 1, "dropout": 0.10, "epochs": 80, "batch": 512, "lr": 1e-3}
else:  # full: multi-seed robustness at the proven 80-epoch config (seed 42 already=+32.7)
    SEEDS = [7, 99]
    GRU_PARAMS = {"window": 24, "hidden": 32, "layers": 1, "dropout": 0.10, "epochs": 80, "batch": 512, "lr": 1e-3}

COMP_NAME = "entry_torch_gru_smac"
TMPL_NAME = {"raw": "n2_smac_v49_gruraw_et50", "rich": "n2_smac_v50_grurich_et50"}.get(
    FSARG, "n2_smac_v48_gru_et50")
ET = 0.50
if SWARG == "fast":
    TMPL_NAME = "n2_smac_v51_gruraw_fast"
    ET = ET_OVERRIDE
elif SWARG == "vfast":
    TMPL_NAME = "n2_smac_v52_gruraw_vfast"
    ET = ET_OVERRIDE
if CUTARG != "none":
    TMPL_NAME = TMPL_NAME + "_" + CUTARG
if _REG:
    TMPL_NAME = TMPL_NAME + "_reg" + str(_REG).replace("0.", "").replace(".", "")
# selective signal-exit threshold (only the high-P(EXIT) non-continuers exit early)
EXIT_THR = 0.70 if CUTARG == "rideplusg" else None


async def get_gru_component():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = ModelComponentRepository(s)
        ex = await repo.get_by_name(COMP_NAME)
        if ex:
            # refresh params (epochs/window may differ between smoke/full)
            ex.params = GRU_PARAMS
            await s.commit()
            print(f"reuse component {ex.id} (params refreshed)")
            return ex.id
        c = await repo.create(name=COMP_NAME, role="entry", algorithm="torch_gru",
                              params=GRU_PARAMS, description="PyTorch GRU sequence entry for SMAC")
        await s.commit()
        print(f"created component {c.id} {COMP_NAME}")
        return c.id


async def mk(gru_id):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(TMPL_NAME)
        if ex:
            print(f"template exists {ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE)
        eng = copy.deepcopy(base.engine_config)
        eng["entry_breadth_gate"] = BG
        eng["reentry_cooldown_bars"] = COOLDOWN
        if CUTARG == "rideplus3":
            # rideplus tuned for MORE TRADES without sacrificing the ride: shallower pullback
            # (0.03 -> skips fewer entries) + tighter structure (Donch60 -> exits the run a bit
            # sooner -> recycles capital). Per-trade quality (pf 3.19) is already > champion;
            # the only gap to PnL127 is trade-count (555 vs 1381).
            eng["entry_pullback_pct"] = 0.03
            eng["entry_pullback_window"] = 40
            eng["trailing_stop_pct"] = 0.08
            eng["trailing_activate_pct"] = 0.15
            eng["trailing_skip_above_ma"] = 10
            eng["trailing_skip_ma_slope_lb"] = 3
            eng["trailing_struct_donch_win"] = 60
            eng["trailing_struct_apply_overext"] = True
            eng["downtrend_hard_stop_pct"] = -0.08
            eng["max_hold_bars"] = 90
            eng["hard_stop_pct"] = None
            eng["exit_priority"] = ["trailing_stop", "max_hold"]
            eng["reentry_cooldown_bars"] = 4
        elif CUTARG == "rideplus2":
            # rideplus + overext SELL-into-strength (apply_overext False -> overext FIRES a sell
            # at +12% over MA20 instead of riding) -> exit parabolic winners FASTER -> shorter
            # hold -> capital recycles -> MORE trades (champion holds 29d/1381tr vs our 84d/555).
            eng["entry_pullback_pct"] = 0.045
            eng["entry_pullback_window"] = 40
            eng["trailing_stop_pct"] = 0.08
            eng["trailing_activate_pct"] = 0.15
            eng["trailing_skip_above_ma"] = 10
            eng["trailing_skip_ma_slope_lb"] = 3
            eng["trailing_struct_donch_win"] = 80
            eng["trailing_struct_apply_overext"] = False
            eng["overext_ma_window"] = 20
            eng["overext_pct"] = 0.12
            eng["overext_trail_pct"] = 0.04
            eng["downtrend_hard_stop_pct"] = -0.08
            eng["max_hold_bars"] = 90
            eng["hard_stop_pct"] = None
            eng["exit_priority"] = ["trailing_stop", "overext", "max_hold"]
            eng["reentry_cooldown_bars"] = 4
        elif CUTARG == "ridestale":
            # rideplus + STALE-EXIT (dead-money cut): exit a trade that GAINED >=5% then made no
            # new high in 18 bars (peaked-and-drifting) -> free capital from stalled winners ->
            # recycle -> MORE trades, WITHOUT cutting young/losing trades or active runners (which
            # keep making new highs and ride on). Attacks the trade-count cap w/o quality loss.
            eng["entry_pullback_pct"] = 0.045
            eng["entry_pullback_window"] = 40
            eng["trailing_stop_pct"] = 0.08
            eng["trailing_activate_pct"] = 0.15
            eng["trailing_skip_above_ma"] = 10
            eng["trailing_skip_ma_slope_lb"] = 3
            eng["trailing_struct_donch_win"] = 80
            eng["trailing_struct_apply_overext"] = True
            eng["stale_exit_bars"] = 18
            eng["stale_exit_min_gain"] = 0.05
            eng["downtrend_hard_stop_pct"] = -0.08
            eng["max_hold_bars"] = 90
            eng["hard_stop_pct"] = None
            eng["exit_priority"] = ["trailing_stop", "stale", "max_hold"]
            eng["reentry_cooldown_bars"] = 4
        elif CUTARG == "rideplusv":
            # rideplus + VOL-ADAPTIVE pullback depth (per-stock): low-vol names that rarely dip
            # 4.5% get a shallower pullback -> MORE fills -> MORE trades; high-vol keep a deep
            # pullback (quality). Smarter than the uniform-shallow pullback that failed (v69).
            eng["entry_pullback_pct"] = 0.045
            eng["entry_pullback_window"] = 40
            eng["entry_pullback_vol_scale"] = True
            eng["entry_pullback_vol_k"] = 2.0
            eng["entry_pullback_vol_window"] = 20
            eng["entry_pullback_vol_lo"] = 0.025
            eng["entry_pullback_vol_hi"] = 0.08
            eng["trailing_stop_pct"] = 0.08
            eng["trailing_activate_pct"] = 0.15
            eng["trailing_skip_above_ma"] = 10
            eng["trailing_skip_ma_slope_lb"] = 3
            eng["trailing_struct_donch_win"] = 80
            eng["trailing_struct_apply_overext"] = True
            eng["downtrend_hard_stop_pct"] = -0.08
            eng["max_hold_bars"] = 90
            eng["hard_stop_pct"] = None
            eng["exit_priority"] = ["trailing_stop", "max_hold"]
            eng["reentry_cooldown_bars"] = 4
        elif CUTARG == "rideplusg":
            # rideplus + SELECTIVE continuation-gate using the model's OWN P(EXIT): re-add a
            # signal-exit that fires ONLY when P(EXIT) is high (exit_threshold 0.7 below) = exit
            # the clear non-continuers FAST (recycle capital -> more trades), RIDE the rest.
            eng["entry_pullback_pct"] = 0.045
            eng["entry_pullback_window"] = 40
            eng["trailing_stop_pct"] = 0.08
            eng["trailing_activate_pct"] = 0.15
            eng["trailing_skip_above_ma"] = 10
            eng["trailing_skip_ma_slope_lb"] = 3
            eng["trailing_struct_donch_win"] = 80
            eng["trailing_struct_apply_overext"] = True
            eng["downtrend_hard_stop_pct"] = -0.08
            eng["max_hold_bars"] = 90
            eng["hard_stop_pct"] = None
            eng["signal_exit_enabled"] = True
            eng["exit_priority"] = ["trailing_stop", "signal", "max_hold"]
            eng["reentry_cooldown_bars"] = 4
        elif CUTARG == "rideplus":
            # ridepb + "LET WINNERS RUN" arms (capture more of the +30% MFE the entries reach):
            # trailing_skip_above_ma (don't trail while price > rising MA10 -> ride the uptrend),
            # structure-RIDE (Donchian-80, ride structure not a fixed %), wider activation +15%.
            eng["entry_pullback_pct"] = 0.045
            eng["entry_pullback_window"] = 40
            eng["trailing_stop_pct"] = 0.08
            eng["trailing_activate_pct"] = 0.15
            eng["trailing_skip_above_ma"] = 10
            eng["trailing_skip_ma_slope_lb"] = 3
            eng["trailing_struct_donch_win"] = 80
            eng["trailing_struct_apply_overext"] = True
            eng["downtrend_hard_stop_pct"] = -0.08
            eng["max_hold_bars"] = 90
            eng["hard_stop_pct"] = None
            eng["exit_priority"] = ["trailing_stop", "max_hold"]
            eng["reentry_cooldown_bars"] = 4
        elif CUTARG == "ridepbconv":
            # ridepb + CONVICTION-graded pullback depth (champion's runner-catch lever): strong
            # signals get a SHALLOW pullback (don't wait for a deep dip -> catch the runaways the
            # fixed 4.5% pullback SKIPS), weak signals a deep one. depth = base*clip(1-k*strength).
            eng["entry_pullback_pct"] = 0.045
            eng["entry_pullback_window"] = 40
            eng["entry_pullback_conv_scale"] = True
            eng["entry_pullback_conv_k"] = 0.4
            eng["entry_pullback_conv_floor"] = 0.5
            eng["entry_pullback_conv_use_combo"] = True
            eng["entry_pullback_conv_combo_w"] = [0.25, 0.2, 0.3, 0.25]
            eng["entry_pullback_conv_head_w"] = 0.5
            eng["entry_pullback_conv_vol_z"] = 1.0
            eng["entry_pullback_conv_vol_lb"] = 40
            eng["trailing_stop_pct"] = 0.10
            eng["trailing_activate_pct"] = 0.08
            eng["downtrend_hard_stop_pct"] = -0.08
            eng["max_hold_bars"] = 90
            eng["hard_stop_pct"] = None
            eng["exit_priority"] = ["trailing_stop", "max_hold"]
            eng["reentry_cooldown_bars"] = 4
        elif CUTARG == "ridepb":
            # FULL champion machinery on GRU entries: pullback-FILL (wait for a 4.5% dip below
            # the signal close -> better entry price -> lower MAE/mdd, the champion's mdd-0.179
            # edge) + ride-exit (capture MFE) + downtrend-cut + cooldown 4 (high trades). The
            # pullback is ENGINE-level so it applies to SMAC signals. Targets champion 127 PnL.
            eng["entry_pullback_pct"] = 0.045
            eng["entry_pullback_window"] = 40
            eng["trailing_stop_pct"] = 0.10
            eng["trailing_activate_pct"] = 0.08
            eng["downtrend_hard_stop_pct"] = -0.08
            eng["max_hold_bars"] = 90
            eng["hard_stop_pct"] = None
            eng["exit_priority"] = ["trailing_stop", "max_hold"]
            eng["reentry_cooldown_bars"] = 4
        elif CUTARG == "ride3":
            # ride2 (ride winners + downtrend-cut) + cooldown 4 (champion's re-entry rate ->
            # ~2x trades). Combines best entry (GRU) + best exit (ride) + high-trade re-entry,
            # targeting the champion's 1381tr/127PnL profile. ride gave 4.5%/trade; 2x trades.
            eng["trailing_stop_pct"] = 0.10
            eng["trailing_activate_pct"] = 0.08
            eng["downtrend_hard_stop_pct"] = -0.08
            eng["max_hold_bars"] = 90
            eng["hard_stop_pct"] = None
            eng["exit_priority"] = ["trailing_stop", "max_hold"]
            eng["reentry_cooldown_bars"] = 4
        elif CUTARG == "ride2":
            # BEST-OF-BOTH: ride winners (trailing, NO signal-exit which cut winners early) +
            # cut losers via downtrend_hard_stop (fires ONLY when close<SMA50 & slope<=0 & -8%
            # = the 2022-type bleeders, NO whipsaw on recoverable uptrend dips). v58 ride gave
            # PnL 28.8 but mdd 0.350 (losers rode to max_hold); this should keep the PnL, cut mdd.
            eng["trailing_stop_pct"] = 0.10
            eng["trailing_activate_pct"] = 0.08
            eng["downtrend_hard_stop_pct"] = -0.08
            eng["max_hold_bars"] = 90
            eng["hard_stop_pct"] = None
            eng["exit_priority"] = ["trailing_stop", "max_hold"]
        elif CUTARG == "champexit":
            # port the rule-champion's engine-level EXIT suite (the part SMAC threw away):
            # overext sell-into-strength (+12% over MA20) + LATE trailing (only after +27%, so
            # winners RUN) + tight overext-trail handoff. The champion does 127 PnL vs SMAC 22;
            # this tests if GRU-entry + champion-exits closes the gap. cooldown 4 -> high trades.
            eng["exit_priority"] = ["trailing_stop", "overext", "signal"]
            eng["overext_ma_window"] = 20
            eng["overext_pct"] = 0.12
            eng["overext_trail_pct"] = 0.04
            eng["trailing_activate_pct"] = 0.27
            eng["trailing_stop_pct"] = 0.08
            eng["max_hold_bars"] = 10000
            eng["reentry_cooldown_bars"] = 4
        elif CUTARG == "ride":
            # RIDE: trailing REPLACES the weak signal-exit entirely (model only ENTERs; exit
            # purely by trail-from-+8% then 10% giveback) -> let winners ride to the MFE the
            # gap-decomp showed they reach (+30%). Mirrors the rule-champion's entry+trail design.
            eng["trailing_stop_pct"] = 0.10
            eng["trailing_activate_pct"] = 0.08
            eng["max_hold_bars"] = 90
            eng["hard_stop_pct"] = None
            eng["exit_priority"] = ["trailing_stop", "max_hold"]
        elif CUTARG in ("trail", "trail12", "trail_act"):
            # gap-decomp: GRU entries reach +30% MFE but signal-exit keeps only 0.86%/trade;
            # a causal trailing stop recovers ~3%/trade (PnL 12->42 in offline test). Add the
            # trailing_stop exit (the rule-champion's main profit arm, thrown away by rules-off).
            tsp = 0.12 if CUTARG == "trail12" else 0.08
            eng["trailing_stop_pct"] = tsp
            eng["trailing_activate_pct"] = 0.10 if CUTARG == "trail_act" else None
            eng["max_hold_bars"] = 60
            eng["hard_stop_pct"] = None
            eng["exit_priority"] = ["trailing_stop", "signal", "max_hold"]
        elif CUTARG in ("cut", "cut2", "timeonly"):
            # enforce the loss-cut: add max_hold + hard_stop to exit_priority (base was
            # ['signal'] only, so the 40-bar cap never fired). Forensic: >30d holds are all
            # net-negative; immediate-bleeders reach -18% MAE.
            if CUTARG == "cut":
                eng["max_hold_bars"] = 21; eng["hard_stop_pct"] = -0.10
                eng["exit_priority"] = ["hard_stop", "signal", "max_hold"]
            elif CUTARG == "cut2":
                eng["max_hold_bars"] = 30; eng["hard_stop_pct"] = -0.12
                eng["exit_priority"] = ["hard_stop", "signal", "max_hold"]
            else:  # timeonly: isolate the time-cap effect (no stop)
                eng["max_hold_bars"] = 21; eng["hard_stop_pct"] = None
                eng["exit_priority"] = ["signal", "max_hold"]
        slots = []
        for sl in base.component_slots:
            if sl.slot_type == "entry":
                tc = dict(TGT); f = FS; mlid = gru_id  # <-- GRU entry component
            else:
                tc = copy.deepcopy(sl.target_config); f = sl.feature_set_name; mlid = sl.ml_component_id
            slots.append({"slot_type": sl.slot_type, "ml_component_id": mlid,
                          "rule_component_id": sl.rule_component_id, "feature_set_name": f,
                          "target_config": tc})
        tt = await repo.create(name=TMPL_NAME, market=base.market, strategy=base.strategy,
                               feature_set_id=base.feature_set_id, target_id=base.target_id,
                               component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
                               signal_threshold=base.signal_threshold, entry_threshold=ET, exit_threshold=EXIT_THR,
                               split_config=copy.deepcopy(base.split_config), engine_config=eng,
                               validation_config=base.validation_config, seed=42,
                               description="SMAC v48 GRU sequence entry, v47 hi-trade base",
                               hypothesis="temporal sequence carries entry-timing signal a snapshot model misses",
                               universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit()
        print(f"created template {tt.id} {TMPL_NAME}")
        return tt.id


def rd(rid):
    c = psycopg2.connect(**PG); cur = c.cursor()
    cur.execute("SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades,wr,avg_hold FROM leaderboard_runs WHERE run_id=%s", (rid,))
    r = cur.fetchone(); c.close(); return r


gru_id = asyncio.run(get_gru_component())
asyncio.run(async_engine.dispose())
tid = asyncio.run(mk(gru_id))
asyncio.run(async_engine.dispose())
print(f"MODE={MODE} template={tid} epochs={GRU_PARAMS['epochs']} window={GRU_PARAMS['window']}", flush=True)
rows = []
for sd in SEEDS:
    r = run_template_experiment(template_id=tid, seed=sd)
    row = rd(r.get("run_id")) if r.get("run_id") else None
    if row:
        rows.append(row)
        print(f"  seed={sd}: comp={row[0]:.1f} PNL={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} TR={row[4]} WR={row[5]:.2f} HOLD={row[6]:.0f}", flush=True)
if rows:
    print(f"== v48_gru: comp={st.mean([r[0] for r in rows]):.0f} PNL={st.mean([r[1] for r in rows]):.0f} "
          f"pf={st.mean([r[2] for r in rows]):.2f} TR={st.mean([r[4] for r in rows]):.0f} "
          f"mdd={st.mean([r[3] for r in rows]):.3f}")
print("== ref v47_et50 (lgbm snapshot): comp+29 PNL? pf1.94 mdd0.254 ~782tr")
print("BUILD_SMAC_V48_SEQ_DONE")
