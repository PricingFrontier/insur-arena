"""Declarative dataset manifest — each row = one sourced dataset (no bespoke loader).

Sourced + dedup'd via the dataset-sourcing sweep; every route is pure-Python, no-auth
(OpenML id / Rdatasets CSV / CASdatasets .rda). Categorical columns are auto-detected
(object/Categorical dtype), so entries only declare target/exposure/group/drops. Validated by
scripts/validate_datasets.py. License is fetch-only unless noted; data is never re-hosted.

This file is the single real-data catalog. Task sections below are readability groupings only; code
derives task-specific views from each entry's ``task`` field.
"""
from __future__ import annotations

from insurarena.datasets.spec import DatasetEntry
from insurarena.tasks import TaskType

_CAS = "https://github.com/dutangc/CASdatasets/raw/master/data"  # .rda mirror
_RDS = "https://vincentarelbundock.github.io/Rdatasets/csv"       # R datasets CSV mirror
# Kaggle "Prudential Life Insurance Assessment" via a faithful raw-GitHub mirror, pinned to a commit
# SHA for reproducibility (no OpenML/HF mirror exists; competition data -> fetch-only, attributed).
_PRU = ("https://raw.githubusercontent.com/AntonUBC/Prudential-Life-Insurance-Assessment/"
        "2d2c7e4e2c82766d19dbe25a4cb9c36aab070b66/data/data")

MANIFEST: list[DatasetEntry] = [
    # ===================== FREQUENCY (Poisson, exposure) =====================
    DatasetEntry(
        name="freMTPL2freq", task=TaskType.FREQ, source="OpenML", access_method="openml_id",
        locator="41214", target_col="ClaimNb", full_n_rows=678013, full_n_predictors=9,
        exposure_col="Exposure", drop_cols=("IDpol",),
        cap={"ClaimNb": 4, "Exposure": 1.0},
        license="CC0 on OpenML (fetch-only)", notes="Canonical French MTPL frequency; ~94% zero.",
    ),
    # (mtpl_shap removed — 1M *synthetic* MTPL; the benchmark carries only real portfolios.)
    DatasetEntry(
        name="ausprivauto", task=TaskType.FREQ, source="Rdatasets", access_method="url_csv",
        locator=f"{_RDS}/insuranceData/dataCar.csv", target_col="numclaims",
        full_n_rows=67856, full_n_predictors=6, exposure_col="exposure",
        drop_cols=("rownames", "X_OBSTAT_", "claimcst0", "clm"),
        license="GPL>=2 (open)", notes="Australian private auto 2004-05 (=dataCar). Also supports sev/clf.",
    ),
    DatasetEntry(
        name="mass_insurance", task=TaskType.FREQ, source="Rdatasets", access_method="url_csv",
        locator=f"{_RDS}/MASS/Insurance.csv", target_col="Claims",
        full_n_rows=64, full_n_predictors=3, exposure_col="Holders",
        drop_cols=("rownames",), license="GPL>=2 (open)",
        notes="Aggregated UK third-party motor tariff cells (64 rows). Fast frequency smoke set.",
    ),
    DatasetEntry(
        name="freMTPLfreq", task=TaskType.FREQ, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/freMTPLfreq.rda::freMTPLfreq", target_col="ClaimNb",
        full_n_rows=413169, full_n_predictors=7, exposure_col="Exposure",
        drop_cols=("PolicyID",), license="CASdatasets (fetch-only)",
        notes="French MTPL v1 (Charpentier), 413k. Reproduces founding Noll-Salzmann-Wuthrich study.",
    ),
    DatasetEntry(
        name="norauto", task=TaskType.FREQ, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/norauto.rda::norauto", target_col="NbClaim",
        full_n_rows=183999, full_n_predictors=4, exposure_col="Expo",
        drop_cols=("ClaimAmount",), license="CASdatasets (fetch-only)",
        notes="Norwegian auto, 184k. Low-dimensional Poisson.",
    ),
    DatasetEntry(
        name="beMTPL97", task=TaskType.FREQ, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/beMTPL97.rda::beMTPL97", target_col="nclaims",
        full_n_rows=163212, full_n_predictors=12, exposure_col="expo",
        drop_cols=("id", "amount", "average", "claim"),
        license="CASdatasets (fetch-only)", notes="Belgian MTPL 1997 (Henckaerts). Has geo coords.",
    ),
    DatasetEntry(
        name="beMTPL16", task=TaskType.FREQ, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/beMTPL16.rda::beMTPL16", target_col="number_of_liability_claims",
        full_n_rows=70791, full_n_predictors=15, exposure_col="exposure",
        drop_cols=("insurance_contract", "claim_value"),
        license="CASdatasets (fetch-only)", notes="Belgian MTPL 2016, 71k.",
    ),
    DatasetEntry(
        name="swautoins", task=TaskType.FREQ, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/swautoins.rda::swautoins", target_col="Claims",
        full_n_rows=2182, full_n_predictors=4, exposure_col="Insured",
        drop_cols=("Payment",), cat_cols=("Zone", "Make"),  # nominal integer codes -> declare categorical
        license="public domain",                            # (Kilometres/Bonus are genuinely ordinal)
        notes="Swedish 1977 aggregated rating cells (2182). Grouped-data edge case.",
    ),
    DatasetEntry(
        name="pg15training", task=TaskType.FREQ, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/pg15training.rda::pg15training", target_col="Numtppd",
        full_n_rows=100021, full_n_predictors=13, exposure_col="Exppdays",
        drop_cols=("PolNum", "Numtpbi", "Indtppd", "Indtpbi", "Adind"),
        license="CASdatasets (fetch-only)", notes="Pricing Game 2015 training, 100k.",
    ),
    DatasetEntry(
        name="catelematic13", task=TaskType.FREQ, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/catelematic13.rda::catelematic13", target_col="NB_Claim",
        full_n_rows=100000, full_n_predictors=49, exposure_col="Duration",
        drop_cols=("AMT_Claim",), license="CASdatasets (fetch-only)",
        notes="Canadian telematics, 100k. Driving-behaviour features.",
    ),
    DatasetEntry(
        name="sgautonb", task=TaskType.FREQ, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/sgautonb.rda::sgautonb", target_col="Clm_Count",
        full_n_rows=7483, full_n_predictors=12, exposure_col="Exp_weights",
        drop_cols=("LNWEIGHT", "LossAmt"),   # LNWEIGHT = log(exposure) -> would fail the exposure-leak guard
        license="open (Frees)", notes="Singapore auto, 7.5k.",
    ),

    # ===================== SEVERITY (Gamma, conditional on a claim) =====================
    # (workers_comp removed — OpenML 42876 states the 100k claims are "synthetically generated"
    #  (Colin Priest); excluded per the no-synthetic-data policy, verified 2026-07.)
    DatasetEntry(
        name="allstate_sev", task=TaskType.SEV, source="OpenML", access_method="openml_id",
        locator="42571", target_col="loss", full_n_rows=188318, full_n_predictors=130,
        severity_is_average=True, drop_cols=("id",),
        license="comp. data on OpenML (fetch-only)", notes="Allstate Claims Severity, 188k, 130 anon features.",
    ),
    # (medical_cost removed — semi-synthetic teaching charges; excluded per the no-synthetic-data policy.)
    DatasetEntry(
        name="danishuni", task=TaskType.SEV, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/danishuni.rda::danishuni", target_col="Loss",
        full_n_rows=2167, full_n_predictors=1, severity_is_average=True,
        license="CASdatasets (fetch-only)", notes="Danish fire losses, 2167. Classic heavy tail.",
    ),
    # The design is a single near-discrete covariate (Year, 21 levels), so the general four-predictor
    # eligibility rule excludes it. The entry remains so historical cells/loaders still resolve.
    DatasetEntry(
        name="norfire", task=TaskType.SEV, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/norfire.rda::norfire", target_col="Loss",
        full_n_rows=9181, full_n_predictors=1, severity_is_average=True,
        drop_cols=("Loss2012",), license="CASdatasets (fetch-only)", notes="Norwegian fire losses, 9181.",
    ),
    DatasetEntry(
        name="lossalae", task=TaskType.SEV, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/lossalae.rda::lossalae", target_col="Loss",
        full_n_rows=1500, full_n_predictors=1, severity_is_average=True,
        license="CASdatasets (fetch-only)", notes="Loss + ALAE, 1500.",
    ),

    # ===================== CLASSIFICATION (Bernoulli) =====================
    DatasetEntry(
        name="ausprivauto_clf", task=TaskType.CLF, source="Rdatasets", access_method="url_csv",
        locator=f"{_RDS}/insuranceData/dataCar.csv", target_col="clm",
        full_n_rows=67856, full_n_predictors=6, exposure_col="exposure",
        drop_cols=("rownames", "X_OBSTAT_", "claimcst0", "numclaims"),
        license="GPL>=2 (open)", notes="Claim occurrence (0/1) on the Australian auto book.",
    ),
    DatasetEntry(
        name="credit_g", task=TaskType.CLF, source="OpenML", access_method="openml_id",
        locator="31", target_col="class", full_n_rows=1000, full_n_predictors=20,
        positive_label="bad",
        license="UCI / CC BY 4.0", redistributable=True,
        notes="German credit (Statlog). Adjacent credit-default. Positive = bad credit.",
    ),
    DatasetEntry(
        name="give_me_some_credit", task=TaskType.CLF, source="OpenML", access_method="openml_id",
        locator="45577", target_col="SeriousDlqin2yrs", full_n_rows=150000,
        full_n_predictors=12, drop_cols=("Unnamed: 0",),
        license="comp. data on OpenML (fetch-only)", notes="Consumer credit delinquency, 150k, ~6.7% positive.",
    ),
    DatasetEntry(
        name="tic2000", task=TaskType.CLF, source="OpenML", access_method="openml_id",
        locator="46281", target_col="target", full_n_rows=9822, full_n_predictors=85,
        license="UCI / open", notes="TIC2000/CoIL caravan-insurance purchase, 9822, ~6% positive (imbalanced).",
    ),

    # ===================== PURE PREMIUM (Tweedie) — single-table loss/exposure =====================
    DatasetEntry(
        name="ausprivauto_pp", task=TaskType.PP, source="Rdatasets", access_method="url_csv",
        locator=f"{_RDS}/insuranceData/dataCar.csv", target_col="claimcst0",
        full_n_rows=67856, full_n_predictors=6, exposure_col="exposure",
        drop_cols=("rownames", "X_OBSTAT_", "numclaims", "clm"), tweedie_power=1.7,
        license="GPL>=2 (open)", notes="Pure premium = total claim cost / exposure (claimcst0 is total).",
    ),
    DatasetEntry(
        name="swautoins_pp", task=TaskType.PP, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/swautoins.rda::swautoins", target_col="Payment",
        full_n_rows=2182, full_n_predictors=4, exposure_col="Insured",
        drop_cols=("Claims",), tweedie_power=1.7,
        cat_cols=("Zone", "Make"),   # nominal integer codes -> categorical (Kilometres/Bonus ordinal)
        license="public domain", notes="Swedish 1977 pure premium (Payment is total claim amount).",
    ),
    DatasetEntry(
        name="beMTPL97_pp", task=TaskType.PP, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/beMTPL97.rda::beMTPL97", target_col="amount",
        full_n_rows=163212, full_n_predictors=12, exposure_col="expo",
        drop_cols=("id", "nclaims", "average", "claim"), tweedie_power=1.7,
        license="CASdatasets (fetch-only)", notes="Belgian MTPL pure premium (amount is total claim amount).",
    ),

    # ============ ADDED: broaden domain/geography (real, non-synthetic; verified loadable) ============
    # --- Frequency (Poisson, exposure) ---
    DatasetEntry(
        name="ohlsson", task=TaskType.FREQ, source="Rdatasets", access_method="url_csv",
        locator=f"{_RDS}/insuranceData/dataOhlsson.csv", target_col="antskad",
        full_n_rows=62474, full_n_predictors=6, exposure_col="duration",
        drop_cols=("rownames", "skadkost"), license="GPL>=2 (open)",
        notes="Swedish MOTORCYCLE (Ohlsson-Johansson), 64.5k. First non-car book. zon/mcklass/bonuskl are coded.",
    ),
    DatasetEntry(
        name="brvehins1", task=TaskType.FREQ, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/brvehins1a.rda::brvehins1a", target_col="ClaimNbTotColl",
        full_n_rows=382621, full_n_predictors=12, exposure_col="ExposTotal",
        drop_cols=("ExposFireRob", "PremTotal", "PremFireRob", "SumInsAvg", "ClaimNbRob", "ClaimNbPartColl",
                   "ClaimNbFire", "ClaimNbOther", "ClaimAmountRob", "ClaimAmountPartColl", "ClaimAmountTotColl",
                   "ClaimAmountFire", "ClaimAmountOther"),
        license="CASdatasets (fetch-only)",
        notes="BRAZILIAN vehicle (aggregated cells), 393k. Total-collision claim frequency. First S. American set.",
    ),
    DatasetEntry(
        name="euhealth", task=TaskType.FREQ, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/euhealthinsurance.rda::euhealthinsurance", target_col="num_visits",
        full_n_rows=176032, full_n_predictors=4, exposure_col="policy_years", group_col="id_anon",
        drop_cols=("num_analysis", "num_dentistry", "num_diagnostics", "num_endoscopy", "num_hospitalizations",
                   "num_mammography", "num_operations", "amt_analysis", "amt_dentistry", "amt_diagnostics",
                   "amt_endoscopy", "amt_hospitalizations", "amt_mammography", "amt_operations", "amt_visits"),
        license="CASdatasets (fetch-only)",
        notes="European HEALTH, 176k. Outpatient-visit frequency per policy-year. Members repeat -> group.",
    ),

    # --- Severity (Gamma, conditional on a claim) ---
    DatasetEntry(
        name="ausautoBI", task=TaskType.SEV, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/ausautoBI8999.rda::ausautoBI8999", target_col="AggClaim",
        full_n_rows=22036, full_n_predictors=13, severity_is_average=True,
        drop_cols=("AccDate", "ReportDate", "FinDate", "FinMth", "OpTime"),  # date serials + settlement-delay leak
        license="CASdatasets (fetch-only)",
        notes="Australian auto BODILY-INJURY closed claims (de Jong-Heller), 22k. Liability severity.",
    ),
    DatasetEntry(
        name="autoclaims_us", task=TaskType.SEV, source="Rdatasets", access_method="url_csv",
        locator=f"{_RDS}/insuranceData/AutoClaims.csv", target_col="PAID",
        full_n_rows=6773, full_n_predictors=4, severity_is_average=True,
        drop_cols=("rownames",), license="GPL>=2 (open)", notes="US auto claim severity, 6.8k.",
    ),
    DatasetEntry(
        name="autobi_us", task=TaskType.SEV, source="Rdatasets", access_method="url_csv",
        locator=f"{_RDS}/insuranceData/AutoBi.csv", target_col="LOSS",
        full_n_rows=1340, full_n_predictors=11, severity_is_average=True,
        drop_cols=("rownames", "CASENUM"), license="GPL>=2 (open)",
        notes="US auto bodily-injury severity, 1.3k (small end of range). Has NaNs -> flag-and-leave.",
    ),
    DatasetEntry(
        name="ohlsson_sev", task=TaskType.SEV, source="Rdatasets", access_method="url_csv",
        locator=f"{_RDS}/insuranceData/dataOhlsson.csv", target_col="skadkost",
        full_n_rows=670, full_n_predictors=6, claim_count_col="antskad",
        drop_cols=("rownames", "duration"), license="GPL>=2 (open)",
        notes="Swedish motorcycle severity = skadkost/antskad (total cost over claim count).",
    ),

    # --- Pure premium (Tweedie, single-table loss/exposure) ---
    DatasetEntry(
        name="ohlsson_pp", task=TaskType.PP, source="Rdatasets", access_method="url_csv",
        locator=f"{_RDS}/insuranceData/dataOhlsson.csv", target_col="skadkost",
        full_n_rows=62474, full_n_predictors=6, exposure_col="duration",
        drop_cols=("rownames", "antskad"), tweedie_power=1.7, license="GPL>=2 (open)",
        notes="Swedish motorcycle pure premium (skadkost is total claim cost).",
    ),
    DatasetEntry(
        name="euhealth_pp", task=TaskType.PP, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/euhealthinsurance.rda::euhealthinsurance", target_col="amt_visits",
        full_n_rows=176032, full_n_predictors=4, exposure_col="policy_years",
        group_col="id_anon", tweedie_power=1.7,
        drop_cols=("num_analysis", "num_dentistry", "num_diagnostics", "num_endoscopy", "num_hospitalizations",
                   "num_mammography", "num_operations", "num_visits", "amt_analysis", "amt_dentistry",
                   "amt_diagnostics", "amt_endoscopy", "amt_hospitalizations", "amt_mammography", "amt_operations"),
        license="CASdatasets (fetch-only)", notes="European health outpatient-visit pure premium per policy-year.",
    ),
    DatasetEntry(
        name="brvehins1_pp", task=TaskType.PP, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/brvehins1a.rda::brvehins1a", target_col="ClaimAmountTotColl",
        full_n_rows=382621, full_n_predictors=12, exposure_col="ExposTotal",
        drop_cols=("ExposFireRob", "PremTotal", "PremFireRob", "SumInsAvg", "ClaimNbRob", "ClaimNbPartColl",
                   "ClaimNbTotColl", "ClaimNbFire", "ClaimNbOther", "ClaimAmountRob", "ClaimAmountPartColl",
                   "ClaimAmountFire", "ClaimAmountOther"), tweedie_power=1.7,
        license="CASdatasets (fetch-only)", notes="Brazilian vehicle total-collision pure premium.",
    ),

    # --- Classification (Bernoulli) — lapse/retention + life + large Brazilian auto ---
    DatasetEntry(
        name="eudirectlapse", task=TaskType.CLF, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/eudirectlapse.rda::eudirectlapse", target_col="lapse",
        full_n_rows=23060, full_n_predictors=18,
        license="CASdatasets (fetch-only)", notes="European motor policy LAPSE/retention (insurance-native clf).",
    ),
    DatasetEntry(
        name="uslapseagent", task=TaskType.CLF, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/uslapseagent.rda::uslapseagent", target_col="surrender",
        full_n_rows=29317, full_n_predictors=9,
        drop_cols=("issue.date", "death", "other", "allcause", "termination.cause"),  # outcome leakage
        license="CASdatasets (fetch-only)", notes="US LIFE-insurance policy surrender, 29k.",
    ),
    DatasetEntry(
        name="porto_seguro", task=TaskType.CLF, source="OpenML", access_method="openml_id",
        locator="42206", target_col="target", full_n_rows=595212, full_n_predictors=50,
        drop_cols=("id",),
        license="comp. data on OpenML (fetch-only)",
        notes="Porto Seguro safe-driver (Brazil), 595k, ~3.6% positive, anonymized features. Large end of range.",
    ),
    DatasetEntry(
        name="brautocoll", task=TaskType.CLF, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/brautocoll.rda::brautocoll", target_col="Severity",
        full_n_rows=1314, full_n_predictors=5, positive_label="Fatal",
        drop_cols=("Date",), license="CASdatasets (fetch-only)",
        notes="Brazilian collision fatality (Fatal/Nonfatal), 1.3k (small end of range).",
    ),

    # --- Reserving-as-severity: predict ULTIMATE claim cost from risk factors + early development ---
    # (individual-claim reserving data; framed as Gamma severity on ultimate incurred — no new task infra.)
    DatasetEntry(
        name="freclaimdam", task=TaskType.SEV, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/freclaimset3dam9207.rda::freclaimset3dam9207", target_col="inc_Y15",
        full_n_rows=3948, full_n_predictors=6, severity_is_average=True,
        feature_cols=("NbEmployee", "NbSite", "Surface", "RiskCateg", "paid_Y0", "paid_Y1", "inc_Y0", "inc_Y1"),
        license="CASdatasets (fetch-only)",
        notes="French commercial-property DAMAGE claims, 110k. Ultimate incurred (Y15) from risk + early dev (Y0-Y1).",
    ),

    # ===================== MULTICLASS (multinomial, K>2 classes) =====================
    DatasetEntry(
        name="prudential_life", task=TaskType.MULTICLF, source="GitHub mirror (Kaggle comp.)",
        access_method="url_csv", locator=f"{_PRU}/train.csv", target_col="Response",
        full_n_rows=59381, full_n_predictors=139, drop_cols=("Id",),
        license="Kaggle competition data (fetch-only, attributed)",
        notes="Prudential LIFE-underwriting risk, 59k, 8-level ordinal Response (imbalanced). Canonical insurance multiclass.",
    ),
    DatasetEntry(
        name="pg17_coverage", task=TaskType.MULTICLF, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/pg17trainpol.rda::pg17trainpol", target_col="pol_coverage",
        full_n_rows=100000, full_n_predictors=27,
        drop_cols=("id_client", "id_vehicle", "id_policy", "id_year"),
        license="CASdatasets (fetch-only)",
        notes="Pricing Game 2017 policies, 100k. Coverage-tier choice (Mini/Median1/Median2/Maxi), 4-class.",
    ),

    # ============ ADDED: behavioural pricing — conversion, instalment propensity, market premium,
    # retention (real, non-synthetic; sourced via the 2026-07 behavioural-dataset sweep) ============
    # --- Conversion (Bernoulli) — quote/lead -> policy ---
    DatasetEntry(
        name="homesite_conv", task=TaskType.CLF, source="OpenML", access_method="openml_id",
        locator="46857", target_col="QuoteConversion_Flag",
        full_n_rows=260753, full_n_predictors=301,
        drop_cols=("QuoteNumber", "Original_Quote_Date"),  # id + raw date (temporal split out of scope)
        license="Kaggle comp. data on OpenML (fetch-only)",
        notes="Homesite QUOTE CONVERSION (US home), 260k, ~19% positive. Anonymized real quote book. "
              "-1 is a missing-value sentinel in several PersonalField/PropertyField columns.",
    ),
    # (health_lead removed 2026-07 — the OpenML 45064 mirror of the AV "Health Insurance Lead
    #  Prediction" data drops Region_Code/Reco_Policy_Cat, which carried the signal: best model
    #  AUC 0.508 and GLM log-loss worse than intercept-only -> fails the non-trivial criterion.)

    # --- Instalment propensity (multinomial) — premium payment-frequency choice ---
    DatasetEntry(
        name="fremotor_payfreq", task=TaskType.MULTICLF, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/fremotor1prem0304a.rda::fremotor1prem0304a", target_col="PayFreq",
        full_n_rows=51949, full_n_predictors=18, group_col="IDpol",
        drop_cols=("PremWindscreen", "PremDamAll", "PremFire", "PremAcc1", "PremAcc2", "PremLegal",
                   "PremTPLM", "PremTPLV", "PremServ", "PremTheft", "PremTot"),  # premiums embed instalment loading
        license="CASdatasets (fetch-only)",
        notes="French motor INSTALMENT PROPENSITY: payment frequency (Annual/Half-yearly/Quarterly/Monthly), "
              "4-class, 52k. Policies span 2003-04 -> group by IDpol.",
    ),
    DatasetEntry(
        name="pg17_payfreq", task=TaskType.MULTICLF, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/pg17trainpol.rda::pg17trainpol", target_col="pol_pay_freq",
        full_n_rows=100000, full_n_predictors=27,
        drop_cols=("id_client", "id_vehicle", "id_policy", "id_year"),
        license="CASdatasets (fetch-only)",
        notes="Pricing Game 2017 INSTALMENT PROPENSITY (Yearly/Monthly/Biannual/Quarterly), 4-class, 100k. "
              "Same table as pg17_coverage, different behavioural target.",
    ),
    DatasetEntry(
        name="pg16_payfreq", task=TaskType.MULTICLF, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/pg16trainpol.rda::pg16trainpol", target_col="PayFreq",
        full_n_rows=87228, full_n_predictors=14, group_col="PolicyID",
        drop_cols=("LicNb", "ClaimNb", "BeginDate", "EndDate", "Exposure", "VehiclCateg"),
        # LicNb id-like; ClaimNb post-dates the payment choice; dates raw serials; VehiclCateg constant
        license="CASdatasets (fetch-only)",
        notes="Pricing Game 2016 COMMERCIAL-FLEET instalment propensity (year/semester/quarter), 3-class, 87k. "
              "Policies span 2011-13 -> group by PolicyID.",
    ),

    # --- Market premium (Gamma on the real charged premium; SEV machinery, no new task infra) ---
    DatasetEntry(
        name="fremotor_prem", task=TaskType.SEV, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/fremotor1prem0304a.rda::fremotor1prem0304a", target_col="PremTot",
        full_n_rows=51949, full_n_predictors=19, severity_is_average=True, group_col="IDpol",
        drop_cols=("PremWindscreen", "PremDamAll", "PremFire", "PremAcc1", "PremAcc2", "PremLegal",
                   "PremTPLM", "PremTPLV", "PremServ", "PremTheft"),  # components of the target
        license="CASdatasets (fetch-only)",
        notes="French motor MARKET PREMIUM: predict the real charged annual premium (PremTot) from rating "
              "factors incl. PayFreq. Gamma regression via the severity machinery.",
    ),

    # --- Retention (Bernoulli) — EU unit-linked savings surrender ---
    DatasetEntry(
        name="eusavingul", task=TaskType.CLF, source="CASdatasets", access_method="casdatasets_rda",
        locator=f"{_CAS}/eusavingULnoPS.rda::eusavingULnoPS", target_col="lapse.reason",
        full_n_rows=21921, full_n_predictors=5, positive_label="Surrender",
        drop_cols=("policy.ID", "issue.date", "termination.date", "premium.frequency",
                   # macro covariates are keyed to termination.date: all in-force rows share one value,
                   # so they'd separate the classes perfectly (verified 2026-07). Panel variants
                   # (eusavingULnoPSperYr) are the right home for macro effects -> future upgrade.
                   "CPI.relvar1mth", "CPI.relvar1qtr", "CPI.relvar1yr", "CPI.relvar2yr",
                   "EUidx.relvar1mth", "EUidx.relvar1qtr", "EUidx.relvar1yr", "EUidx.relvar2yr",
                   "rate1Y.relvar1mth", "rate1Y.relvar1qtr", "rate2Y.relvar1mth", "rate2Y.relvar1qtr",
                   "rate10Y.relvar1mth", "rate10Y.relvar1qtr", "unemploy.relvar1mth", "unemploy.relvar1qtr",
                   "industry.relvar1mth", "industry.relvar1qtr", "RTV.relvar1mth", "RTV.relvar1qtr"),
        license="CASdatasets (fetch-only)",
        notes="EU UNIT-LINKED savings SURRENDER (Milhaud), 21.9k, ~36% positive. Death claims (195) fold "
              "into the negative class. premium.frequency is a redacted placeholder ('unique') -> dropped.",
    ),
]
