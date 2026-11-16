"""Prepare the heat-level datasets from the raw Kaggle CSV files."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

SCRAP_MATERIALS = [
    "16A",
    "25A",
    "25A1",
    "26A",
    "3A",
    "3ACh",
    "3B23",
    "Ferrous scrap 6A",
    "Iron scrap",
    "Ladle scrap",
    "Liquid scrap",
    "Scrap 3B22",
    "Scrap B23",
    "Scrap G/B 25A",
    "Scrap G/B 25AS",
    "Separation scrap",
    "Separation waste 3A",
    "Steel scrap G/B 25A",
    "B11",
    "B68",
    "Briquet BMK-60",
    "Briquets 6A",
]
SLAG_FORMERS = [
    "Dolomite",
    "In-house produced lime",
    "Lime",
    "Lime, grade 2",
    "Magma",
    "Magma-P",
    "Magnesite brick scrap",
]
OUTLIER_COLUMNS = ["MW", "power_on", "power_off", "Calc_tap_mass", "GAS_AMOUNT", "O2_AMOUNT"]


def exclude_negative_charges(charges: pd.DataFrame, amount: str) -> pd.DataFrame:
    invalid_heats = charges.loc[charges[amount] < 0, "HEATID"].unique()
    return charges.loc[~charges["HEATID"].isin(invalid_heats)]


def charge_by_material(charges: pd.DataFrame, amount: str) -> pd.DataFrame:
    return charges.pivot_table(
        index="HEATID", columns="MAT_DEC", values=amount, aggfunc="sum", fill_value=0
    )


def duration_seconds(duration: str) -> int:
    try:
        minutes, seconds = duration.split(":")
        return int(minutes) * 60 + int(seconds)
    except (AttributeError, ValueError):
        return 0


def build_combined(data_dir: Path) -> pd.DataFrame:
    baskets = pd.read_csv(
        data_dir / "basket_charged.csv",
        usecols=["HEATID", "DATETIME", "MAT_DEC", "CHARGED_AMOUNT"],
        decimal=",",
    )
    # These two late basket entries were excluded by row number in notebook 1.
    incorrect_baskets = baskets["HEATID"].eq("7F5937") & baskets["DATETIME"].eq(
        "2017-09-01 22:20:00"
    )
    baskets = exclude_negative_charges(baskets.loc[~incorrect_baskets], "CHARGED_AMOUNT")
    added = pd.read_csv(
        data_dir / "eaf_added_materials.csv",
        usecols=["HEATID", "MAT_DEC", "CHARGE_AMOUNT"],
        decimal=",",
    )
    added = exclude_negative_charges(added, "CHARGE_AMOUNT")

    charges = (
        charge_by_material(baskets, "CHARGED_AMOUNT")
        .add(charge_by_material(added, "CHARGE_AMOUNT"), fill_value=0)
        .fillna(0)
        .reset_index()
    )
    charges["Scrap"] = charges[SCRAP_MATERIALS].sum(axis=1)
    charges["OBM"] = charges["Steelmaking iron"] + charges["Hot briquet iron"]
    charges["Scrap_ratio"] = charges["Scrap"] / (charges["Scrap"] + charges["OBM"])

    temperatures = pd.read_csv(
        data_dir / "eaf_temp.csv", usecols=["HEATID", "DATETIME", "TEMP"], decimal=","
    )
    temperatures["TEMP"] = pd.to_numeric(temperatures["TEMP"], errors="coerce")
    temperatures["DATETIME"] = pd.to_datetime(
        temperatures["DATETIME"], format=TIMESTAMP_FORMAT, errors="coerce"
    )
    temperatures_by_heat = temperatures.groupby("HEATID")
    final_temperature = temperatures_by_heat["TEMP"].last()

    gas_chunks = pd.read_csv(
        data_dir / "eaf_gaslance_mat.csv",
        usecols=["HEATID", "O2_AMOUNT", "GAS_AMOUNT"],
        decimal=",",
        chunksize=500_000,
    )
    # Match the notebook's last available value in file order for each counter.
    gas_by_heat = pd.concat(
        chunk.groupby("HEATID").last() for chunk in gas_chunks
    ).groupby(level="HEATID").last()

    transformer = pd.read_csv(
        data_dir / "eaf_transformer.csv",
        usecols=["HEATID", "DURATION", "MW"],
        decimal=",",
    )
    transformer["duration_seconds"] = transformer["DURATION"].map(duration_seconds)
    transformer_by_heat = transformer.groupby("HEATID")
    energy = transformer_by_heat["MW"].sum()
    power_on = transformer_by_heat["duration_seconds"].sum() / 60

    baskets["DATETIME"] = pd.to_datetime(
        baskets["DATETIME"], format=TIMESTAMP_FORMAT, errors="coerce"
    )
    start = baskets.groupby("HEATID")["DATETIME"].min()
    end = temperatures_by_heat["DATETIME"].max()
    elapsed_minutes = (end - start).dt.total_seconds() / 60
    timings = pd.DataFrame(
        {"power_on": power_on, "power_off": elapsed_minutes - power_on}
    ).round(2)

    combined = (
        charges.merge(final_temperature, on="HEATID")
        .merge(gas_by_heat, on="HEATID")
        .join(timings, on="HEATID")
        .merge(energy, on="HEATID")
        .dropna()
    )
    add_statistical_estimate(combined)
    return combined


def add_statistical_estimate(heats: pd.DataFrame) -> None:
    """Add the notebook's specific energy estimate and calculated tap mass."""
    ferrous_charge = heats["Scrap"] + heats["OBM"]
    slag_formers = heats[SLAG_FORMERS].sum(axis=1)
    tap_mass_kg = 0.9 * ferrous_charge - 0.3125 * slag_formers
    gas_per_tonne = heats["GAS_AMOUNT"] / 0.8032 / tap_mass_kg * 1000
    oxygen_per_tonne = heats["O2_AMOUNT"] / 1.43 / tap_mass_kg * 1000

    # The historical column name is retained. Its unit is kWh/t.
    heats["stat_MW"] = (
        375
        + 400 * (ferrous_charge / tap_mass_kg - 1)
        + 80 * heats["OBM"] / tap_mass_kg
        - 350 * heats["Liquid scrap"] / tap_mass_kg
        + 1000 * slag_formers / tap_mass_kg
        + 0.3 * (heats["TEMP"] - 1600)
        + (heats["power_on"] + heats["power_off"])
        - 8 * gas_per_tonne
        - 4.3 * oxygen_per_tonne
    )
    heats["Calc_tap_mass"] = tap_mass_kg / 1000


def clean_heats(heats: pd.DataFrame) -> pd.DataFrame:
    """Apply notebook 3's sequential filters in their original order."""
    cleaned = heats.loc[heats["HEATID"] != "7F5937"]
    for column in OUTLIER_COLUMNS:
        values = cleaned[column]
        midpoint = values.median()
        width = 3 * values.std()
        cleaned = cleaned.loc[
            values.between(midpoint - width, midpoint + width, inclusive="neither")
        ]
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Raw CSV folder")
    parser.add_argument("--output-dir", type=Path, help="Output folder (default: DATA_DIR/prepared)")
    args = parser.parse_args()

    try:
        combined = build_combined(args.data_dir)
    except FileNotFoundError as error:
        parser.error(f"missing raw data file: {error.filename}")

    cleaned = clean_heats(combined)
    output_dir = args.output_dir or args.data_dir / "prepared"
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as staging_dir:
        staging = Path(staging_dir)
        combined.to_csv(staging / "df_combined.csv")
        cleaned.to_csv(staging / "df_combined_cleaned.csv")
        for name, count in [
            ("df_combined.csv", len(combined)),
            ("df_combined_cleaned.csv", len(cleaned)),
        ]:
            destination = output_dir / name
            (staging / name).replace(destination)
            print(f"Wrote {count:,} heats to {destination}")


if __name__ == "__main__":
    main()
