"""Download the raw data enabled in ``config.py``.

Run from the project root: ``python analysis/01_download_data.py``.
"""

import climate_fit
import config
from utah_ejpi import download as dl


def main() -> None:
    """Run every enabled download and report what was fetched."""
    raw = config.RAW
    if config.DOWNLOAD_GEOGRAPHY:
        dl.download_geography(raw / "geography", config.STATES)
    if config.DOWNLOAD_ACS:
        tables = config.ACS_HOUSING_TABLES + config.ACS_JUSTICE_TABLES
        dl.download_acs(
            raw / "acs" / f"acs5_{config.ACS_YEAR}",
            tables,
            config.STATES,
            config.ACS_YEAR,
        )
    if config.DOWNLOAD_DECENNIAL:
        dl.download_decennial(raw / "census_2020", states=config.STATES)
    if config.DOWNLOAD_BLOCKS:
        dl.download_decennial_blocks(raw / "census_2020", states=config.STATES)
        dl.download_block_relationship(raw / "geography", states=config.STATES)
    if config.DOWNLOAD_RESSTOCK:
        for state in climate_fit.all_fit_states():
            own = state in config.STATES
            dl.download_resstock(
                raw / "resstock" / "2025_amy2018_release_1",
                state,
                config.RESSTOCK_UPGRADES if own else (0,),
                weather=own,
                timeseries_buildings=config.RESSTOCK_TIMESERIES if own else (),
            )
    if config.DOWNLOAD_COMPARISON_STATES:
        for state in config.COMPARISON_STATES:
            dl.download_resstock(
                raw / "resstock" / "2025_amy2018_release_1", state, (0,), weather=False
            )
            dl.download_eia_state_totals(
                raw / f"eia_state_totals_{state.lower()}", state
            )
    if config.DOWNLOAD_RECS:
        dl.download_recs(raw / "recs_2020")
    if config.DOWNLOAD_PRISM:
        dl.download_prism(
            raw / "climate" / f"prism_4km_tmean_daily_{config.PRISM_YEAR}",
            config.PRISM_YEAR,
        )
    if config.DOWNLOAD_EIA930:
        dl.download_eia930(raw / "eia930_2018", 2018)
    if config.DOWNLOAD_EIA_STATE_TOTALS:
        dl.download_eia_state_totals(
            raw / f"eia_state_totals_{config.STATE.lower()}", config.STATE
        )
    if config.DOWNLOAD_EIA_MONTHLY:
        dl.download_eia_monthly(
            raw / f"eia_state_totals_{config.STATE.lower()}", config.STATE
        )
    if config.DOWNLOAD_LEAD:
        dl.download_lead(raw / "lead_2022", config.STATES)
    if config.DOWNLOAD_AC_PREVALENCE:
        dl.download_ac_prevalence(raw / "ac_prevalence")


if __name__ == "__main__":
    main()
