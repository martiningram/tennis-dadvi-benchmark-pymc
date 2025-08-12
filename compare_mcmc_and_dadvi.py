import os
import pandas as pd
import numpy as np
from time import time
from glob import glob
import pickle
import arviz as az
import pymc as pm
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser

from pymc_extras.inference.deterministic_advi.api import fit_deterministic_advi


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--start_year", required=True, type=int)
    args = parser.parse_args()

    start_year = args.start_year

    print(f"Fitting from {start_year} onwards.")

    files = glob(
        "/Users/martin.ingram/Projects/personal/tennis/tennis_atp/atp_matches_????.csv"
    )

    data = pd.concat([pd.read_csv(x) for x in files])

    data["w_svpt_won"] = data["w_1stWon"] + data["w_2ndWon"]
    data["l_svpt_won"] = data["l_1stWon"] + data["l_2ndWon"]

    data = data.dropna(subset=["w_svpt_won", "l_svpt_won", "w_svpt", "l_svpt"])

    data["date"] = pd.to_datetime(data["tourney_date"], format="%Y%m%d")

    data = data[data["date"].dt.year >= start_year]

    data = data.sort_values("date")

    player_encoder = LabelEncoder()

    player_encoder.fit(
        np.concatenate([data["winner_name"].values, data["loser_name"].values])
    )

    data["winner_encoded"] = player_encoder.transform(data["winner_name"].values)
    data["loser_encoded"] = player_encoder.transform(data["loser_name"].values)

    server_ids = np.concatenate(
        [data["winner_encoded"].values, data["loser_encoded"].values]
    )
    returner_ids = np.concatenate(
        [data["loser_encoded"].values, data["winner_encoded"].values]
    )

    server_won = np.concatenate([data["w_svpt_won"].values, data["l_svpt_won"].values])
    server_out_of = np.concatenate([data["w_svpt"].values, data["l_svpt"].values])

    n_players = len(player_encoder.classes_)

    with pm.Model() as m:

        server_skill_sd = pm.HalfNormal("server_sd", sigma=1.0)
        returner_skill_sd = pm.HalfNormal("returner_sd", sigma=1.0)

        server_skills = pm.Normal(
            "server_skills", shape=(n_players,), mu=0.0, sigma=server_skill_sd
        )
        returner_skills = pm.Normal(
            "returner_skills", shape=(n_players,), mu=0.0, sigma=returner_skill_sd
        )

        serve_intercept = pm.Normal("serve_intercept", sigma=1.0)

        pred_logit = (
            server_skills[server_ids] - returner_skills[returner_ids] + serve_intercept
        )

        likelihood = pm.Binomial(
            "likelihood", logit_p=pred_logit, n=server_out_of, observed=server_won
        )

    # Fit DADVI
    with m:
        start_time = time()
        dadvi_result = fit_deterministic_advi()
        end_time = time()
        dadvi_runtime = end_time - start_time

    # Fit regular ADVI
    with m:
        start_time = time()
        advi_res = pm.fit(n=10000).sample(draws=1000)
        end_time = time()
        advi_default_runtime = end_time - start_time

    # Fit regular ADVI for longer
    with m:
        start_time = time()
        advi_res_longer = pm.fit(n=100000).sample(draws=1000)
        end_time = time()
        advi_longer_runtime = end_time - start_time

    # Fit NUTS
    with m:
        start_time = time()
        idata_nuts = pm.sample()
        end_time = time()
        nuts_runtime = end_time - start_time

    os.makedirs(f"./fit_results/{start_year}", exist_ok=True)

    az.to_netcdf(
        dadvi_result,
        os.path.join("./fit_results", str(start_year), "dadvi_draws.netcdf"),
    )

    az.to_netcdf(
        idata_nuts,
        os.path.join("./fit_results", str(start_year), "nuts_draws.netcdf"),
    )

    az.to_netcdf(
        advi_res,
        os.path.join("./fit_results", str(start_year), "advi_draws.netcdf"),
    )

    az.to_netcdf(
        advi_res_longer,
        os.path.join("./fit_results", str(start_year), "advi_longer_draws.netcdf"),
    )

    pickle.dump(
        player_encoder,
        open(os.path.join("./fit_results", str(start_year), "encoder.pkl"), "wb"),
    )

    runtimes = pd.Series(
        {
            "nuts": nuts_runtime,
            "dadvi": dadvi_runtime,
            "advi_default": advi_default_runtime,
            "advi_longer": advi_longer_runtime,
        }
    )

    runtimes.to_csv(os.path.join("./fit_results", str(start_year), "runtimes.csv"))
