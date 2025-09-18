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
from os.path import join

from pymc_extras.inference import fit_dadvi, fit
from pymc.variational.callbacks import CheckParametersConvergence


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--start_year", required=True, type=int)
    parser.add_argument("--include_challengers", required=False, action="store_true")
    parser.add_argument("--tennis_atp_dir", required=False, default="./tennis_atp/")
    args = parser.parse_args()

    start_year = args.start_year

    print(f"Fitting from {start_year} onwards.")

    files = glob(join(args.tennis_atp_dir, "atp_matches_????.csv"))

    if len(files) == 0:
        print(f"Couldn't find any tennis CSV files with the expected pattern.")
        print(
            f"If you haven't yet done so, please download the tennis_atp data using the following command:"
        )
        print(f"git clone https://github.com/JeffSackmann/tennis_atp.git")
        print(
            f"If you cloned it somewhere else than this directory, you can use the --tennis_atp_dir argument to point the script at it."
        )
        raise ValueError()

    if args.include_challengers:
        files += glob(join(args.tennis_atp_dir, "atp_matches_qual_chall_????.csv"))

    data = pd.concat([pd.read_csv(x) for x in files])

    data["w_svpt_won"] = data["w_1stWon"] + data["w_2ndWon"]
    data["l_svpt_won"] = data["l_1stWon"] + data["l_2ndWon"]

    data = data.dropna(subset=["w_svpt_won", "l_svpt_won", "w_svpt", "l_svpt"])
    data["date"] = pd.to_datetime(data["tourney_date"], format="%Y%m%d")
    data = data[data["date"].dt.year >= start_year]

    data = data[data["tourney_level"] != "D"]

    data = data.sort_values("date")

    print(f"There are {data.shape[0]} matches left for fitting.")

    player_encoder = LabelEncoder()

    player_encoder.fit(
        np.concatenate([data["winner_name"].values, data["loser_name"].values])
    )

    print(f"There are {len(player_encoder.classes_)} players involved.")

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

    os.makedirs(f"./fit_results/{start_year}", exist_ok=True)
    pickle.dump(
        player_encoder,
        open(os.path.join("./fit_results", str(start_year), "encoder.pkl"), "wb"),
    )

    print("Fitting DADVI...")
    with m:
        start_time = time()
        idata_dadvi_pytensor = fit(method="dadvi", optimizer_method="trust-ncg")
        end_time = time()
        pytensor_runtime = end_time - start_time
    print("Done.")

    az.to_netcdf(
        idata_dadvi_pytensor,
        os.path.join("./fit_results", str(start_year), "dadvi_draws_pytensor.netcdf"),
    )

    print("Fitting ADVI for 10k steps...")
    with m:
        start_time = time()
        advi_res = pm.fit(n=10000).sample(draws=1000)
        end_time = time()
        advi_default_runtime = end_time - start_time
    print("Done.")

    az.to_netcdf(
        advi_res,
        os.path.join("./fit_results", str(start_year), "advi_draws.netcdf"),
    )

    print("Fitting ADVI for up to 100k steps using default convergence criterion...")
    with m:
        start_time = time()
        advi_res_default_convergence_crit = pm.fit(
            n=100000, callbacks=[CheckParametersConvergence()]
        ).sample(draws=1000)
        end_time = time()
        advi_default_convergence_crit_runtime = end_time - start_time
    print("Done.")

    az.to_netcdf(
        advi_res_default_convergence_crit,
        os.path.join(
            "./fit_results",
            str(start_year),
            "advi_default_convergence_crit_draws.netcdf",
        ),
    )

    print("Fitting ADVI for up to 100k steps using absolute convergence criterion...")
    with m:
        start_time = time()
        advi_res_abs_convergence_crit = pm.fit(
            n=100000, callbacks=[CheckParametersConvergence(diff="absolute")]
        ).sample(draws=1000)
        end_time = time()
        advi_abs_convergence_crit_runtime = end_time - start_time
    print("Done.")

    az.to_netcdf(
        advi_res_abs_convergence_crit,
        os.path.join(
            "./fit_results", str(start_year), "advi_abs_convergence_crit_draws.netcdf"
        ),
    )

    print("Fitting PyMC's default sampler...")
    with m:
        start_time = time()
        idata_nuts = pm.sample()
        end_time = time()
        nuts_runtime = end_time - start_time
    print("Done.")

    az.to_netcdf(
        idata_nuts,
        os.path.join("./fit_results", str(start_year), "nuts_draws.netcdf"),
    )

    runtimes = pd.Series(
        {
            "nuts": nuts_runtime,
            "advi_default": advi_default_runtime,
            "advi_default_convergence_crit": advi_default_convergence_crit_runtime,
            "advi_abs_convergence_crit": advi_abs_convergence_crit_runtime,
            "dadvi_pytensor": pytensor_runtime,
        }
    )

    runtimes.to_csv(os.path.join("./fit_results", str(start_year), "runtimes.csv"))
