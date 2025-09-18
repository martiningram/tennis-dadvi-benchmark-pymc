# DADVI Tennis benchmark

This repository accompanies the blog post.

To get things going, please install PyMC and pymc-extras.

Then, please clone the `tennis_atp` data using:

```
git clone https://github.com/JeffSackmann/tennis_atp.git
```

After that, you should be able to run the model using:

```
python compare_mcmc_and_dadvi.py --start_year 1990
```

And you can find the results in `./fit_results/1990/`.
