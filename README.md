# DADVI Tennis benchmark

This repository accompanies the blog post.

To get things going, please install PyMC and pymc-extras. Instructions for that
are...:

* Here for PyMC: https://www.pymc.io/projects/docs/en/stable/installation.html
* And here for pymc-extras: https://www.pymc.io/projects/extras/en/stable/

Then, please clone the `tennis_atp` data using:

```
git clone https://github.com/JeffSackmann/tennis_atp.git
```

After that, you should be able to run the model using, for example:

```
python compare_mcmc_and_dadvi.py --start_year 1990
```

And you can find the results in `./fit_results/1990/`.
