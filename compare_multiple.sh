for start_year in 2010 2000 1990; do
    echo "$start_year"
    python compare_mcmc_and_dadvi.py --start_year $start_year
done
