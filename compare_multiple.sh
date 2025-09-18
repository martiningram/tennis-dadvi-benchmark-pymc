for start_year in 1990 2000 2010; do
    echo "$start_year"
    python compare_mcmc_and_dadvi.py --start_year $start_year
done
