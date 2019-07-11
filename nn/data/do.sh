for kmode in per_recipe per_portion per_100g; do
    echo ------$kmode
    python split.py --mode matched --kcal-mode $kmode --out-dir extracted_v1_$kmode
done