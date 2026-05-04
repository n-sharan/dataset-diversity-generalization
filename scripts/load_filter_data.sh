#!/bin/bash

cd ..

mkdir data
mkdir data/SDD
mkdir data/DBI

pip install kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d jessicali9530/stanford-dogs-dataset -p data/SDD
kaggle datasets download -d eward96/dog-breed-images -p data/DBI

unzip data/SDD/stanford-dogs-dataset.zip -d data/SDD
unzip data/DBI/dog-breed-images.zip -d data/DBI

rm -r data/SDD/stanford-dogs-dataset.zip
rm -r data/SDD/annotations/
rm -r data/DBI/dog-breed-images.zip

mkdir -p data/filtered/SDD && find data/SDD/images/Images -maxdepth 1 -type d | grep -i -E "Bernese_mountain_dog|Border_collie|Chihuahua|Golden_retriever|Labrador_retriever|Pug|Siberian_husky" | xargs -I {} mv "{}" data/filtered/SDD
mkdir -p data/filtered/DBI && find data/DBI -maxdepth 1 -type d | grep -i -E "Bernese_mountain_dog|Border_collie|Chihuahua|Golden_retriever|Labrador_retriever|Labrador|Pug|Siberian_husky" | xargs -I {} mv "{}" data/filtered/DBI

dir_names=("SDD" "DBI")

for d_dir in "${dir_names[@]}"; do 
    rm -r data/$d_dir
    for dir in data/filtered/$d_dir/*/; do
        new_name=$(echo $(dirname "$dir"))/$(echo $(basename "$dir") | sed -E 's/^n[0-9]+-//' | tr '[:upper:]' '[:lower:]' | sed 's:/$::')
        if [[ "$(basename "$new_name")" == "labrador" ]]; then
            new_name=$(echo $(dirname "$dir"))/"labrador_retriever"
        fi
        echo $dir .. $new_name
        if [[ "$dir" != "$new_name/" && "$dir" != "$new_name" ]]; then
            mv $dir $new_name
        fi
    done
    zip -r "data/${d_dir}subset.zip" data/filtered/$d_dir/
done

rm -r data/filtered/
