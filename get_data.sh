#!/bin/bash
echo "Start downloading datasets"
dir="datasets"
links=(
  "https://www.dropbox.com/s/ebz9v9kdnvykzcb/wbc.mat?dl=1"
  "https://www.dropbox.com/s/ag469ssk0lmctco/lympho.mat?dl=0"
  "https://www.dropbox.com/s/iq3hjxw77gpbl7u/glass.mat?dl=0"
  "https://www.dropbox.com/s/pa26odoq6atq9vx/vowels.mat?dl=0"
  "https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=0"
  "https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=0"
  "https://www.dropbox.com/s/we6aqhb0m38i60t/musk.mat?dl=0"
  "https://www.dropbox.com/s/hckgvu9m6fs441p/satimage-2.mat?dl=0"
  "https://www.dropbox.com/s/rt9i95h9jywrtiy/letter.mat?dl=0"
  "https://www.dropbox.com/s/5kuqb387sgvwmrb/vertebral.mat?dl=0"
  "https://www.dropbox.com/s/1x8rzb4a0lia6t1/pendigits.mat?dl=0"
  "https://www.dropbox.com/s/n3wurjt8v9qi6nc/mnist.mat?dl=0"
  "https://www.dropbox.com/s/lpn4z73fico4uup/ionosphere.mat?dl=0"
  "https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=0"
  "https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=0"
)

mkdir -p $dir

for link in ${links[@]}
do
  name=$(basename $link | grep -m1 -oP '(?<=).*(?=[?])')
  dirpath="${dir}/${name}"
  if [ ! -f $dirpath ]; then
    wget -O $dirpath $link
  fi
done

echo "Finished downloading datasets"
