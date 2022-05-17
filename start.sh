mkdir dataset
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16r_EpYQT3W5zew2T511q4DGqbLL5cXIB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16r_EpYQT3W5zew2T511q4DGqbLL5cXIB" -O dataset.zip && rm -rf ~/cookies.txt
unzip dataset.zip -d dataset
git clone https://github.com/wonbeomjang/sign-language-classification-dacon
mv sign-language-classification-dacon/* ./
rm -rf sign-language-classification-dacon
pip install wandb
python preprocess.py