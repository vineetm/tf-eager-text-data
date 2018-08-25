# tf-eager-text-data
Handling Text data with tf.eager


#### Step 1: Getting data
* We provide the first 100K sentences of English Wikipedia. Feel free to use your favorite dataset.
```bash
./get_data.sh
```

* Unzip and take a peek:
```bash
unzip text8.zip
head text8
```


#### Install and Setup
We will list here instructions to set up your environment using Conda environment. You can use any other equivalent setup:
```bash
conda create -n tf-eager-text python=3.6
source activate tf-eager-text
(tf-eager-text) pip install -r requirements.txt
```