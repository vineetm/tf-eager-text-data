# tf-eager-text-data
Handling Text data with tf.eager

### Setup
#### Step 1: Getting data
* We provide the first 100K sentences of English Wikipedia and a vocab file. Feel free to use your favorite dataset.
    ```bash
    cp data/wiki.100K.txt.zip .
    cp data/vocab.txt .
    unzip wiki.100K.txt
    ```

* Take a peek:
    ```bash
    head -2 wiki.100K.txt
   ```
   ```bash
    anarchism is a political philosophy that advocates self-governed societies based on voluntary institutions .
    these are often described as stateless societies , although several authors have defined them more specifically as institutions based on non-hierarchical or free associations .
    ```


#### Step 2: Install and Setup
We will list here instructions to set up your environment using Conda environment. You can use any other equivalent setup:
```bash
conda create -n tf-eager-text python=3.6
source activate tf-eager-text
(tf-eager-text) pip install -r requirements.txt
```


#### Step 3: Launch Jupyter!
```bash
jupyter notebook
```