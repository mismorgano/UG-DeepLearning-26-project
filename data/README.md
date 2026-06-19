
## Data source

The data came from [Kaggle Multi Noises for Image Denoising](https://www.kaggle.com/datasets/goutham1208/multi-noises-for-image-denoising)

## How to use it?

Before run the script you need to [set up your API keys](https://www.kaggle.com/docs/api#authentication) because it's a *kaggle* dataset.

From the parent directory you just need to run
```bash 
python -m scripts.get_raw_data
```

It will download the zip file from the [url](https://www.kaggle.com/api/v1/datasets/download/goutham1208/multi-noises-for-image-denoising) and unzip it then put it in the `data/raw` directory.

If the [url](https://www.kaggle.com/api/v1/datasets/download/goutham1208/multi-noises-for-image-denoising) changes, then we have to modify the the global variable `data_url` from the `scripts/config.py` file to point to the new corrected address (luckily this will change).

## References

Go to the [Kaggle Multi Noises for Image Denoising](https://www.kaggle.com/datasets/goutham1208/multi-noises-for-image-denoising) page.
