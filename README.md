# Social sentiment analysis

An end-to-end, local-first social media sentiment project. The training pipeline downloads **Sentiment140** from Hugging Face (`stanfordnlp/sentiment140`), cleans and deduplicates the tweets, maps the original labels to binary sentiment, tokenizes with the BERT tokenizer, and fine-tunes `bert-base-uncased`. A FastAPI service exposes single and batch predictions and an in-memory trend feed; the React dashboard displays sentiment mix and recent predictions.

Sentiment140 contains roughly 1.6 million English tweets with noisy distant-supervision labels (0 = negative, 4 = positive). The labels are encoded to 0/1. Since the source labels are generated from emoticons, they should not be treated as human-verified ground truth. Dataset card: [Hugging Face: stanfordnlp/sentiment140](https://huggingface.co/datasets/stanfordnlp/sentiment140).

## Run locally

Requirements: Docker Desktop / Docker Engine with Compose. The first training run needs internet access to download the dataset and pretrained BERT weights. Training the full dataset can take hours without a GPU.

```sh
cp .env.example .env
docker compose up --build
```

The API is at http://localhost:8000, interactive API docs at http://localhost:8000/docs, and dashboard at http://localhost:5173. On first start, the API trains the model if `artifacts/model` is absent. For a quicker smoke/demo run, set `TRAIN_SAMPLE_SIZE=20000` in `.env` (the model is correspondingly less robust). Training artifacts and cached data persist in Docker volumes.

To run preprocessing/training outside Docker, install `backend/requirements.txt`, then:

```sh
python -m backend.train --sample-size 20000
```

You can instead provide a CSV with `text` and `sentiment` columns using `--csv path/to/file.csv`; accepted sentiment values include negative/positive and 0/4 or 0/1. Cleaned data is written to `data/processed/`.

## Data processing

`backend/preprocessing.py` normalizes Unicode, removes URLs and user mentions, retains hashtag words, strips HTML/control characters, collapses whitespace, removes empty rows and exact duplicate texts, validates labels, and performs a stratified train/validation split. It deliberately does not remove stopwords, punctuation, or emojis: those carry sentiment information. BERT's own tokenizer creates input IDs and attention masks; labels are encoded as integers.

## API

- `GET /health` — service/model status
- `POST /predict` — `{ "text": "..." }`
- `POST /predict/batch` — `{ "texts": ["...", "..."] }`
- `GET /api/trends?hours=24` — recent prediction counts and time buckets

Predictions are kept in process memory for the live dashboard and reset when the API restarts. Connect a social platform's authorized streaming API to `/predict` for live ingestion; the app does not scrape social networks or include platform credentials.

## Deploy to AWS EC2

1. Launch an EC2 instance with enough memory/disk for BERT and Docker (a GPU instance is optional). Attach an IAM role if pulling from ECR or writing to S3.
2. Install Docker and the Compose plugin on the instance. Clone this repository and set `.env` values. For production, build the model artifact ahead of time and store/mount it at `artifacts/model` or use an ECR image containing it.
3. Restrict the EC2 security group to trusted sources. Expose the dashboard through a reverse proxy with TLS; keep API/admin access private where possible. Do not expose an unauthenticated prediction service publicly.
4. Run `docker compose up --build -d`; configure a domain and HTTPS reverse proxy (for example, Nginx + Let's Encrypt) and monitor `/health` and container logs.

The included Compose stack is for development and single-instance demonstration. It does not provision AWS infrastructure, TLS, authentication, persistent prediction storage, or autoscaling.
