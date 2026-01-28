# Dataset manifests

Place each dataset under `datasets/<dataset_id>/` and provide a JSONL manifest named `test.jsonl`.
Each line must include:

- `audio_path`: path to the audio file (relative to the manifest file is OK)
- `text`: reference transcript

Example:

{"audio_path": "audio/0001.wav", "text": "..."}
{"audio_path": "audio/0002.wav", "text": "..."}

Update `config/datasets.json` if you want to change dataset names or paths.
