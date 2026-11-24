Captures placed here are **not** used by `train.py` (it only reads the top level of
`pcap_files/`). Use this folder for unseen captures to demo `predict.py`:

    python model_scripts/predict.py pcap_files/holdout/<file>.pcapng --label 0
