from datasets import get_dataset_config_names, get_dataset_split_names
from huggingface_hub import HfApi


def main():
    api = HfApi()
    for query in ["WenetSpeech", "wenet_speech", "wenetspeech"]:
        print(f"QUERY {query}")
        for ds in api.list_datasets(search=query, limit=20):
            print(ds.id)

    candidates = [
        "lmms-lab/WenetSpeech",
        "Wenetspeech/WenetSpeech",
        "wenet-e2e/WenetSpeech",
        "DynamicSuperb/WenetSpeech",
    ]
    for dataset_id in candidates:
        print(f"DATASET {dataset_id}")
        try:
            print("configs", get_dataset_config_names(dataset_id)[:20])
            print("splits", get_dataset_split_names(dataset_id))
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
