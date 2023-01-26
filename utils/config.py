from dataclasses import dataclass


@dataclass
class DataConfig:
    data_dir: str
    english_labels: str
    english_rep: str


data_configs = {
    "bert_english": DataConfig(
        data_dir="/ivi/ilps/projects/ltl-mt/probings/",
        english_labels="mnli_valid_enligh_labels.npy",
        english_rep="mnli_valid_enligh_rep.npy",
    ),
}
