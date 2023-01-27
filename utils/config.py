from dataclasses import dataclass


@dataclass
class DataConfig:
    data_dir: str
    valid_labels: str
    valid_rep: str
    train_labels: str = None
    train_rep: str = None


data_configs = {
    "bert_english": DataConfig(
        data_dir="/ivi/ilps/projects/ltl-mt/probings/",
        train_labels="mnli_training_enligh_labels.npy",
        train_rep="mnli_training_enligh_rep.npy",
        valid_labels="mnli_valid_enligh_labels.npy",
        valid_rep="mnli_valid_enligh_rep.npy",
    ),
    "bert_greek": DataConfig(
        data_dir="/ivi/ilps/projects/ltl-mt/probings/",
        valid_labels="mnli_valid_greek_labels.npy",
        valid_rep="mnli_valid_greek_rep.npy",
    ),
    "debug": DataConfig(
        data_dir="/ivi/ilps/projects/ltl-mt/probings/",
        train_labels="mnli_valid_enligh_labels.npy",
        train_rep="mnli_valid_enligh_rep.npy",
        valid_labels="mnli_valid_enligh_labels.npy",
        valid_rep="mnli_valid_enligh_rep.npy",
    ),
}
