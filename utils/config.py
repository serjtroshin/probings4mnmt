from dataclasses import dataclass


@dataclass
class DataConfig:
    data_dir: str
    valid_rep: str
    valid_labels: str = None
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
        data_dir="data/",
        valid_labels="mnli_valid_enligh_labels.npy",
        valid_rep="mnli_valid_enligh_rep.npy",
    ),
    "cordInv_probing": DataConfig(
        data_dir="/ivi/ilps/projects/ltl-mt/probings/",
        train_labels="cordInv_training_enligh_labels.npy",
        train_rep="cordInv_training_enligh_rep.npy",
        valid_labels="cordInv_valid_enligh_labels.npy",
        valid_rep="cordInv_valid_enligh_rep.npy",
    ),
    "projecting_rep_el": DataConfig(
        data_dir="/ivi/ilps/projects/ltl-mt/probings/",
        train_rep="mnli_valid_greek_rep.npy",
        valid_rep="mnli_valid_greek_rep.npy",
    ),
    "projecting_rep_en": DataConfig(
        data_dir="/ivi/ilps/projects/ltl-mt/probings/",
        train_rep="mnli_training_enligh_rep.npy",
        valid_rep="mnli_valid_enligh_rep.npy",
    ),
}


@dataclass
class ModelConfig:
    probe_dir: str


model_configs ={
    "cordInv": ModelConfig(probe_dir='models/cordInv_valid_enligh_labels.npy')
}
