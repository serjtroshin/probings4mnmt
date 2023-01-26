from dataclasses import dataclass


@dataclass
class DataConfig:
    data_dir: str
    train_labels: str
    train_rep: str
    valid_labels: str
    valid_rep: str


data_configs = {
    "bert_english":
        DataConfig(
        data_dir="/ivi/ilps/projects/ltl-mt/probings/",
            train_labels="mnli_training_enligh_labels.npy",
            train_rep="mnli_training_enligh_rep.npy",
            valid_labels="mnli_valid_enligh_labels.npy",
            valid_rep="mnli_valid_enligh_rep.npy",
        ),

    "debug":
        DataConfig(
        data_dir="/ivi/ilps/projects/ltl-mt/probings/",
            train_labels="mnli_valid_enligh_labels.npy",
            train_rep="mnli_valid_enligh_rep.npy",
            valid_labels="mnli_valid_enligh_labels.npy",
            valid_rep="mnli_valid_enligh_rep.npy",
        ),

    "cordInv_probing":
        DataConfig(
        data_dir="/ivi/ilps/projects/ltl-mt/probings/",
            train_labels="cordInv_training_enligh_labels.npy",
            train_rep="cordInv_training_enligh_rep.npy",
            valid_labels="cordInv_valid_enligh_labels.npy",
            valid_rep="cordInv_valid_enligh_rep.npy",
        ),

    "projecting_rep_el":
        DataConfig(
            data_dir="/ivi/ilps/projects/ltl-mt/probings/",
            train_labels="mnli_valid_greek_labels.npy",
            train_rep="mnli_valid_greek_rep.npy",
            valid_labels="mnli_valid_greek_labels.npy",
            valid_rep="mnli_valid_greek_rep.npy",
        ),

    "projecting_rep_en":
        DataConfig(
            data_dir="/ivi/ilps/projects/ltl-mt/probings/",
            train_labels="mnli_training_enligh_labels.npy",
            train_rep="mnli_training_enligh_rep.npy",
            valid_labels="mnli_valid_enligh_labels.npy",
            valid_rep="mnli_valid_enligh_rep.npy",
        ),
}
