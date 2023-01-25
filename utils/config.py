from dataclasses import dataclass

@dataclass
class DataConfig:
    data_dir: str = "/ivi/ilps/projects/ltl-mt/probings/"
    english_labels: str = "mnli_valid_enligh_labels.npy"
    english_rep: str = "mnli_valid_enligh_rep.npy"
