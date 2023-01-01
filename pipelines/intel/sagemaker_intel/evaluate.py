import os
import json
import tarfile

import pytorch_lightning as pl
from pathlib import Path

from model import LitResnet
from dataset import IntelDataModule


ml_root = Path("/opt/ml")
model_artifacts = ml_root / "processing" / "model"
dataset_dir = ml_root / "processing" / "test"


def eval_model(trainer, model, datamodule):
    test_res = trainer.test(model, datamodule)[0]

    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {
                "value": test_res["test/acc"],
                "standard_deviation": "0",
            },
            "confusion_matrix" : {
                "0" : {
                    "0" : test_res["test/confusion_matrix/00"],
                    "1" : test_res["test/confusion_matrix/01"],
                    "2" : test_res["test/confusion_matrix/02"],
                    "3" : test_res["test/confusion_matrix/03"],
                    "4" : test_res["test/confusion_matrix/04"],
                    "5" : test_res["test/confusion_matrix/05"]
                },
                "1" : {
                    "0" : test_res["test/confusion_matrix/10"],
                    "1" : test_res["test/confusion_matrix/11"],
                    "2" : test_res["test/confusion_matrix/12"],
                    "3" : test_res["test/confusion_matrix/13"],
                    "4" : test_res["test/confusion_matrix/14"],
                    "5" : test_res["test/confusion_matrix/15"]
                },
                "2" : {
                    "0" : test_res["test/confusion_matrix/20"],
                    "1" : test_res["test/confusion_matrix/21"],
                    "2" : test_res["test/confusion_matrix/22"],
                    "3" : test_res["test/confusion_matrix/23"],
                    "4" : test_res["test/confusion_matrix/24"],
                    "5" : test_res["test/confusion_matrix/25"]
                },
                "3" : {
                    "0" : test_res["test/confusion_matrix/30"],
                    "1" : test_res["test/confusion_matrix/31"],
                    "2" : test_res["test/confusion_matrix/32"],
                    "3" : test_res["test/confusion_matrix/33"],
                    "4" : test_res["test/confusion_matrix/34"],
                    "5" : test_res["test/confusion_matrix/35"]
                },
                "4" : {
                    "0" : test_res["test/confusion_matrix/40"],
                    "1" : test_res["test/confusion_matrix/41"],
                    "2" : test_res["test/confusion_matrix/42"],
                    "3" : test_res["test/confusion_matrix/43"],
                    "4" : test_res["test/confusion_matrix/44"],
                    "5" : test_res["test/confusion_matrix/45"]
                },
                "5" : {
                    "0" : test_res["test/confusion_matrix/50"],
                    "1" : test_res["test/confusion_matrix/51"],
                    "2" : test_res["test/confusion_matrix/52"],
                    "3" : test_res["test/confusion_matrix/53"],
                    "4" : test_res["test/confusion_matrix/54"],
                    "5" : test_res["test/confusion_matrix/55"]
                }
            }
        }
    }
    
    eval_folder = ml_root / "processing" / "evaluation"
    eval_folder.mkdir(parents=True, exist_ok=True)
    
    out_path = eval_folder / "evaluation.json"
    
    print(f":: Writing to {out_path.absolute()}")
    
    with out_path.open("w") as f:
        f.write(json.dumps(report_dict))
        
        
if __name__ == '__main__':
    
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    datamodule = IntelDataModule(
        train_data_dir=dataset_dir.absolute(),
        test_data_dir=dataset_dir.absolute(),
        num_workers=os.cpu_count()
    )
    datamodule.setup()
    
    model = LitResnet.load_from_checkpoint(checkpoint_path="last.ckpt")
    
    trainer = pl.Trainer(
        accelerator="auto",
    )
    
    print(":: Evaluating Model")
    eval_model(trainer, model, datamodule)
