import json
import omegaconf
import hydra
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from generate_samples import GenerateTextSamplesCallback


def train(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)
    os.environ["WANDB_CACHE_DIR"] = "/work3/s174159/LLM_Thesis/cache_dir"
    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id=0,
        early_stopping=False,
        no_repeat_ngram_size=0,
        # cache_dir=conf.cache_dir,
        # revision=conf.model_revision,
        # use_auth_token=True if conf.use_auth_token else None,
    )
    print("***************************")
    print(conf.output_dir)

    tokenizer_kwargs = {
        "use_fast": conf.use_fast_tokenizer,
        "additional_special_tokens": [
            "<obj>",
            "<subj>",
            "<triplet>",
            "<head>",
            "</head>",
            "<tail>",
            "</tail>",
        ],  # Here the tokens for head and tail are legacy and only needed if finetuning over the public REBEL checkpoint, but are not used. If training from scratch, remove this line and uncomment the next one.
        #         "additional_special_tokens": ['<obj>', '<subj>', '<triplet>'],
    }

    tokenizer = AutoTokenizer.from_pretrained(
        conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
        **tokenizer_kwargs,
    )

    if conf.dataset_name.split("/")[-1] == "CoNLL04_typed.py":
        tokenizer.add_tokens(
            ["<peop>", "<org>", "<other>", "<loc>"], special_tokens=True
        )
    elif conf.dataset_name.split("/")[-1] == "nyt_typed.py":
        tokenizer.add_tokens(["<loc>", "<org>", "<per>"], special_tokens=True)
    elif conf.dataset_name.split("/")[-1] == "DocRED_typed.py":
        tokenizer.add_tokens(
            ["<loc>", "<misc>", "<per>", "<num>", "<time>", "<org>"],
            special_tokens=True,
        )
    else:
        path = conf.entities
        json_entities = open(path, "r")
        entities = json.load(json_entities)
        print(entities)
        tokenizer.add_tokens(list(entities.values()), special_tokens=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_name_or_path,
        config=config,
    )
    # if not conf.finetune:
    model.resize_token_embeddings(len(tokenizer))

    # data module declaration
    pl_data_module = BasePLDataModule(conf, tokenizer, model)

    # main module declaration
    pl_module = BasePLModule(conf, config, tokenizer, model)
    pl_module = pl_module.load_from_checkpoint(
        checkpoint_path=conf.checkpoint_path,
        config=config,
        tokenizer=tokenizer,
        model=model,
    )
    # pl_module.hparams.predict_with_generate = True
    pl_module.hparams.test_file = pl_data_module.conf.test_file
    # trainer
    trainer = pl.Trainer(
        gpus=conf.gpus,
    )
    # Manually run prep methods on DataModule
    pl_data_module.prepare_data()
    pl_data_module.setup()

    trainer.test(pl_module, test_dataloaders=pl_data_module.test_dataloader())


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
