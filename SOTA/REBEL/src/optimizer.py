import json
import omegaconf
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from generate_samples import GenerateTextSamplesCallback
import os
import wandb


def train(conf: omegaconf.DictConfig) -> None:
    with wandb.init():
        tmp = wandb.config
        # Extract project name from configuration
    project_name = conf.get("wandb_project_name")
    # Update conf with the values from wandb.config
    conf.learning_rate = tmp["learning_rate"]
    conf.max_grad_norm = tmp["max_grad_norm"]
    conf.num_train_epochs = tmp["num_train_epochs"]
    conf.train_batch_size = tmp["train_batch_size"]
    conf.warmup_steps = tmp["warmup_steps"]
    conf.weight_decay = tmp["weight_decay"]

    # Dynamically generate a name for the run based on hyperparameters
    run_name = f"lr_{conf.learning_rate}_epochs_{conf.num_train_epochs}_bs_{conf.train_batch_size}"

    pl.seed_everything(conf.seed)
    os.environ["WANDB_CACHE_DIR"] = "/work3/s174159/LLM_Thesis/cache_dir"

    # Initialize W&B logger
    wandb_logger = WandbLogger(project=project_name, name=run_name)

    # Log hyperparameters
    wandb_logger.log_hyperparams(conf)

    print("run_name: ", run_name)

    # Print hyperparameters to console

    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id=0,
        early_stopping=False,
        no_repeat_ngram_size=0,
        dropout=conf.dropout,
        forced_bos_token_id=None,
    )

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
        ],
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
        with open(path, "r") as json_entities:
            entities = json.load(json_entities)
            tokenizer.add_tokens(list(entities.values()), special_tokens=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_name_or_path,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))

    # data module declaration
    pl_data_module = BasePLDataModule(conf, tokenizer, model)

    # main module declaration
    pl_module = BasePLModule(conf, config, tokenizer, model)

    callbacks_store = []

    if conf.apply_early_stopping:
        callbacks_store.append(
            EarlyStopping(
                monitor=conf.monitor_var,
                mode=conf.monitor_var_mode,
                patience=conf.patience,
            )
        )

    callbacks_store.append(
        ModelCheckpoint(
            monitor=conf.monitor_var,
            dirpath=f"experiments/{conf.model_name}",
            save_top_k=conf.save_top_k,
            verbose=True,
            save_last=True,
            mode=conf.monitor_var_mode,
        )
    )
    callbacks_store.append(GenerateTextSamplesCallback(conf.samples_interval))
    callbacks_store.append(LearningRateMonitor(logging_interval="step"))

    # trainer
    trainer = pl.Trainer(
        gpus=conf.gpus,
        accumulate_grad_batches=conf.gradient_acc_steps,
        gradient_clip_val=conf.gradient_clip_value,
        val_check_interval=conf.val_check_interval,
        callbacks=callbacks_store,
        max_steps=conf.max_steps,
        precision=conf.precision,
        amp_level=conf.amp_level,
        logger=wandb_logger,
        resume_from_checkpoint=conf.checkpoint_path,
        limit_val_batches=conf.val_percent_check,
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)


def setup_wandb_sweep(sweep_config_path: str):
    # Load the sweep configuration from a file
    with open(sweep_config_path, "r") as file:
        sweep_config_data = json.load(file)

    project_name = sweep_config_data["project_name"]
    sweep_config = sweep_config_data["sweep_config"]

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    # Run the sweep
    wandb.agent(sweep_id, function=main)


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    # Path to the W&B sweep configuration file
    sweep_config_path = (
        "/work3/s174159/LLM_Thesis/SOTA/REBEL/src/config/sweep_config.json"
    )

    # Uncomment the following line if you want to run the sweep
    setup_wandb_sweep(sweep_config_path)

    # Comment out the following line if running a sweep
    # main()
