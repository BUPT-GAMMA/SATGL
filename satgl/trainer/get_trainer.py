from satgl.trainer.trainer import Trainer

def get_trainer(config):
    device = config["device"]
    early_stopping = config["early_stopping"]
    eval_step = config["eval_step"]
    epochs = config["epochs"]
    save_model = config["save_model"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    log_file = config["log_file"]
    valid_metric = config["valid_metric"]

    return Trainer(
        device=device,
        early_stopping=early_stopping,
        eval_step=eval_step,
        epochs=epochs,
        save_model=save_model,
        lr=lr,
        weight_decay=weight_decay,
        log_file=log_file,
        valid_metric=valid_metric
    )
