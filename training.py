"""
Training dictionaries
"""

import copy
import torch as t
from .dictionary import AutoEncoder
from .buffer import SkipActivationBuffer, SkipActivationBufferWithAddition
import os
from tqdm import tqdm
from .trainers.standard import StandardTrainer
import wandb
import json
# from .evaluation import evaluate
from .evaluation_tinystories import loss_recovered


def get_loss_recovered(model, data, ae, submodule_in, submodule_out):
    num_batches = 256
    total_base_loss = 0
    total_recovered_loss = 0
    total_zero_loss = 0

    for _ in range(num_batches):
        base_loss, recovered_loss, zero_loss = loss_recovered(
            data.text_batch(batch_size=16),
            model=model,
            submodule_in=submodule_in,
            submodule_out=submodule_out,
            dictionary=ae,
            max_len=128,
            normalize_batch=False,
            io="out",
            tracer_args = {},
        )
        total_base_loss += base_loss.cpu().item()
        total_recovered_loss += recovered_loss.cpu().item()
        total_zero_loss += zero_loss.cpu().item()

    avg_base_loss = total_base_loss / num_batches
    avg_recovered_loss = total_recovered_loss / num_batches
    avg_zero_loss = total_zero_loss / num_batches
    return (avg_recovered_loss - avg_zero_loss) / (avg_base_loss - avg_zero_loss), avg_base_loss, avg_recovered_loss, avg_zero_loss


def log_stats(
    model,
    data,
    trainers,
    step: int,
    act: t.Tensor,
    use_wandb: bool,
    activations_split_by_head: bool,
    transcoder: bool,
):
    
    log = {}
    with t.no_grad():
        # quick hack to make sure all trainers get the same x
        # TODO make this less hacky
        if len(act) == 2: 
            act_in, act_out = act
            z1 = act_in.clone()
            z2 = act_out.clone()
        else: 
            act_in, act_out, act_add = act
            z1 = act_in.clone()
            z2 = act_out.clone()
            z3 = act_add.clone()
        for i, trainer in enumerate(trainers):
            if len(act) == 2:
                act_in = z1.clone()
                act_out = z2.clone()
                act_add = None
            else: 
                act_in = z1.clone()
                act_out = z2.clone()
                act_add = z3.clone()

            if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                raise Exception
                act = act[..., i, :]
            trainer_name = f'{trainer.config["wandb_name"]}-{i}'
            if not transcoder:
                act_out, act_hat, f, losslog = trainer.loss(act_in, act_out, act_add, step=step, model=model, logging=True)  # act is x

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

                # fraction of variance explained
                total_variance = t.var(act_out, dim=0).sum()
                residual_variance = t.var(act_out - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log[f"{trainer_name}/frac_variance_explained"] = frac_variance_explained.item()

                # loss recovered
                frac_loss_rec, base_loss, recovered_loss, zero_loss = get_loss_recovered(
                    model, data, trainer.ae, data.submodule_in, data.submodule_out
                )
                log[f"{trainer_name}/frac_loss_rec"] = frac_loss_rec
                log[f"{trainer_name}/base_loss"] = base_loss
                log[f"{trainer_name}/recovered_loss"] = recovered_loss
                log[f"{trainer_name}/zero_loss"] = zero_loss
                
            else:  # transcoder
                act_out, act_hat, f, losslog = trainer.loss(act_in, act_out, act_add, step=step, logging=True)  # act is x, y

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

                # fraction of variance explained
                # TODO: adapt for transcoder
                # total_variance = t.var(x, dim=0).sum()
                # residual_variance = t.var(x - x_hat, dim=0).sum()
                # frac_variance_explained = (1 - residual_variance / total_variance)
                # log[f'{trainer_name}/frac_variance_explained'] = frac_variance_explained.item()

            # log parameters from training
            log.update({f"{trainer_name}/{k}": v for k, v in losslog.items()})
            log[f"{trainer_name}/l0"] = l0
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                log[f"{trainer_name}/{name}"] = value

    if use_wandb:
        wandb.log(log, step=step)


def trainSAE(
        model,
        data, 
        eval_data,
        trainer_configs,
        use_wandb = False,
        wandb_entity = "",
        wandb_project = "",
        wandb_group = "",
        wandb_name = "",
        steps=None,
        save_steps=None,
        save_dir=None, # use {run} to refer to wandb run
        log_steps=None,
        activations_split_by_head=False, # set to true if data is shape [batch, pos, num_head, head_dim/resid_dim]
        transcoder=False,
):
    """
    Train SAEs using the given trainers
    """

    if trainer_configs[0]["position_config"]:
        position_config_path = trainer_configs[0]["position_config"].replace("[", ".").replace("]", "")
        save_dir = os.path.join("outputs", position_config_path)

    trainers = []
    for config in trainer_configs:
        trainer = config['trainer']
        del config['trainer']
        trainers.append(
            trainer(
                **config,
            )
        )

    if log_steps is not None:
        if use_wandb:
            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                group=wandb_group,
                name=wandb_name,
                config={f'{trainer.config["wandb_name"]}-{i}' : trainer.config for i, trainer in enumerate(trainers)}
            )
            # process save_dir in light of run name
            if save_dir is not None:
                save_dir = save_dir.format(run=wandb.run.name)

    # make save dirs, export config
    if save_dir is not None:
        save_dirs = [os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            # save config
            config = {'trainer' : trainer.config}
            try:
                config['buffer'] = data.config
            except: 
                pass
            with open(os.path.join(dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]
    
    for step, act in enumerate(tqdm(data, total=steps)):
        if steps is not None and step >= steps:
            break
        
        if not isinstance(data, SkipActivationBuffer) and not isinstance(data, SkipActivationBufferWithAddition):
            act = (act, act)
        
        # logging
        if log_steps is not None and step % log_steps == 0:
            log_stats(model, eval_data, trainers, step, act, use_wandb, activations_split_by_head, transcoder)
            
        # saving
        if save_steps is not None and step % save_steps == 0:
            for dir, trainer in zip(save_dirs, trainers):
                if dir is not None:
                    if not os.path.exists(os.path.join(dir, "checkpoints")):
                        os.mkdir(os.path.join(dir, "checkpoints"))
                    t.save(
                        trainer.ae.state_dict(), 
                        os.path.join(dir, "checkpoints", f"ae_{step}.pt")
                        )
                    
        # training
        for trainer in trainers:
            trainer.update(step, act, model)
    
    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if save_dir is not None:
            t.save(trainer.ae.state_dict(), os.path.join(save_dir, "ae.pt"))

    # End the wandb run
    if log_steps is not None and use_wandb:
        wandb.finish()