def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import time
import math
import random
from pathlib import Path
from copy import deepcopy
import csv
import numpy as np
import torch

from classification.src.config import args
from classification.src import classifiers
import classification.src.utils_classify as u_cls
from classification.src.dataset import MyDataset, process_dataset

import utils as u


class Runner:
    def __init__(self, config=None):

        self.config = config
        # Set the random seed manually for reproducibility.
        if self.config['seed'] > 0:
            np.random.seed(self.config['seed'])
            torch.manual_seed(self.config['seed'])
            torch.cuda.manual_seed(self.config['seed'])
            random.seed(self.config['seed'])

        self.train_step = 0
        self.n_samples_total = 0
        self.init_hours = 0
        self.epoch = 0
        self.not_improving = 0
        self.best_val_loss = float("inf")
        self.best_result = 0
        self.init_time = time.time()
        self.once = True

        if not self.config['debug']:
            os.makedirs(self.config['output_dir'])

        log_path = self.config['output_dir'] / 'log.log'
        self.log = u.Logger(log_path, log_=not self.config['debug'])
        self.log(f"Run started at {self.config['start_time']}")

        # DATASET
        process_dataset(self.config)

        self.trn_loader = MyDataset('trn', self.config['feature_lengths'], self.config['feature_dims'])
        trn_data_size = len(self.trn_loader)        
        self.trn_loader = torch.utils.data.DataLoader(self.trn_loader, batch_size=self.config['batch_size'], 
                                                        shuffle=True, num_workers=self.config['num_workers'],
                                                        pin_memory=self.config['pin_memory'])

        self.val_loader = MyDataset('val', self.config['feature_lengths'], self.config['feature_dims'])
        val_data_size = len(self.val_loader)
        self.val_loader = torch.utils.data.DataLoader(self.val_loader, batch_size=self.config['batch_size'], 
                                                        shuffle=False, num_workers=self.config['num_workers'],
                                                        pin_memory=self.config['pin_memory'])
        
        self.tst_loader = MyDataset('tst', self.config['feature_lengths'], self.config['feature_dims'])
        tst_data_size = len(self.tst_loader)
        self.tst_loader = torch.utils.data.DataLoader(self.tst_loader, batch_size=self.config['batch_size'], 
                                                        shuffle=False, num_workers=self.config['num_workers'],
                                                        pin_memory=self.config['pin_memory'])

        self.initialize_model()
        
        if args.pos_weight < 0:
            pos_weight = None
        else:
            pos_weight = torch.ones((args.n_labels)) * args.pos_weight
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.config['device'])

        # Print log
        log_str = '=' * 140 + '\n'
        for k, v in self.config.items():
            log_str += '    - {} : {}'.format(k, v) + '\n' 

        log_str += '=' * 140 + '\n' 
        n_params = sum([p.nelement() for p in self.model.parameters()])
        n_params_trainable = sum([p.nelement() for p in self.model.parameters() if p.requires_grad])
        log_str += '# of trainable parameters: {:,} / {:,}\n'.format(n_params_trainable, n_params)
        log_str += f"Dataset sizes: TRN: {trn_data_size}, VAL: {val_data_size}, TST: {tst_data_size} \n"
        if self.config['device'] == torch.device("cuda"):
            log_str += "Using GPU" + '\n' 
        else:
            log_str += "Using CPU" + '\n' 
        if self.config['amp']:
            log_str += "Using automatic mixed precision" + '\n' 
        else:
            log_str += "Using float32" + '\n' 

        self.log(log_str)


    def initialize_model(self):
        if self.config['restart_dir']:
            # Load existing model
            self.model_config = torch.load(os.path.join(self.config['restart_dir'], "config.pt"))
        else:
            self.model_config = deepcopy(self.config)

        self.model = classifiers.init_model(self.model_config)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config['amp'])
        self.train_step = 0
        self.init_hours = 0
        self.hours_total = 0
        self.epoch = 0
        self.n_samples_total = 0
        csv_input = None

        if self.config['restart_dir']:
            model_fp = os.path.join(self.config['restart_dir'], 'model.pt')
            optimizer_fp = os.path.join(self.config['restart_dir'], 'optimizer.pt')
            stats_fp = os.path.join(self.config['restart_dir'], 'stats.pt')
            scaler_fp = os.path.join(self.config['restart_dir'], 'scaler.pt')

            self.model.load_state_dict(
                torch.load(model_fp, map_location=lambda storage, loc: storage))
            self.log(f"Model loaded from {model_fp}")

            
            if self.model_config['dropout'] != self.config['dropout']:
                classifiers.set_dropout(self.model, self.config['dropout'])
                self.log(f"Dropout changed to {self.config['dropout']}")

            # Optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
            if os.path.exists(optimizer_fp):
                try:
                    ckpt = torch.load(optimizer_fp, map_location=self.config['device'])
                    self.optimizer.load_state_dict(ckpt)
                    self.log(f"Optimizer loaded from {optimizer_fp}")
                except:
                    pass
            else:
                self.log('Optimizer was not saved. Start from scratch.')
            if self.config['overwrite_lr']:
                # New learning rate
                for p in self.optimizer.param_groups:
                    p['lr'] = self.config['lr']
                self.log(f"New learning rate: {self.config['lr']}")
            # Scaler
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.config['amp'])
            if os.path.exists(scaler_fp) and not self.config['reset_scaler']:
                try:
                    self.scaler.load_state_dict(torch.load(scaler_fp, map_location=self.config['device']))
                except:
                    pass            
            # Stats
            self.train_step = 0
            try:
                stats = torch.load(stats_fp)
                # self.train_step = stats["step"]
                self.init_hours = stats["hour"]
                self.epoch = stats["epoch"]
                self.n_samples_total = stats["sample"]
                self.hours_total = self.init_hours
            except:
                # self.train_step = 0
                self.init_hours = 0
                self.epoch = 0
                self.n_samples_total = 0
                self.hours_total = 0
            csv_input = os.path.join(self.config['restart_dir'], 'performance.csv')

        performance_keys = [
            "epoch", "step", "hour", "lr", 
            "trn_loss", 'trn_map', 'trn_precision', 'trn_recall',  
            "val_loss", 'val_map', 'val_precision', 'val_recall', 
            ]

        self.csv_writer = u.CsvWriter(os.path.join(self.config['output_dir'], "performance.csv"),
            performance_keys, input_path=csv_input, debug=self.config['debug'] or self.config['test_only'])
        
    def process_batch(self, x, target, is_training):
        with torch.set_grad_enabled(is_training):
            with torch.cuda.amp.autocast(enabled=self.config['amp']):
                y = self.model(x)
                loss = self.criterion(y, target)
        
        return loss, y

    def train(self):
        # Turn on training mode which enables dropout.
        self.model.train()

        train_loss_accumulated = 0
        n_samples_total = 0
        interval_start = time.time()
        once = True
        train_loss = np.nan
        train_outputs, train_targets = [], []

        elapsed_data, elapsed_network = 0, 0

        while True:
            t0_data = time.time()
            for input_, target, video_name in self.trn_loader:
                
                t1_data = time.time()
                elapsed_data += t1_data - t0_data

                # Evaluate non-trained model first
                if (self.train_step % self.config['eval_step'] == 0) and (not self.config['skip_first_eval'] or self.train_step > 0):
                    # Evaluate model
                    interval_start = time.time()
                    val_loss, metrics, n_samples = self.evaluate(self.val_loader)
                    metrics['loss'] = val_loss
                    elapsed_interval = time.time() - interval_start
                    n_batches_eval = len(self.val_loader) if self.config['max_eval_step'] <= 0 and \
                         self.config['max_eval_step'] < len(self.val_loader) else self.config['max_eval_step']
                    n_samples_eval = n_batches_eval * self.config['batch_size']
                    elapsed_total = time.time() - self.init_time
                    hours_elapsed = elapsed_total / 3600.0
                    self.hours_total = self.init_hours + hours_elapsed
                    lr = self.optimizer.param_groups[0]['lr']
                    ms_per_batch = elapsed_interval * 1000 / n_batches_eval
                    ms_per_sample = elapsed_interval * 1000 / n_samples_eval

                    if (metrics['map_macro'] > self.best_result):
                        self.not_improving = 0
                        self.best_result = metrics['map_macro']

                        if not self.config['debug']:
                            dir_ = os.path.join(self.config['output_dir'], "best_map_model")
                            os.makedirs(dir_, exist_ok=True)
                            self.save_model(dir_)

                    else:
                        if self.not_improving >= self.config['patience']:
                            self.log(f"Validation performance didn't improve for {self.not_improving} steps. Ending training.")
                            return train_loss, self.best_result
                        self.not_improving += self.config['eval_step']

                    csv_dict = {"epoch": self.epoch, "step": self.train_step, "hour": self.hours_total, "lr": lr, 
                        "val_loss": val_loss,  'val_map': metrics['map_macro'], 
                        'val_precision': metrics['precision_macro'], 'val_recall': metrics['recall_macro']
                        }
                    self.csv_writer.update(csv_dict)
                    if not self.config['debug']:
                        u_cls.plot_performance(os.path.join(self.config['output_dir'], "performance.csv"), 
                                               title=self.config['note'])

                    self.log('-' * 140)
                    log_str = 'VAL: step {:>8d} | now: {} | {:>3.1f} h | {:>4.0f} ms/batch' \
                            ' | {:>4.0f} ms/sample | n_samples: {:8.0f} | loss {:7.4f}' \
                            ' | map {:5.2f} | P {:5.2f} | R {:5.2f}'.format(
                        self.train_step, time.strftime("%d-%m-%H:%M"), self.hours_total, 
                        ms_per_batch, ms_per_sample, n_samples, val_loss, metrics['map_macro']*100,
                        metrics['precision_macro']*100, metrics['recall_macro']*100)
                        
                    self.log(log_str)
                    self.log('-' * 140)
                    interval_start = time.time()
                    self.model.train()
                
                t0_network = time.time()

                loss, train_output = self.process_batch(input_, target, is_training=True)
                train_outputs.append(deepcopy(train_output.detach().cpu()))
                train_targets.append(deepcopy(target.detach().cpu()))

                loss_val = loss.item()
                loss /= self.config['accumulate_step']

                self.scaler.scale(loss).backward()
                if self.train_step % self.config['accumulate_step'] == 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.config['clip'] > 0 and self.config['model'] != 'mlp':
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.model.zero_grad()

                n_samples = len(video_name)
                self.n_samples_total += n_samples
                if not math.isnan(loss_val):
                    train_loss_accumulated += n_samples * loss_val
                    n_samples_total += n_samples
       
                if (self.train_step % self.config['log_step'] == 0) and n_samples_total > self.config['batch_size']:
                    train_loss = train_loss_accumulated / n_samples_total
                    train_outputs = torch.cat(train_outputs, 0)
                    train_targets = torch.cat(train_targets, 0)
                    metrics = u_cls.calculate_metrics(train_outputs, train_targets)
                    metrics['trn_loss'] = train_loss
                    train_outputs, train_targets = [], []

                    elapsed_total = time.time() - self.init_time
                    elapsed_interval = time.time() - interval_start
                    hours_elapsed = elapsed_total / 3600.0
                    self.hours_total = self.init_hours + hours_elapsed
                    lr = self.optimizer.param_groups[0]['lr']
                    # now = time.strftime("%d-%m-%H:%M")
                    ms_per_batch = elapsed_interval * 1000 / self.config['log_step']
                    ms_per_sample = elapsed_interval * 1000 / n_samples_total
                    
                    log_str = 'TRN: step {:>8d} | now: {} | {:>3.1f} h | {:>4.0f} ms/batch' \
                            ' | {:>4.0f} ms/sample | n_samples: {:8.0f} | loss {:7.4f}' \
                            ' | map {:5.2f} | P {:5.2f} | R {:5.2f}'.format(
                        self.train_step, time.strftime("%d-%m-%H:%M"), self.hours_total, 
                        ms_per_batch, ms_per_sample, n_samples, train_loss, metrics['map_macro']*100,
                        metrics['precision_macro']*100, metrics['recall_macro']*100)

                    self.log(log_str)

                    csv_dict = {"epoch": self.epoch, "step": self.train_step, "hour": self.hours_total, "lr": lr, 
                        "trn_loss": train_loss, 'trn_map': metrics['map_macro'], 
                        'trn_precision': metrics['precision_macro'], 'trn_recall': metrics['recall_macro']
                    }
                    self.csv_writer.update(csv_dict)
                    if not self.config['debug']:
                        u_cls.plot_performance(os.path.join(self.config['output_dir'], "performance.csv"), 
                                               title=self.config['note'])
                    if torch.cuda.is_available() and once:
                        self.log(u.memory())
                        self.log("-" * 100)
                    once = False
                    
                    train_loss_accumulated = 0
                    n_samples_total = 0

                    interval_start = time.time() 

                self.train_step += 1
                if self.train_step >= self.config['max_step']:
                    return train_loss, self.best_result
                
                t0_data = time.time()
                
            self.epoch += 1
            if self.train_step >= self.config['max_step']:
                return train_loss, self.best_result
            

    def evaluate(self, loader):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        n_samples_total, loss_accumulated = 0, 0.
        val_outputs, val_targets = [], []
        with torch.no_grad():
            for i, (input_, target, video_name) in enumerate(loader):
                if self.config['max_eval_step'] > 0 and i >= self.config['max_eval_step']:
                    break
                loss, val_output = self.process_batch(input_, target, is_training=False)
                val_outputs.append(deepcopy(val_output.detach().cpu()))
                val_targets.append(deepcopy(target.detach().cpu()))
                loss_val = loss.item()
                if loss_val != float('nan'):
                    n_samples = len(video_name)
                    loss_accumulated += n_samples * loss_val
                    n_samples_total += n_samples
                else:
                    self.log(f"Val loss is NaN for {video_name}")

            if n_samples_total == 0:
                val_loss_average = float('nan')
                metrics = None
            else:
                val_loss_average = loss_accumulated / n_samples_total
                val_outputs = torch.cat(val_outputs, 0)
                val_targets = torch.cat(val_targets, 0)
                val_outputs = torch.nn.functional.sigmoid(val_outputs)
                
                metrics = u_cls.calculate_metrics(val_outputs, val_targets)

            return val_loss_average, metrics, n_samples_total

    def save_model(self, dir_):
        os.makedirs(dir_, exist_ok=True)
        model_fp = os.path.join(dir_, 'model.pt')
        torch.save(self.model.state_dict(), model_fp)
        optimizer_fp = os.path.join(dir_, 'optimizer.pt')
        torch.save(self.optimizer.state_dict(), optimizer_fp)
        scaler_fp = os.path.join(dir_, 'scaler.pt')
        torch.save(self.scaler.state_dict(), scaler_fp)
        stats_fp = os.path.join(dir_, 'stats.pt')
        torch.save({"step": self.train_step, "hour": self.hours_total, "epoch": self.epoch,
                    "sample": self.n_samples_total}, stats_fp)
        config_fp = os.path.join(dir_, "config.pt")
        torch.save(self.config, config_fp)

    def run(self):

        # Loop over epochs.
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            if not self.config['test_only']:       
                train_loss, best_result = self.train()
                self.log('-' * 140)
                self.log(f'End of training. \nBest validation mAP: {best_result*100:.2f}')
            self.log('-' * 140)
            self.log('\n--- TESTING LATEST MODEL ---')
            _, metrics_latest, _ = self.evaluate(self.tst_loader)
            self.log(f"mAP: {metrics_latest['map_macro']*100:.2f} | P: {metrics_latest['precision_macro']*100:.2f} | R: {metrics_latest['recall_macro']*100:.2f}")
            
            if not self.config['debug']:
                self.log('\n--- TESTING BEST MAP MODEL ---')
                model_fp = self.config['output_dir'] / 'best_map_model' / 'model.pt'
                stats_fp = self.config['output_dir'] / 'best_map_model' / 'stats.pt'
                stats = torch.load(stats_fp) 
                self.model.load_state_dict(
                    torch.load(model_fp, map_location=lambda storage, loc: storage))
                self.log(f"Model loaded from {model_fp}, step {stats['step']}")
                _, metrics_map, _ = self.evaluate(self.tst_loader)
                self.log(f"mAP: {metrics_map['map_macro']*100:.2f} | P: {metrics_map['precision_macro']*100:.2f} | R: {metrics_map['recall_macro']*100:.2f}")
            
            self.log('\nEnd of testing')

            if not self.config['debug']:
                # Save best result
                results_csv_path = Path('classification/output/results.csv')
                results_dict = {
                    'note': self.config['note'], 
                    'mAP': metrics_map['map_macro'] * 100, 
                    'P': metrics_map['precision_macro'] * 100, 
                    'R': metrics_map['recall_macro'] * 100, 
                    'experiment': self.config['start_time']
                    }
                write_header = not results_csv_path.exists()
                with open(results_csv_path, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=results_dict.keys())
                    if write_header:
                        writer.writeheader()
                    writer.writerow(results_dict)

        except KeyboardInterrupt:
            self.log('-' * 140)
            self.log('Exiting from training early')


if __name__ == "__main__":
    if args.restart_dir != None:
        # Use configuration of pretrained model
        model_config = torch.load(os.path.join(args.restart_dir, "config.pt"))
        # Except certain keys
        new_config = vars(args)
        overwrite_keys = ('device', 'output_dir', 'test_only', 'restart_dir', 'note')
        for key in overwrite_keys:
            model_config[key] = new_config[key]
        # Add missing ones
        for key, value in new_config.items():
            if key not in model_config.keys():
                model_config[key] = value
    else:
        model_config = vars(args)
    runner = Runner(config=model_config)
    runner.run()